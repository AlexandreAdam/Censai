import tensorflow as tf
from censai.models import VAE
from censai.data.lenses_tng_v3 import decode_train, decode_physical_model_info
from censai.data.cosmos import decode_image
from censai import PhysicalModelv2
import h5py
import os, glob, json
import numpy as np


def main(args):
    files = glob.glob(os.path.join(args.dataset, "*.tfrecords"))
    files = tf.data.Dataset.from_tensor_slices(files)
    dataset = files.interleave(lambda x: tf.data.TFRecordDataset(x, compression_type=args.compression_type),
                               block_length=1, num_parallel_calls=tf.data.AUTOTUNE)
    for physical_params in dataset.map(decode_physical_model_info):
        break
    dataset = dataset.map(decode_train)

    files = glob.glob(os.path.join(args.source_dataset, "*.tfrecords"))
    files = tf.data.Dataset.from_tensor_slices(files)
    source_dataset = files.interleave(lambda x: tf.data.TFRecordDataset(x, compression_type=args.compression_type),
                               block_length=1, num_parallel_calls=tf.data.AUTOTUNE)
    source_dataset = source_dataset.map(decode_image).shuffle(10000).batch(args.sample_size)

    with open(os.path.join(args.kappa_vae, "model_hparams.json"), "r") as f:
        kappa_vae_hparams = json.load(f)
    kappa_vae = VAE(**kappa_vae_hparams)
    ckpt1 = tf.train.Checkpoint(step=tf.Variable(1), net=kappa_vae)
    checkpoint_manager1 = tf.train.CheckpointManager(ckpt1, args.kappa_vae, 1)
    checkpoint_manager1.checkpoint.restore(checkpoint_manager1.latest_checkpoint).expect_partial()

    # For some reason model was not saved on Narval, waiting on Beluga to open up again or retrain a COSMOS VAE.
    # with open(os.path.join(args.source_vae, "model_hparams.json"), "r") as f:
    #     source_vae_hparams = json.load(f)
    # source_vae = VAE(**source_vae_hparams)
    # ckpt2 = tf.train.Checkpoint(step=tf.Variable(1), net=source_vae)
    # checkpoint_manager2 = tf.train.CheckpointManager(ckpt2, args.source_vae, 1)
    # checkpoint_manager2.checkpoint.restore(checkpoint_manager2.latest_checkpoint).expect_partial()

    phys = PhysicalModelv2(
        pixels=physical_params["pixels"].numpy(),
        kappa_pixels=physical_params["kappa pixels"].numpy(),
        src_pixels=physical_params["src pixels"].numpy(),
        image_fov=physical_params["image fov"].numpy(),
        kappa_fov=physical_params["kappa fov"].numpy(),
        src_fov=physical_params["source fov"].numpy(),
        method="fft"
    )

    # simulate observations
    kappa = kappa_vae.sample(args.sample_size)
    # source = source_vae.sample(args.sample_size)
    for source in source_dataset:
        break
    fwhm = tf.random.normal(shape=[args.sample_size], mean=1.5*phys.image_fov/phys.pixels, stddev=0.5*phys.image_fov/phys.pixels)
    noise_rms = tf.random.normal(shape=[args.sample_size], mean=args.noise_mean, stddev=args.noise_std)
    psf = phys.psf_models(fwhm, cutout_size=20)
    y_vae = phys.noisy_forward(source, kappa, noise_rms, psf)

    with h5py.File(os.path.join(os.getenv("CENSAI_PATH"), "results", args.output_name + ".h5"), 'w') as hf:
        # rank these observations against the dataset with L2 norm
        for i in range(args.sample_size):
            distances = []
            for y_d, _, _, _, _ in dataset:
                distances.append(tf.sqrt(tf.reduce_sum((y_d - y_vae[i][None, ...])**2)).numpy().astype(np.float32))
            k_indices = np.argsort(distances[i])[:args.k]

            g = hf.create_group(f"sample_{i:02d}")
            g.create_dataset(name="matched_source",     shape=[args.k, phys.src_pixels, phys.src_pixels],       dtype=np.float32)
            g.create_dataset(name="matched_kappa",      shape=[args.k, phys.kappa_pixels, phys.kappa_pixels],   dtype=np.float32)
            g.create_dataset(name="matched_obs",        shape=[args.k, phys.pixels, phys.pixels],               dtype=np.float32)
            g.create_dataset(name="matched_psf",        shape=[args.k, 20, 20],                                 dtype=np.float32)
            g.create_dataset(name="matched_noise_rms",  shape=[args.k],                                         dtype=np.float32)
            g.create_dataset(name="obs_L2_distance",    shape=[args.k],                                         dtype=np.float32)

            for rank, j in enumerate(k_indices):
                # fetch back the matched observation
                for y_d, source_d, kappa_d, noise_rms_d, psf_d in dataset.skip(j):
                    break
                # save results
                g["vae_source"] = source[i, ..., 0].numpy().astype(np.float32)
                g["vae_kappa"] = kappa[i, ..., 0].numpy().astype(np.float32)
                g["vae_obs"] = y_vae[i, ..., 0].numpy().astype(np.float32)
                g["vae_psf"] = psf[i, ..., 0].numpy().astype(np.float32)
                g["vae_noise_rms"] = noise_rms[i].numpy().astype(np.float32)
                g["matched_source"][rank] = source_d[0, ..., 0].numpy().astype(np.float32)
                g["matched_kappa"][rank] = kappa_d[0, ..., 0].numpy().astype(np.float32)
                g["matched_obs"][rank] = y_d[0, ..., 0].numpy().astype(np.float32)
                g["matched_noise_rms"][rank] = noise_rms_d.numpy().astype(np.float32)
                g["matched_psf"][rank] = psf_d[0, ..., 0].numpy().astype(np.float32)
                g["obs_L2_distance"][rank] = distances[j]


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--dataset",            required=True)
    parser.add_argument("--kappa_vae",          required=True)
    # parser.add_argument("--source_vae",         required=True)
    parser.add_argument("--source_dataset",     required=True)
    parser.add_argument("--output_name",        required=True)
    parser.add_argument("--compression_type",   default="GZIP")
    parser.add_argument("-k",                   default=50,     type=int)
    parser.add_argument("--sample_size",        default=50,     type=int)
    parser.add_argument("--noise_mean",         default=1e-2,   type=float)
    parser.add_argument("--noise_std",          default=5e-3,   type=float)

    args = parser.parse_args()
    main(args)