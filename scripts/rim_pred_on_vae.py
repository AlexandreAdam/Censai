import tensorflow as tf
from censai.models import Model, VAE
from censai import PhysicalModel, RIM
import h5py, os, json
import numpy as np
from scipy.stats import truncnorm

# total number of slurm workers detected
# defaults to 1 if not running under SLURM
N_WORKERS = int(os.getenv('SLURM_ARRAY_TASK_COUNT', 1))

# this worker's array index. Assumes slurm array job is zero-indexed
# defaults to zero if not running under SLURM
THIS_WORKER = int(os.getenv('SLURM_ARRAY_TASK_ID', 0)) ## it starts from 1!!


def main(args):
    with open(os.path.join(args.kappa_model, "model_hparams.json"), "r") as f:
        kappa_vae_hparams = json.load(f)
    kappa_vae = VAE(**kappa_vae_hparams)
    ckpt1 = tf.train.Checkpoint(step=tf.Variable(1), net=kappa_vae)
    checkpoint_manager1 = tf.train.CheckpointManager(ckpt1, args.kappa_model, 1)
    checkpoint_manager1.checkpoint.restore(checkpoint_manager1.latest_checkpoint).expect_partial()

    with open(os.path.join(args.source_model, "model_hparams.json"), "r") as f:
        source_vae_hparams = json.load(f)
    source_vae = VAE(**source_vae_hparams)
    ckpt2 = tf.train.Checkpoint(step=tf.Variable(1), net=source_vae)
    checkpoint_manager2 = tf.train.CheckpointManager(ckpt2, args.source_model, 1)
    checkpoint_manager2.checkpoint.restore(checkpoint_manager2.latest_checkpoint).expect_partial()

    phys = PhysicalModel(pixels=128, method="fft")

    with open(os.path.join(args.model, "unet_hparams.json")) as f:
        unet_params = json.load(f)
    unet = Model(**unet_params)
    ckpt = tf.train.Checkpoint(net=unet)
    checkpoint_manager = tf.train.CheckpointManager(ckpt, args.model, 1)
    checkpoint_manager.checkpoint.restore(checkpoint_manager.latest_checkpoint).expect_partial()
    with open(os.path.join(args.model, "rim_hparams.json")) as f:
        rim_params = json.load(f)
    rim_params["source_link"] = "relu"
    rim = RIM(phys, unet, **rim_params)

    with h5py.File(os.path.join(os.getenv("CENSAI_PATH"), "results", f"rim_pred_on_vae_{THIS_WORKER:02d}.h5"), "w") as hf:
        hf.create_dataset("observation", shape=[args.total, phys.pixels, phys.pixels])
        hf.create_dataset("kappa", shape=[args.total, phys.kappa_pixels, phys.kappa_pixels])
        hf.create_dataset("source", shape=[args.total, phys.src_pixels, phys.src_pixels])
        hf.create_dataset("observation_pred", shape=[args.total, phys.pixels, phys.pixels])
        hf.create_dataset("kappa_pred", shape=[args.total, phys.pixels, phys.pixels])
        hf.create_dataset("source_pred", shape=[args.total, phys.pixels, phys.pixels])
        hf.create_dataset("noise_rms", shape=[args.total])
        hf.create_dataset("psf_fwhm", shape=[args.total])
        hf.create_dataset("chi_squared", shape=[args.total])
        hf.create_dataset("kappa_mse", shape=[args.total])
        hf.create_dataset("source_mse", shape=[args.total])

        for batch in range(args.total//args.batch_size):
            z = tf.random.normal(shape=[args.batch_size, source_vae.latent_size])
            source = tf.nn.relu(source_vae.decode(z))
            source /= tf.reduce_max(source, axis=(1, 2, 3), keepdims=True)
            z = tf.random.normal(shape=[args.batch_size, kappa_vae.latent_size])
            kappa = 10**kappa_vae.decode(z)
            noise_rms = truncnorm.rvs(1e-3, 0.1, loc=0.02, scale=0.02, size=args.batch_size)
            psf_fwhm = truncnorm.rvs(phys.image_fov/128, 4*phys.image_fov/128, loc=1.5*phys.image_fov/128, scale=0.5*phys.image_fov/128, size=args.batch_size)
            psf = phys.psf_models(psf_fwhm, cutout_size=20)
            obs = phys.noisy_forward(source, kappa, noise_rms, psf)
            source_pred, kappa_pred, chisq = rim.predict(obs, noise_rms, psf)
            source_mse = tf.reduce_mean((source_pred[-1] - source)**2, axis=(1, 2, 3))
            kappa_mse = tf.reduce_mean((kappa_pred[-1] - kappa)**2, axis=(1, 2, 3))
            obs_pred = phys.forward(source_pred[-1], kappa_pred[-1], psf)

            start = batch * args.batch_size
            end = (batch+1) * args.batch_size
            hf["observation"][start:end] = obs.numpy().squeeze().astype(np.float32)
            hf["source"][start:end] = source.numpy().squeeze().astype(np.float32)
            hf["kappa"][start:end] = kappa.numpy().squeeze().astype(np.float32)
            hf["observation_pred"][start:end] = obs_pred.numpy().squeeze().astype(np.float32)
            hf["kappa_pred"][start:end] = kappa_pred[-1].numpy().squeeze().astype(np.float32)
            hf["source_pred"][start:end] = source_pred[-1].numpy().squeeze().astype(np.float32)
            hf["chi_squared"][start:end] = chisq[-1].numpy().squeeze().astype(np.float32)
            hf["source_mse"][start:end] = source_mse.numpy().astype(np.float32)
            hf["kappa_mse"][start:end] = kappa_mse.numpy().astype(np.float32)
            hf["noise_rms"][start:end] = noise_rms.astype(np.float32)
            hf["psf_fwhm"][start:end] = psf_fwhm.astype(np.float32)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--model", required=True, help="Full path to the model checkpoints")
    parser.add_argument("--source_model", required=True, help="Full path to source VAE checkpoints")
    parser.add_argument("--kappa_model", required=True, help="Full path to kappa VAE checkpoints")
    parser.add_argument("--batch_size", default=10, type=int)
    parser.add_argument("--total", default=1000, type=int)

    args = parser.parse_args()
    main(args)