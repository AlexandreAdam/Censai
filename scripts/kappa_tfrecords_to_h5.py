import tensorflow as tf
from censai.data.kappa_tng import decode_train, decode_shape
import h5py, json, os, glob
from tqdm import tqdm
import numpy as np


def main(args):
    with open(os.path.join(args.dataset, "script_params.json")) as f:
        params = json.load(f)
    compression_type = params["compression_type"]
    len_dataset = params["len_dataset"]
    files = glob.glob(os.path.join(args.dataset, "*.tfrecords"))
    files = tf.data.Dataset.from_tensor_slices(files)
    dataset = files.interleave(lambda x: tf.data.TFRecordDataset(x, compression_type=compression_type),
                               block_length=1, num_parallel_calls=tf.data.AUTOTUNE)
    for pixels in dataset.map(decode_shape):
        break
    dataset = dataset.map(decode_train).batch(args.batch_size)
    with h5py.File(args.output_path, "w") as hf:
        hf.create_dataset("kappa", shape=[len_dataset, pixels, pixels], compression="gzip")
        for i, kappa in tqdm(enumerate(dataset)):
            hf["kappa"][i*args.batch_size:(i+1)*args.batch_size] = kappa.numpy().squeeze().astype(np.float32)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--batch_size", default=10, type=int)

    args = parser.parse_args()
    main(args)