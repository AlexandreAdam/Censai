import tensorflow as tf
import numpy as np

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
    # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
        print(e)
from censai.models import Autoencoder
from censai.cosmos_utils import decode as decode_cosmos
from censai.utils import nullwriter
import os
from datetime import datetime



def main(args):
    pass

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--model_id", type=str, default="None",
                        help="Start from this model id checkpoint. None means start from scratch")
    parser.add_argument("--pixels", required=False, default=128, type=int, help="Number of pixels on a side, should be fixed for a given cosmos tfrecord")

    # training params
    # parser.add_argument("-t", "--total_items", default=100, type=int, required=False, help="Total images in an epoch")
    parser.add_argument("-b", "--batch_size", default=100, type=int, required=False, help="Number of images in a batch")
    parser.add_argument("-e", "--epochs", required=False, default=1, type=int, help="Number of epochs for training")
    parser.add_argument("--patience", required=False, default=np.inf, type=float, help="Number of epoch at which "
                                                                "training is stop if no improvement have been made")
    parser.add_argument("--tolerance", required=False, default=0, type=float,
                        help="Percentage [0-1] of improvement required for patience to reset. The most lenient "
                                                        "value is 0 (any improvement reset patience)")

    # hyperparameters
    parser.add_argument("--learning_rate", required=False, default=1e-4, type=float)
    parser.add_argument("--decay_rate", type=float, default=1,
                        help="Decay rate of the exponential decay schedule of the learning rate. 1=no decay")
    parser.add_argument("--decay_steps", type=int, default=100)
    parser.add_argument("--staircase", action="store_true", help="Learning schedule is a staircase "
                                                                 "function if added to arguments")
    parser.add_argument("--noise_rms", required=False, default=1e-3, type=float, help="Pixel value rms of lensed image")
    parser.add_argument("--time_steps", required=False, default=16, type=int, help="Number of time steps of RIM")
    parser.add_argument("--kappalog", required=False, default=True, type=bool)
    parser.add_argument("--adam", required=False, default=True, type=bool,
                        help="ADAM update for the log-likelihood gradient")
    parser.add_argument("--strides", required=False, default=2, type=int, help="Value of the stride parameter in the 3 "
                                                    "downsampling and upsampling layers")
    # logs
    parser.add_argument("--logdir", required=False, default="None",
                        help="Path of logs directory. Default if None, no logs recorded")
    parser.add_argument("--model_dir", required=False, default="None",
                        help="Path to the directory where to save models checkpoints")
    parser.add_argument("--checkpoints", required=False, default=10, type=int,
                        help="Save a checkpoint of the models each {%} iteration")
    parser.add_argument("--max_to_keep", required=False, default=3, type=int,
                        help="Max model checkpoint to keep")
    args = parser.parse_args()

    main(args)
