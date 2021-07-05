import numpy as np
import os
from datetime import datetime
from scripts.train_rim_unet import main
import copy

# total number of slurm workers detected
# defaults to 1 if not running under SLURM
N_WORKERS = int(os.getenv('SLURM_ARRAY_TASK_COUNT', 1))

# this worker's array index. Assumes slurm array job is zero-indexed
# defaults to zero if not running under SLURM
THIS_WORKER = int(os.getenv('SLURM_ARRAY_TASK_ID', 0)) ## it starts from 1!!

DATE = datetime.now().strftime("%y-%m-%d_%H")

# Parameters on which we do a grid search. Fixed parameter are commented out.
RIM_HPARAMS = [
    # "adam",
    "steps",
    # "kappalog",
    # "kappa_normalize"
]
SOURCE_MODEL_HPARAMS = [
    "kappa_filters",
    "kappa_filter_scaling",
    "kappa_kernel_size",
    "kappa_layers",
    "kappa_block_conv_layers",
    "kappa_strides",
    "kappa_bottleneck_kernel_size",
    "kappa_bottleneck_filters",
    "kappa_resampling_kernel_size",
    "kappa_gru_kernel_size",
    # "kappa_upsampling_interpolation",
    "kappa_kernel_regularizer_amp",
    "kappa_bias_regularizer_amp",
    "kappa_activation",
    "kappa_alpha",
    # "kappa_initializer"
]
KAPPA_MODEL_HPARAMS = [
    "source_filters",
    "source_filter_scaling",
    "source_kernel_size",
    "source_layers",
    "source_block_conv_layers",
    "source_strides",
    "source_bottleneck_kernel_size",
    "source_bottleneck_filters",
    "source_resampling_kernel_size",
    "source_gru_kernel_size",
    # "source_upsampling_interpolation",
    "source_kernel_regularizer_amp",
    "source_bias_regularizer_amp",
    "source_activation",
    "source_alpha",
    # "source_initializer"
]

EXTRA_PARAMS = [
    "total_items"
]


def single_instance_args_generator(args):
    """

    Args:
        args: Namespace of argument parser

    Returns: A modified deep copy generator of Namespace to be fed to main of train_rim_unet.py

    """
    if args.strategy == "uniform":
        return uniform_grid_search(args)
    elif args.strategy == "exhaustive":
        return exhaustive_grid_search(args)


def uniform_grid_search(args):
    for gridsearch_id in range(1, args.n_models + 1):
        new_args = copy.deepcopy(args)
        args_dict = vars(new_args)
        args_dict.update({"logname": args.logname_prefixe + "_" + DATE + "_" + f"{gridsearch_id:03d}"})
        for p in RIM_HPARAMS + SOURCE_MODEL_HPARAMS + KAPPA_MODEL_HPARAMS + EXTRA_PARAMS:
            if isinstance(args_dict[p], list):
                args_dict[p] = np.random.choice(args_dict[p], size=1)[0]
        yield new_args


def exhaustive_grid_search(args):
    """
    Lexicographic ordering of given parameter lists, up to n_models deep.
    """
    indexes = []
    for p in RIM_HPARAMS + SOURCE_MODEL_HPARAMS + KAPPA_MODEL_HPARAMS + EXTRA_PARAMS:
        if isinstance(vars(args)[p], list):
            indexes.append(0)
    for gridsearch_id in range(1, args.n_models + 1):
        new_args = copy.deepcopy(args)
        args_dict = vars(new_args)
        args_dict.update({"logname": args.logname_prefixe + "_" + DATE + "_" + f"{gridsearch_id:03d}"})
        i = 0
        roll = True
        for p in RIM_HPARAMS + SOURCE_MODEL_HPARAMS + KAPPA_MODEL_HPARAMS + EXTRA_PARAMS:
            if isinstance(args_dict[p], list):
                if roll:
                    if indexes[i] < len(args_dict[p]):
                        args_dict[p] = args_dict[p][indexes[i]]
                        indexes[i] += 1
                        roll = False
                    else:
                        if i + 1 <= len(indexes)-1:
                            indexes[i + 1] += 1
                        args_dict[p] = args_dict[p][0]
                else:
                    args_dict[p] = args_dict[p][0]
                i += 1
        yield new_args


def distributed_strategy(args):
    gridsearch_args = list(single_instance_args_generator(args))
    for gridsearch_id in range((THIS_WORKER - 1), len(gridsearch_args), N_WORKERS):
        main(gridsearch_args[gridsearch_id])


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--n_models",               default=10,     type=int,       help="Models to train")
    # parser.add_argument("--datasets",               required=True,  nargs="+",      help="Path to directories that contains tfrecords of dataset. Can be multiple inputs (space separated)")
    parser.add_argument("--compression_type",       default=None,                   help="Compression type used to write data. Default assumes no compression.")
    parser.add_argument("--strategy",               default="uniform",              help="Allowed startegies are 'uniform' and 'exhaustive'.")

    # Physical model hyperparameter
    parser.add_argument("--forward_method",         default="conv2d",               help="One of ['conv2d', 'fft', 'unet']. If the option 'unet' is chosen, the parameter "
                                                                                         "'--raytracer' must be provided and point to model checkpoint directory.")
    parser.add_argument("--raytracer",              default=None,                   help="Path to raytracer checkpoint dir if method 'unet' is used.")

    parser.add_argument("--steps",                  default=16, nargs="+", type=int, help="Number of time steps of RIM")
    parser.add_argument("--adam",                   action="store_true",            help="ADAM update for the log-likelihood gradient.")
    parser.add_argument("--kappalog",               action="store_true")
    parser.add_argument("--kappa_normalize",        action="store_true")

    # Kappa model hyperparameters
    parser.add_argument("--kappa_filters",                  default=32, nargs="+",      type=int)
    parser.add_argument("--kappa_filter_scaling",           default=1, nargs="+",       type=int)
    parser.add_argument("--kappa_kernel_size",              default=3, nargs="+",       type=int)
    parser.add_argument("--kappa_layers",                   default=2, nargs="+",       type=int)
    parser.add_argument("--kappa_block_conv_layers",        default=2, nargs="+",       type=int)
    parser.add_argument("--kappa_strides",                  default=2, nargs="+",       type=int)
    parser.add_argument("--kappa_bottleneck_kernel_size",   default=None, nargs="+",    type=int)
    parser.add_argument("--kappa_bottleneck_filters",       default=None, nargs="+",    type=int)
    parser.add_argument("--kappa_resampling_kernel_size",   default=None, nargs="+",    type=int)
    parser.add_argument("--kappa_gru_kernel_size",          default=None, nargs="+",    type=int)
    parser.add_argument("--kappa_upsampling_interpolation", action="store_true")
    parser.add_argument("--kappa_kernel_regularizer_amp",   default=1e-4, nargs="+",    type=float)
    parser.add_argument("--kappa_bias_regularizer_amp",     default=1e-4, nargs="+",    type=float)
    parser.add_argument("--kappa_activation",               default="leaky_relu", nargs="+")
    parser.add_argument("--kappa_alpha",                    default=0.1, nargs="+",     type=float)
    parser.add_argument("--kappa_initializer",              default="glorot_normal")

    # Source model hyperparameters
    parser.add_argument("--source_filters",                  default=32, nargs="+",     type=int)
    parser.add_argument("--source_filter_scaling",           default=1, nargs="+",      type=int)
    parser.add_argument("--source_kernel_size",              default=3, nargs="+",      type=int)
    parser.add_argument("--source_layers",                   default=2, nargs="+",      type=int)
    parser.add_argument("--source_block_conv_layers",        default=2, nargs="+",      type=int)
    parser.add_argument("--source_strides",                  default=2, nargs="+",      type=int)
    parser.add_argument("--source_bottleneck_kernel_size",   default=None, nargs="+",   type=int)
    parser.add_argument("--source_bottleneck_filters",       default=None, nargs="+",   type=int)
    parser.add_argument("--source_resampling_kernel_size",   default=None, nargs="+",   type=int)
    parser.add_argument("--source_gru_kernel_size",          default=None, nargs="+",   type=int)
    parser.add_argument("--source_upsampling_interpolation", action="store_true")
    parser.add_argument("--source_kernel_regularizer_amp",   default=1e-4, nargs="+",   type=float)
    parser.add_argument("--source_bias_regularizer_amp",     default=1e-4, nargs="+",   type=float)
    parser.add_argument("--source_activation",               default="leaky_relu", nargs="+")
    parser.add_argument("--source_alpha",                    default=0.1, nargs="+",    type=float)
    parser.add_argument("--source_initializer",              default="glorot_normal")

    # Training set params
    parser.add_argument("-b", "--batch_size",       default=1,      type=int,       help="Number of images in a batch. ")
    parser.add_argument("--train_split",            default=0.8,    type=float,     help="Fraction of the training set.")
    parser.add_argument("--total_items",            required=True,  nargs="+", type=int,  help="Total images in an epoch.")
    # ... for tfrecord dataset
    parser.add_argument("--num_parallel_reads",     default=10,     type=int,       help="TFRecord dataset number of parallel reads when loading data.")
    parser.add_argument("--cache_file",             default=None,                   help="Path to cache file, useful when training on server. Use ${SLURM_TMPDIR}/cache")
    parser.add_argument("--cycle_length",           default=4,      type=int,       help="Number of files to read concurrently.")
    parser.add_argument("--block_length",           default=1,      type=int,       help="Number of example to read from each files.")

    # Optimization params
    parser.add_argument("-e", "--epochs",           default=10,     type=int,       help="Number of epochs for training.")
    parser.add_argument("--initial_learning_rate",  default=1e-3,   type=float,     help="Initial learning rate.")
    parser.add_argument("--decay_rate",             default=1.,     type=float,     help="Exponential decay rate of learning rate (1=no decay).")
    parser.add_argument("--decay_steps",            default=1000,   type=int,       help="Decay steps of exponential decay of the learning rate.")
    parser.add_argument("--staircase",              action="store_true",            help="Learning rate schedule only change after decay steps if enabled.")
    parser.add_argument("--clipping",               action="store_true",            help="Clip backprop gradients between -10 and 10.")
    parser.add_argument("--patience",               default=np.inf, type=int,       help="Number of step at which training is stopped if no improvement is recorder.")
    parser.add_argument("--tolerance",              default=0,      type=float,     help="Current score <= (1 - tolerance) * best score => reset patience, else reduce patience.")

    # logs
    parser.add_argument("--logdir",                  default="None",                help="Path of logs directory. Default if None, no logs recorded.")
    parser.add_argument("--logname_prefixe",         default="RIMUnet512",          help="If name of the log is not provided, this prefix is prepended to the date")
    parser.add_argument("--model_dir",               default="None",                help="Path to the directory where to save models checkpoints.")
    parser.add_argument("--checkpoints",             default=10,    type=int,       help="Save a checkpoint of the models each {%} iteration.")
    parser.add_argument("--max_to_keep",             default=3,     type=int,       help="Max model checkpoint to keep.")
    parser.add_argument("--n_residuals",             default=1,     type=int,       help="Number of residual plots to save. Add overhead at the end of an epoch only.")

    # Make sure each model train on the same dataset
    parser.add_argument("--seed",                   default=42,   type=int,       help="Random seed for numpy and tensorflow.")

    args = parser.parse_args()

    gridsearch_args = list(single_instance_args_generator(args))
    from pprint import pprint
    for a in gridsearch_args:
        pprint(vars(a))
    # distributed_strategy(args)
