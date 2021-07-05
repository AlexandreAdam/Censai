import numpy as np
import os
from datetime import datetime
from scripts.train_raytracer import main
import copy

# total number of slurm workers detected
# defaults to 1 if not running under SLURM
N_WORKERS = int(os.getenv('SLURM_ARRAY_TASK_COUNT', 1))

# this worker's array index. Assumes slurm array job is zero-indexed
# defaults to zero if not running under SLURM
THIS_WORKER = int(os.getenv('SLURM_ARRAY_TASK_ID', 0)) ## it starts from 1!!

DATE = datetime.now().strftime("%y-%m-%d_%H")

# Parameters on which we do a grid search. Fixed parameter are commented out.
RAYTRACER_HPARAMS = [
    # "pixels",
    "filter_scaling",
    "layers",
    "block_conv_layers",
    "kernel_size",
    "filters",
    "strides",
    "bottleneck_filters",
    "resampling_kernel_size",
    "upsampling_interpolation",
    "kernel_regularizer_amp",
    "activation",
    # "initializer",
    # "kappalog",
    # "normalize",
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
    else:
        raise NotImplementedError(f"{args.strategy} not in ['uniform', 'exhaustive']")


def uniform_grid_search(args):
    for gridsearch_id in range(1, args.n_models + 1):
        new_args = copy.deepcopy(args)
        args_dict = vars(new_args)
        args_dict.update({"logname": args.logname_prefixe + "_" + DATE + "_" + f"{gridsearch_id:03d}"})
        for p in RAYTRACER_HPARAMS + EXTRA_PARAMS:
            if isinstance(args_dict[p], list):
                args_dict[p] = np.random.choice(args_dict[p], size=1)[0]
        yield new_args


def exhaustive_grid_search(args):
    """
    Lexicographic ordering of given parameter lists, up to n_models deep.
    """
    indexes = []
    for p in RAYTRACER_HPARAMS + EXTRA_PARAMS:
        if isinstance(vars(args)[p], list):
            indexes.append(0)
    for gridsearch_id in range(1, args.n_models + 1):
        new_args = copy.deepcopy(args)
        args_dict = vars(new_args)
        args_dict.update({"logname": args.logname_prefixe + "_" + DATE + "_" + f"{gridsearch_id:03d}"})
        i = 0
        roll = True
        for p in RAYTRACER_HPARAMS + EXTRA_PARAMS:
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
    parser.add_argument("--strategy",               default="uniform",              help="Allowed startegies are 'uniform' and 'exhaustive'.")

    parser.add_argument("--datasets",                   required=True,  nargs="+",   help="Datasets to use, paths that contains tfrecords of dataset. User can provide multiple "
                                                                                          "directories to mix datasets")
    parser.add_argument("--compression_type",           default=None,                help="Compression type used to write data. Default assumes no compression.")

    # Model hyper parameters
    parser.add_argument("--pixels",                         required=True,                  type=int,     help="Size of input tensors, need to match dataset size!!")
    parser.add_argument("--kernel_size",                    default=3, nargs="+",           type=int,     help="Main kernel size of U-net")
    parser.add_argument("--filters",                        default=32, nargs="+",          type=int,     help="Number of filters of conv layers")
    parser.add_argument("--filter_scaling",                 default=1, nargs="+",           type=float,   help="Scaling of the number of filters at each layers (1=no scaling)")
    parser.add_argument("--layers",                         default=2, nargs="+",           type=int,     help="Number of layers of Unet (number of downsampling and upsampling")
    parser.add_argument("--block_conv_layers",              default=2, nargs="+",           type=int,     help="Number of convolutional layers in a unet layer")
    parser.add_argument("--strides",                        default=2, nargs="+",           type=int,     help="Strides of downsampling and upsampling layers")
    parser.add_argument("--bottleneck_filters",             default=None, nargs="+",        type=int,     help="Number of filters of bottleneck layers. Default None, use normal scaling of filters.")
    parser.add_argument("--resampling_kernel_size",         default=None, nargs="+",        type=int,     help="Kernel size of downsampling and upsampling layers. None, use same kernel size as the others.")
    parser.add_argument("--upsampling_interpolation",       action="store_true",                          help="True: Use Bilinear interpolation for upsampling, False use Fractional Striding Convolution")
    parser.add_argument("--kernel_regularizer_amp",         default=1e-3, nargs="+",        type=float,   help="l2 regularization on weights")
    parser.add_argument("--kappalog",                       action="store_true",                          help="Input is log of kappa")
    parser.add_argument("--normalize",                      action="store_true",                          help="Normalize log of kappa with max and minimum values defined in definitions.py")
    parser.add_argument("--activation",                     default="linear", nargs="+",                  help="Non-linearity of layers")
    parser.add_argument("--initializer",                    default="glorot_uniform",                     help="Weight initializer")

    # Training set params
    parser.add_argument("-b", "--batch_size",               default=10,     type=int,               help="Number of images in a batch")
    parser.add_argument("--train_split",                    default=0.8,    type=float,             help="Fraction of the training set")
    parser.add_argument("--total_items",                    required=True,  nargs="+", type=int,    help="Total images in an epoch.")
    # ... for tfrecord dataset
    parser.add_argument("--num_parallel_reads",             default=10,     type=int,               help="TFRecord dataset number of parallel reads when loading data")
    parser.add_argument("--cache_file",                     default=None,                           help="Path to cache file, useful when training on server. Use ${SLURM_TMPDIR}/cache")
    parser.add_argument("--cycle_length",                   default=4,      type=int,               help="Number of files to read concurrently.")
    parser.add_argument("--block_length",                   default=1,      type=int,               help="Number of example to read from each files.")

    # Logs
    parser.add_argument("--logdir",                         default="None",                         help="Path of logs directory.")
    parser.add_argument("--logname",                        default=None,                           help="Name of the logs, default is 'RT_' + date")
    parser.add_argument("--logname_prefixe",                default="RayTracer",                    help="If name of the log is not provided, this prefix is prepended to the date")
    parser.add_argument("--model_dir",                      default="None",                         help="Directory where to save model weights")
    parser.add_argument("--checkpoints",                    default=10,     type=int,               help="Save a checkpoint of the models each {%} iteration")
    parser.add_argument("--max_to_keep",                    default=3,      type=int,               help="Max model checkpoint to keep")
    parser.add_argument("--n_residuals",                    default=1,      type=int,               help="Number of residual plots to save. Add overhead at the end of an epoch only.")
    parser.add_argument("--profile",                        action="store_true",                    help="If added, we will profile the last training step of the first epochAnalyticalPhysicalModel")
    parser.add_argument("--source_fov",                     default=3,      type=float,             help="Source fov for lens residuals")
    parser.add_argument("--source_w",                       default=0.1,    type=float,             help="Width of gaussian of source gaussian")
    parser.add_argument("--psf_sigma",                      default=0.04,   type=float,             help="Sigma of PSF for lens resiudal")

    # Optimization params
    parser.add_argument("-e", "--epochs",                   default=10,     type=int,               help="Number of epochs for training.")
    parser.add_argument("--initial_learning_rate",          default=1e-3,   type=float,             help="Initial learning rate.")
    parser.add_argument("--decay_rate",                     default=1.,     type=float,             help="Exponential decay rate of learning rate (1=no decay).")
    parser.add_argument("--decay_steps",                    default=1000,   type=int,               help="Decay steps of exponential decay of the learning rate.")
    parser.add_argument("--clipping",                       action="store_true",                    help="Clip backprop gradients between -10 and 10")
    parser.add_argument("--patience",                       default=np.inf, type=int,               help="Number of step at which training is stopped if no improvement is recorder")
    parser.add_argument("--tolerance",                      default=0,      type=float,             help="Current score <= (1 - tolerance) * best score => reset patience, else reduce patience.")
    parser.add_argument("--track_train",                    action="store_true")

    # Make sure each model train on the same dataset
    parser.add_argument("--seed",                   default=42,   type=int,       help="Random seed for numpy and tensorflow.")

    # Keep these as default, they need to be in Namespace but we dont use them for this script
    parser.add_argument("--model_id",                   default="None",              help="Start training from previous "
                                                                                          "checkpoint of this model if provided")
    parser.add_argument("--load_checkpoint",            default="best",              help="One of 'best', 'lastest' or the specific checkpoint index")
    parser.add_argument("--json_override",                  default=None,             help="A json filepath that will override every command line parameters. "
                                                                                           "Useful for reproducibility")

    args = parser.parse_args()
    distributed_strategy(args)