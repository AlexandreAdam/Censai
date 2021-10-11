import numpy as np
import os
from datetime import datetime
from scripts.train_rim_unet_wvae import main
import copy
import pandas as pd

# total number of slurm workers detected
# defaults to 1 if not running under SLURM
N_WORKERS = int(os.getenv('SLURM_ARRAY_TASK_COUNT', 1))

# this worker's array index. Assumes slurm array job is zero-indexed
# defaults to zero if not running under SLURM
THIS_WORKER = int(os.getenv('SLURM_ARRAY_TASK_ID', 0)) ## it starts from 1!!

DATE = datetime.now().strftime("%y%m%d%H%M%S")

RIM_HPARAMS = [
    "adam",
    "steps",
    "kappalog",
    "kappa_normalize",
    "source_link",
    "kappa_init",
    "source_init",
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
    "kappa_upsampling_interpolation",
    "kappa_kernel_l2_amp",
    "kappa_bias_l2_amp",
    "kappa_kernel_l1_amp",
    "kappa_bias_l1_amp",
    "kappa_activation",
    "kappa_alpha",
    "kappa_initializer",
    "kappa_batch_norm",
    "kappa_dropout_rate"
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
    "source_upsampling_interpolation",
    "source_kernel_l2_amp",
    "source_bias_l2_amp",
    "source_kernel_l1_amp",
    "source_bias_l1_amp",
    "source_activation",
    "source_alpha",
    "source_initializer",
    "source_batch_norm",
    "source_dropout_rate"
]

EXTRA_PARAMS = [
    "total_items",
    "optimizer",
    "seed",
    "batch_size",
    "initial_learning_rate",
    "decay_rate",
    "decay_steps",
    "delay",
    "time_weights"
]

PARAMS_NICKNAME = {
    "total_items": "TI",
    "optimizer": "O",
    "seed": "",
    "batch_size": "B",
    "delay": "D",
    "time_weights": "TW",

    "kappa_filters": "KF",
    "kappa_filter_scaling": "KFS",
    "kappa_kernel_size": "KK",
    "kappa_layers": "KL",
    "kappa_block_conv_layers": "KBCL",
    "kappa_strides": "KS",
    "kappa_upsampling_interpolation": "KBU",
    "kappa_resampling_kernel_size": "KRK",
    "kappa_gru_kernel_size": "KGK",
    "kappa_kernel_l2_amp": "KKl2",
    "kappa_kernel_l1_amp": "KKl1",
    "kappa_bias_l2_amp": "KBl2",
    "kappa_bias_l1_amp": "KBl1",
    "kappa_activation": "KA",
    "kappa_batch_norm": "KBN",
    "kappa_dropout_rate": "Kdr",

    "source_filters": "SF",
    "source_filter_scaling": "SFS",
    "source_kernel_size": "SK",
    "source_layers": "SL",
    "source_block_conv_layers": "SBCL",
    "source_strides": "SS",
    "source_upsampling_interpolation": "SBU",
    "source_resampling_kernel_size": "SRK",
    "source_gru_kernel_size": "SGK",
    "source_kernel_l2_amp": "SKl2",
    "source_kernel_l1_amp": "SKl1",
    "source_bias_l2_amp": "SBl2",
    "source_bias_l1_amp": "SBl1",
    "source_activation": "SA",
    "source_batch_norm": "SBN",
    "source_dropout_rate": "Sdr",

    "steps": "TS",
    "kappalog": "KaL",
    "kappa_normalize": "KaN",
    "adam": "A",
    "source_link": "Sli",
    "source_init": "Sini",
    "kappa_init": "Kini",
}


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
        nicknames = []
        params = []
        for p in RIM_HPARAMS + SOURCE_MODEL_HPARAMS + KAPPA_MODEL_HPARAMS + EXTRA_PARAMS:
            if isinstance(args_dict[p], list):
                if len(args_dict[p]) > 1:
                    # this way, numpy does not cast int to int64 or float to float32
                    args_dict[p] = args_dict[p][np.random.choice(range(len(args_dict[p])))]
                    nicknames.append(PARAMS_NICKNAME[p])
                    params.append(args_dict[p])
                else:
                    args_dict[p] = args_dict[p][0]
        param_str = "_" + "_".join([f"{nickname}{param}" for nickname, param in zip(nicknames, params)])
        args_dict.update({"logname": args.logname_prefixe + "_" + f"{gridsearch_id:03d}" + param_str + "_" + DATE})
        yield new_args


def exhaustive_grid_search(args):
    """
    Lexicographic ordering of given parameter lists, up to n_models deep.
    """
    from itertools import product
    grid_params = []
    for p in RIM_HPARAMS + SOURCE_MODEL_HPARAMS + KAPPA_MODEL_HPARAMS + EXTRA_PARAMS:
        if isinstance(vars(args)[p], list):
            if len(vars(args)[p]) > 1:
                grid_params.append(vars(args)[p])
    lexicographically_ordered_grid_params = product(*grid_params)
    for gridsearch_id, lex in enumerate(lexicographically_ordered_grid_params):
        if gridsearch_id >= args.n_models:
            return
        new_args = copy.deepcopy(args)
        args_dict = vars(new_args)
        nicknames = []
        params = []
        i = 0
        for p in RIM_HPARAMS + SOURCE_MODEL_HPARAMS + KAPPA_MODEL_HPARAMS + EXTRA_PARAMS:
            if isinstance(args_dict[p], list):
                if len(args_dict[p]) > 1:
                    args_dict[p] = lex[i]
                    i += 1
                    nicknames.append(PARAMS_NICKNAME[p])
                    params.append(args_dict[p])
                else:
                    args_dict[p] = args_dict[p][0]
        param_str = "_" + "_".join([f"{nickname}{param}" for nickname, param in zip(nicknames, params)])
        args_dict.update({"logname": args.logname_prefixe + "_" + f"{gridsearch_id:03d}" + param_str + "_" + DATE})
        yield new_args


def distributed_strategy(args):
    gridsearch_args = list(single_instance_args_generator(args))
    for gridsearch_id in range((THIS_WORKER - 1), len(gridsearch_args), N_WORKERS):
        run_args = gridsearch_args[gridsearch_id]
        history, best_score = main(run_args)
        params_dict = {k: v for k, v in vars(run_args).items() if k in RIM_HPARAMS + SOURCE_MODEL_HPARAMS + KAPPA_MODEL_HPARAMS + EXTRA_PARAMS}
        params_dict.update({
            "experiment_id": run_args.logname,
            "cost": history["cost"][-1],
            "chi_squared": history["chi_squared"][-1],
            "kappa_cost": history["kappa_cost"][-1],
            "source_cost": history["source_cost"][-1],
            "best_score": best_score
        })
        # Save hyperparameters and scores in shared csv for this gridsearch
        df = pd.DataFrame(params_dict, index=[gridsearch_id])
        grid_csv_path = os.path.join(os.getenv("CENSAI_PATH"), "results", f"{args.logname_prefixe}.csv")
        this_run_csv_path = os.path.join(os.getenv("CENSAI_PATH"), "results", f"{run_args.logname}.csv")
        if not os.path.exists(grid_csv_path):
            mode = "w"
            header = True
        else:
            mode = "a"
            header = False
        df.to_csv(grid_csv_path, header=header, mode=mode)
        pd.DataFrame(history).to_csv(this_run_csv_path)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--n_models",                   default=10,     type=int,       help="Models to train")
    parser.add_argument("--strategy",                   default="uniform",              help="Allowed startegies are 'uniform' and 'exhaustive'.")
    parser.add_argument("--kappa_first_stage_vae",      required=True)
    parser.add_argument("--kappa_second_stage_vae",     default=None)
    parser.add_argument("--source_first_stage_vae",     required=True)
    parser.add_argument("--source_second_stage_vae",    default=None)

    # Physical model hyperparameter
    parser.add_argument("--forward_method",         default="conv2d",               help="One of ['conv2d', 'fft', 'unet']. If the option 'unet' is chosen, the parameter "
                                                                                         "'--raytracer' must be provided and point to model checkpoint directory.")
    parser.add_argument("--raytracer",              default=None,                   help="Path to raytracer checkpoint dir if method 'unet' is used.")

    parser.add_argument("--steps",                  default=16, nargs="+", type=int,            help="Number of time steps of RIM")
    parser.add_argument("--adam",                   default=0,  nargs="+",  type=int,           help="ADAM update for the log-likelihood gradient.")
    parser.add_argument("--kappalog",               action="store_true")
    parser.add_argument("--kappa_normalize",        action="store_true")
    parser.add_argument("--source_link",            default="identity",  nargs="+",             help="One of 'exp', 'source' or 'identity' (default).")
    parser.add_argument("--kappa_init",             default=1e-1, nargs="+",  type=float,       help="Initial value of kappa for RIM")
    parser.add_argument("--source_init",            default=1e-3, nargs="+",  type=float,       help="Initial value of source for RIM")

    # Kappa model hyperparameters
    parser.add_argument("--kappa_filters",                  default=32, nargs="+",      type=int)
    parser.add_argument("--kappa_filter_scaling",           default=1, nargs="+",       type=float)
    parser.add_argument("--kappa_kernel_size",              default=3, nargs="+",       type=int)
    parser.add_argument("--kappa_layers",                   default=2, nargs="+",       type=int)
    parser.add_argument("--kappa_block_conv_layers",        default=2, nargs="+",       type=int)
    parser.add_argument("--kappa_strides",                  default=2, nargs="+",       type=int)
    parser.add_argument("--kappa_bottleneck_kernel_size",   default=None, nargs="+",    type=int)
    parser.add_argument("--kappa_bottleneck_filters",       default=None, nargs="+",    type=int)
    parser.add_argument("--kappa_resampling_kernel_size",   default=None, nargs="+",    type=int)
    parser.add_argument("--kappa_gru_kernel_size",          default=None, nargs="+",    type=int)
    parser.add_argument("--kappa_upsampling_interpolation", action="store_true")
    parser.add_argument("--kappa_kernel_l2_amp",            default=1e-4, nargs="+",    type=float)
    parser.add_argument("--kappa_bias_l2_amp",              default=1e-4, nargs="+",    type=float)
    parser.add_argument("--kappa_kernel_l1_amp",            default=1e-4, nargs="+",    type=float)
    parser.add_argument("--kappa_bias_l1_amp",              default=1e-4, nargs="+",    type=float)
    parser.add_argument("--kappa_activation",               default="leaky_relu", nargs="+")
    parser.add_argument("--kappa_alpha",                    default=0.1, nargs="+",     type=float)
    parser.add_argument("--kappa_initializer",              default="glorot_normal")
    parser.add_argument("--kappa_batch_norm",               default=0,   nargs="+",    type=int)
    parser.add_argument("--kappa_dropout_rate",             default=None, nargs="+",   type=float)

    # Source model hyperparameters
    parser.add_argument("--source_filters",                  default=32, nargs="+",     type=int)
    parser.add_argument("--source_filter_scaling",           default=1, nargs="+",      type=float)
    parser.add_argument("--source_kernel_size",              default=3, nargs="+",      type=int)
    parser.add_argument("--source_layers",                   default=2, nargs="+",      type=int)
    parser.add_argument("--source_block_conv_layers",        default=2, nargs="+",      type=int)
    parser.add_argument("--source_strides",                  default=2, nargs="+",      type=int)
    parser.add_argument("--source_bottleneck_kernel_size",   default=None, nargs="+",   type=int)
    parser.add_argument("--source_bottleneck_filters",       default=None, nargs="+",   type=int)
    parser.add_argument("--source_resampling_kernel_size",   default=None, nargs="+",   type=int)
    parser.add_argument("--source_gru_kernel_size",          default=None,   nargs="+",  type=int)
    parser.add_argument("--source_upsampling_interpolation", action="store_true")
    parser.add_argument("--source_kernel_l2_amp",           default=0, nargs="+",   type=float)
    parser.add_argument("--source_bias_l2_amp",             default=0, nargs="+",   type=float)
    parser.add_argument("--source_kernel_l1_amp",           default=0, nargs="+",   type=float)
    parser.add_argument("--source_bias_l1_amp",             default=0, nargs="+",   type=float)
    parser.add_argument("--source_activation",               default="leaky_relu", nargs="+")
    parser.add_argument("--source_alpha",                    default=0.1, nargs="+",    type=float)
    parser.add_argument("--source_initializer",              default="glorot_normal")
    parser.add_argument("--source_batch_norm",               default=0,   nargs="+",    type=int)
    parser.add_argument("--source_dropout_rate",             default=None, nargs="+",   type=float)

    # Training set params
    parser.add_argument("--batch_size",             default=1,      nargs="+",  type=int,       help="Number of images in a batch. ")
    parser.add_argument("--total_items",            required=True,  nargs="+",  type=int,       help="Total images in an epoch.")
    parser.add_argument("--image_pixels",           default=512,                type=int,       help="Number of pixels on a side of the lensed image")
    parser.add_argument("--image_fov",              default=20,                 type=float,     help="Field of view of lensed image in arcsec")
    parser.add_argument("--kappa_fov",              default=18,                 type=float,     help="Field of view of kappa map (in lens plane), in arcsec")
    parser.add_argument("--source_fov",             default=3,                  type=float,     help="Field of view of source map, in arcsec")
    parser.add_argument("--noise_rms",              default=1e-2,               type=float,     help="RMS of white noise added to lensed image")
    parser.add_argument("--psf_sigma",              default=0.08,               type=float,     help="Size, in arcseconds, of the gaussian blurring PSF")

    # Optimization params
    parser.add_argument("-e", "--epochs",           default=10,                 type=int,       help="Number of epochs for training.")
    parser.add_argument("--delay",                  default=2,       nargs="+", type=int,       help="Number of epochs kappa model trains alone")
    parser.add_argument("--optimizer",              default="Adam",  nargs="+",                 help="Class name of the optimizer (e.g. 'Adam' or 'Adamax')")
    parser.add_argument("--initial_learning_rate",  default=1e-3,    nargs="+", type=float,     help="Initial learning rate.")
    parser.add_argument("--decay_rate",             default=1.,      nargs="+", type=float,     help="Exponential decay rate of learning rate (1=no decay).")
    parser.add_argument("--decay_steps",            default=1000,    nargs="+", type=int,       help="Decay steps of exponential decay of the learning rate.")
    parser.add_argument("--staircase",              action="store_true",                        help="Learning rate schedule only change after decay steps if enabled.")
    parser.add_argument("--clipping",               action="store_true",                        help="Clip backprop gradients between -10 and 10.")
    parser.add_argument("--patience",               default=np.inf,             type=int,       help="Number of step at which training is stopped if no improvement is recorder.")
    parser.add_argument("--tolerance",              default=0,                  type=float,     help="Current score <= (1 - tolerance) * best score => reset patience, else reduce patience.")
    parser.add_argument("--max_time",               default=np.inf,             type=float,     help="Time allowed for the training, in hours.")
    parser.add_argument("--time_weights",           default="uniform", nargs="+",               help="uniform: w_t=1 for all t, linear: w_t~t, quadratic: w_t~t^2")

    # logs
    parser.add_argument("--logdir",                  default="None",                help="Path of logs directory. Default if None, no logs recorded.")
    parser.add_argument("--logname_prefixe",         default="RIMUnetwVAE",         help="If name of the log is not provided, this prefix is prepended to the date")
    parser.add_argument("--model_dir",               default="None",                help="Path to the directory where to save models checkpoints.")
    parser.add_argument("--checkpoints",             default=10,    type=int,       help="Save a checkpoint of the models each {%} iteration.")
    parser.add_argument("--max_to_keep",             default=3,     type=int,       help="Max model checkpoint to keep.")
    parser.add_argument("--n_residuals",             default=1,     type=int,       help="Number of residual plots to save. Add overhead at the end of an epoch only.")

    # Make sure each model train on the same dataset
    parser.add_argument("--seed",                   default=42, nargs="+",  type=int, help="Random seed for numpy and tensorflow.")

    # Keep these as default, they need to be in Namespace but we dont use them for this script
    parser.add_argument("--model_id",                   default="None",              help="Start training from previous "
                                                                                          "checkpoint of this model if provided")
    parser.add_argument("--json_override", default=None, nargs="+", help="A json filepath that will override every command line parameters. Useful for reproducibility")
    args = parser.parse_args()
    distributed_strategy(args)
