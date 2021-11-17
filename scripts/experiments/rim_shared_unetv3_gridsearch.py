import numpy as np
import os
from datetime import datetime
from scripts.train_rim_shared_unetv3 import main
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
    "source_link"
]

UNET_MODEL_HPARAMS = [
    "filters",
    "filter_scaling",
    "kernel_size",
    "layers",
    "block_conv_layers",
    "strides",
    "bottleneck_kernel_size",
    "resampling_kernel_size",
    "input_kernel_size",
    "gru_kernel_size",
    "upsampling_interpolation",
    "batch_norm",
    "dropout_rate",
    "kernel_l2_amp",
    "bias_l2_amp",
    "kernel_l1_amp",
    "bias_l1_amp",
    "activation",
    "initializer",
    "gru_architecture",
    "flux_lagrange_multiplier"
]

EXTRA_PARAMS = [
    "total_items",
    "optimizer",
    "seed",
    "batch_size",
    "initial_learning_rate",
    "decay_rate",
    "decay_steps",
    "time_weights",
    "kappa_residual_weights",
    "source_residual_weights"
]


PARAMS_NICKNAME = {
    "total_items": "TI",
    "optimizer": "O",
    "seed": "",
    "batch_size": "B",
    "initial_learning_rate": "lr",
    "decay_rate": "dr",
    "decay_steps": "ds",
    "time_weights": "TW",
    "kappa_residual_weights": "KRW",
    "source_residual_weights": "SRW",

    "filters": "F",
    "filter_scaling": "FS",
    "kernel_size": "K",
    "layers": "L",
    "block_conv_layers": "BCL",
    "strides": "S",
    "upsampling_interpolation": "BU",
    "resampling_kernel_size": "RK",
    "input_kernel_size": "IK",
    "gru_kernel_size": "GK",
    "kernel_l2_amp": "Kl2",
    "kernel_l1_amp": "Kl1",
    "bias_l2_amp": "Bl2",
    "bias_l1_amp": "Bl1",
    "kappalog": "KaL",
    "kappa_normalize": "KaN",
    "activation": "NL",
    "batch_norm": "BN",
    "dropout_rate": "D",
    "gru_architecture": "GA",

    "adam": "A",
    "steps": "TS",
    "source_link": "Sli",
    "flux_lagrange_multiplier": "FLM"
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
    else:
        raise NotImplementedError(f"{args.strategy} not in ['uniform', 'exhaustive']")


def uniform_grid_search(args):
    for gridsearch_id in range(1, args.n_models + 1):
        new_args = copy.deepcopy(args)
        args_dict = vars(new_args)
        nicknames = []
        params = []
        for p in RIM_HPARAMS + UNET_MODEL_HPARAMS + EXTRA_PARAMS:
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
    for p in RIM_HPARAMS + UNET_MODEL_HPARAMS + EXTRA_PARAMS:
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
        for p in RIM_HPARAMS + UNET_MODEL_HPARAMS + EXTRA_PARAMS:
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
        params_dict = {k: v for k, v in vars(run_args).items() if k in RIM_HPARAMS + UNET_MODEL_HPARAMS + EXTRA_PARAMS}
        params_dict.update({
            "experiment_id": run_args.logname,
            "train_cost": history["train_cost"][-1],
            "val_cost": history["val_cost"][-1],
            "train_chi_squared": history["train_chi_squared"][-1],
            "val_chi_squared": history["val_chi_squared"][-1],
            "best_score": best_score,
            "train_kappa_cost": history["train_kappa_cost"][-1],
            "val_kappa_cost": history["val_kappa_cost"][-1],
            "train_source_cost": history["train_source_cost"][-1],
            "val_source_cost": history["val_source_cost"][-1]
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
    parser.add_argument("--n_models",               default=10,     type=int,       help="Models to train")
    parser.add_argument("--datasets",               required=True,  nargs="+",      help="Path to directories that contains tfrecords of dataset. Can be multiple inputs (space separated)")
    parser.add_argument("--val_datasets",           default=None,   nargs="+",      help="Validation dataset path")
    parser.add_argument("--compression_type",       default=None,                   help="Compression type used to write data. Default assumes no compression.")
    parser.add_argument("--strategy",               default="uniform",              help="Allowed startegies are 'uniform' and 'exhaustive'.")

    # Physical model hyperparameter
    parser.add_argument("--forward_method",         default="conv2d",               help="One of ['conv2d', 'fft', 'unet']. If the option 'unet' is chosen, the parameter "
                                                                                         "'--raytracer' must be provided and point to model checkpoint directory.")
    parser.add_argument("--raytracer",              default=None,                   help="Path to raytracer checkpoint dir if method 'unet' is used.")

    # RIM hyperparameters
    parser.add_argument("--steps",              default=16, nargs="+",    type=int,       help="Number of time steps of RIM")
    parser.add_argument("--adam",               default=0,  nargs="+",    type=int,       help="ADAM update for the log-likelihood gradient.")
    parser.add_argument("--kappalog",           action="store_true")
    parser.add_argument("--kappa_normalize",    action="store_true")
    parser.add_argument("--source_link",        default="identity",  nargs="+",           help="One of 'exp', 'source' or 'identity' (default).")
    parser.add_argument("--kappa_init",         required=True,                  help="Path to initial kappa (npy file)")
    parser.add_argument("--source_init",        required=True,                  help="Path to initial source (npy file)")
    parser.add_argument("--flux_lagrange_multiplier",       default=0.,   nargs="+", type=float,     help="Value of Lagrange multiplier for the flux constraint")

    # Shared Unet params
    parser.add_argument("--filters",                                    default=32, nargs="+",    type=int)
    parser.add_argument("--filter_scaling",                             default=1, nargs="+",     type=float)
    parser.add_argument("--kernel_size",                                default=3, nargs="+",     type=int)
    parser.add_argument("--layers",                                     default=2, nargs="+",     type=int)
    parser.add_argument("--block_conv_layers",                          default=2, nargs="+",     type=int)
    parser.add_argument("--strides",                                    default=2, nargs="+",     type=int)
    parser.add_argument("--bottleneck_kernel_size",                     default=None, nargs="+",  type=int)
    parser.add_argument("--resampling_kernel_size",                     default=None, nargs="+",  type=int)
    parser.add_argument("--input_kernel_size",                          default=11,   nargs="+",  type=int)
    parser.add_argument("--gru_kernel_size",                            default=None, nargs="+",  type=int)
    parser.add_argument("--upsampling_interpolation",                   default=0,    nargs="+",  type=int)
    parser.add_argument("--batch_norm",                                 default=0,    nargs="+",  type=int)
    parser.add_argument("--dropout_rate",                               default=None, nargs="+",  type=float)
    parser.add_argument("--kernel_l2_amp",                              default=0, nargs="+",  type=float)
    parser.add_argument("--bias_l2_amp",                                default=0, nargs="+",  type=float)
    parser.add_argument("--kernel_l1_amp",                              default=0, nargs="+",  type=float)
    parser.add_argument("--bias_l1_amp",                                default=0, nargs="+",  type=float)
    parser.add_argument("--activation",                                 default="leaky_relu", nargs="+")
    parser.add_argument("--initializer",                                default="glorot_normal", nargs="+",)
    parser.add_argument("--gru_architecture",                           default="concat", nargs="+",  help="'concat': architecture of Laurence. 'plus': original RNN architecture")


    # Training set params
    parser.add_argument("--batch_size",             default=1, nargs="+",  type=int,       help="Number of images in a batch. ")
    parser.add_argument("--train_split",            default=0.8,    type=float,     help="Fraction of the training set.")
    parser.add_argument("--total_items",            required=True,  nargs="+", type=int,  help="Total images in an epoch.")
    # ... for tfrecord dataset
    parser.add_argument("--num_parallel_reads",     default=10,     type=int,       help="TFRecord dataset number of parallel reads when loading data.")
    parser.add_argument("--cache_file",             default=None,                   help="Path to cache file, useful when training on server. Use ${SLURM_TMPDIR}/cache")
    parser.add_argument("--cycle_length",           default=4,      type=int,       help="Number of files to read concurrently.")
    parser.add_argument("--block_length",           default=1,      type=int,       help="Number of example to read from each files.")

    # Optimization params
    parser.add_argument("-e", "--epochs",           default=10,     type=int,      help="Number of epochs for training.")
    parser.add_argument("--optimizer",              default="Adam", nargs="+",     help="Class name of the optimizer (e.g. 'Adam' or 'Adamax')")
    parser.add_argument("--initial_learning_rate",  default=1e-3,   nargs="+",   type=float,     help="Initial learning rate.")
    parser.add_argument("--decay_rate",             default=1.,     nargs="+",   type=float,     help="Exponential decay rate of learning rate (1=no decay).")
    parser.add_argument("--decay_steps",            default=1000,   nargs="+",   type=int,       help="Decay steps of exponential decay of the learning rate.")
    parser.add_argument("--staircase",              action="store_true",            help="Learning rate schedule only change after decay steps if enabled.")
    parser.add_argument("--patience",               default=np.inf, type=int,       help="Number of step at which training is stopped if no improvement is recorder.")
    parser.add_argument("--tolerance",              default=0,      type=float,     help="Current score <= (1 - tolerance) * best score => reset patience, else reduce patience.")
    parser.add_argument("--track_train",            action="store_true",            help="Track training metric instead of validation metric, in case we want to overfit")
    parser.add_argument("--max_time",               default=np.inf, type=float,     help="Time allowed for the training, in hours.")
    parser.add_argument("--buffer_size",            default=1000,   type=int,       help="Buffer size for shuffling at each epoch.")
    parser.add_argument("--time_weights",           default="uniform", nargs="+",   help="uniform: w_t=1 for all t, linear: w_t~t, quadratic: w_t~t^2")
    parser.add_argument("--unroll_time_steps",      action="store_true",            help="Unroll time steps of RIM in GPU usinf tf.function")
    parser.add_argument("--reset_optimizer_states",  action="store_true",            help="When training from pre-trained weights, reset states of optimizer.")
    parser.add_argument("--kappa_residual_weights",         default="uniform",        nargs="+",     help="Options are ['uniform', 'linear', 'quadratic', 'sqrt']")
    parser.add_argument("--source_residual_weights",        default="uniform",        nargs="+",     help="Options are ['uniform', 'linear', 'quadratic']")

    # logs
    parser.add_argument("--logdir",                  default="None",                help="Path of logs directory. Default if None, no logs recorded.")
    parser.add_argument("--logname_prefixe",         default="RIMUnet512",          help="If name of the log is not provided, this prefix is prepended to the date")
    parser.add_argument("--model_dir",               default="None",                help="Path to the directory where to save models checkpoints.")
    parser.add_argument("--checkpoints",             default=10,    type=int,       help="Save a checkpoint of the models each {%} iteration.")
    parser.add_argument("--max_to_keep",             default=3,     type=int,       help="Max model checkpoint to keep.")
    parser.add_argument("--n_residuals",             default=1,     type=int,       help="Number of residual plots to save. Add overhead at the end of an epoch only.")

    # Make sure each model train on the same dataset
    parser.add_argument("--seed",                   default=42, nargs="+",  type=int, help="Random seed for numpy and tensorflow.")

    # Keep these as default, they need to be in Namespace but we dont use them for this script
    parser.add_argument("--model_id",               default="None",                 help="Start training from previous checkpoint of this model if provided")
    parser.add_argument("--json_override",          default=None,   nargs="+",      help="A json filepath that will override every command line parameters. Useful for reproducibility")
    parser.add_argument("--v2",                     action="store_true",            help="Use v2 decoding of tfrecords")

    args = parser.parse_args()
    distributed_strategy(args)
