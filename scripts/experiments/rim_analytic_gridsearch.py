import numpy as np
import os
from datetime import datetime
from scripts.train_rim_analytic import main
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
    "steps"
]

MODEL_HPARAMS = [
    "layers",
    "units",
    "unit_scaling",
    "mlp_before_gru",
    "activation"
]

EXTRA_PARAMS = [
    "total_items",
    "optimizer",
    "seed",
    "batch_size",
    "initial_learning_rate",
    "decay_rate",
    "decay_steps",
    "time_weights"
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

    "adam": "A",
    "steps": "TS",

    "layers": "L",
    "units": "U",
    "unit_scaling": "US",
    "mlp_before_gru": "LBG",
    "activation": "A",
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
        for p in RIM_HPARAMS + MODEL_HPARAMS + EXTRA_PARAMS:
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
    for p in RIM_HPARAMS + MODEL_HPARAMS + EXTRA_PARAMS:
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
            "train_chi_squared": history["train_chi_squared"][-1],
            "best_score": best_score,
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
    parser.add_argument("--n_models",               default=1,     type=int,       help="Models to train")
    parser.add_argument("--strategy",               default="uniform",              help="Allowed startegies are 'uniform' and 'exhaustive'.")

    # RIM hyperparameters
    parser.add_argument("--steps",                  default=4,     nargs="+",   type=int,       help="Number of time steps of RIM")
    parser.add_argument("--adam",                   default=0,      nargs="+",  type=int,       help="ADAM update for the log-likelihood gradient.")

    # Physical parameters
    parser.add_argument("--pixels",                 default=128,    type=int)
    parser.add_argument("--image_fov",              default=7.68,   type=float)
    parser.add_argument("--src_fov",                default=3.,     type=float)
    parser.add_argument("--psf_cutout_size",        default=16,     type=int)
    parser.add_argument("--r_ein_min",              default=0.5,    type=float)
    parser.add_argument("--r_ein_max",              default=2.5,    type=float)
    parser.add_argument("--n_max",                  default=3.,     type=float)
    parser.add_argument("--n_min",                  default=1.,     type=float)
    parser.add_argument("--r_eff_min",              default=0.2,    type=float)
    parser.add_argument("--r_eff_max",              default=1.,     type=float)
    parser.add_argument("--max_gamma",              default=0.1,    type=float)
    parser.add_argument("--max_ellipticity",        default=0.4,    type=float)
    parser.add_argument("--max_lens_shift",         default=0.3,    type=float)
    parser.add_argument("--max_source_shift",       default=0.3,    type=float)
    parser.add_argument("--noise_rms_min",          default=0.001,  type=float)
    parser.add_argument("--noise_rms_max",          default=0.05,   type=float)
    parser.add_argument("--noise_rms_mean",         default=0.08,   type=float)
    parser.add_argument("--noise_rms_std",          default=0.08,   type=float)
    parser.add_argument("--psf_fwhm_min",           default=0.06,   type=float)
    parser.add_argument("--psf_fwhm_max",           default=0.5,    type=float)
    parser.add_argument("--psf_fwhm_mean",          default=0.1,    type=float)
    parser.add_argument("--psf_fwhm_std",           default=0.1,    type=float)

    # Model hyperparameters
    parser.add_argument("--layers",                 default=2,      nargs="+",  type=int)
    parser.add_argument("--units",                  default=32,     nargs="+",  type=int)
    parser.add_argument("--unit_scaling",           default=4,      nargs="+",  type=int)
    parser.add_argument("--unit_cap",               default=1024,   type=int)
    parser.add_argument("--mlp_before_gru",         default=2,      nargs="+",  type=int)
    parser.add_argument("--activation",             default="tanh", nargs="+")

    # Training set params
    parser.add_argument("--batch_size",             default=1,      nargs="+",  type=int,       help="Number of images in a batch. ")
    parser.add_argument("--total_items",            required=True,  nargs="+",  type=int,       help="Total images in an epoch.")

    # Optimization params
    parser.add_argument("-e", "--epochs",           default=10,     type=int,      help="Number of epochs for training.")
    parser.add_argument("--optimizer",              default="Adam", nargs="+",     help="Class name of the optimizer (e.g. 'Adam' or 'Adamax')")
    parser.add_argument("--initial_learning_rate",  default=1e-3,   nargs="+",   type=float,     help="Initial learning rate.")
    parser.add_argument("--decay_rate",             default=1.,     nargs="+",   type=float,     help="Exponential decay rate of learning rate (1=no decay).")
    parser.add_argument("--decay_steps",            default=1000,   nargs="+",   type=int,       help="Decay steps of exponential decay of the learning rate.")
    parser.add_argument("--staircase",              action="store_true",             help="Learning rate schedule only change after decay steps if enabled.")
    parser.add_argument("--patience",               default=np.inf, type=int,        help="Number of step at which training is stopped if no improvement is recorder.")
    parser.add_argument("--tolerance",              default=0,      type=float,      help="Current score <= (1 - tolerance) * best score => reset patience, else reduce patience.")
    parser.add_argument("--track_train",            action="store_true",             help="Track training metric instead of validation metric, in case we want to overfit")
    parser.add_argument("--max_time",               default=np.inf, type=float,      help="Time allowed for the training, in hours.")
    parser.add_argument("--time_weights",           default="uniform", nargs="+",    help="uniform: w_t=1 for all t, linear: w_t~t, quadratic: w_t~t^2")
    parser.add_argument("--reset_optimizer_states",  action="store_true",            help="When training from pre-trained weights, reset states of optimizer.")

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

    args = parser.parse_args()
    distributed_strategy(args)
