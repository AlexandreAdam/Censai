import numpy as np
import os
from datetime import datetime
from scripts.train_kappa_resnetvae_first_stage import main
import copy
import pandas as pd

# total number of slurm workers detected
# defaults to 1 if not running under SLURM
N_WORKERS = int(os.getenv('SLURM_ARRAY_TASK_COUNT', 1))

# this worker's array index. Assumes slurm array job is zero-indexed
# defaults to zero if not running under SLURM
THIS_WORKER = int(os.getenv('SLURM_ARRAY_TASK_ID', 0)) ## it starts from 1!!

DATE = datetime.now().strftime("%y%m%d%H%M%S")

VAE_HPARAMS = [
    "layers",
    "res_blocks_in_layer",
    "conv_layers_per_block",
    "filter_scaling",
    "filters",
    "kernel_size",
    "res_architecture",
    "activation",
    "dropout_rate",
    "batch_norm",
    "latent_size"
]

EXTRA_PARAMS = [
    "total_items",
    "optimizer",
    "seed",
    "batch_size",
    "initial_learning_rate",
    "decay_rate",
    "beta_cyclical",
    "skip_strength_decay_power",
    "l2_bottleneck_decay_power",
]

PARAMS_NICKNAME = {
    "layers": "L",
    "res_blocks_in_layer": "RB",
    "conv_layers_per_block": "CB",
    "filter_scaling": "FS",
    "filters": "F",
    "kernel_size": "K",
    "res_architecture": "A",
    "activation": "NL",
    "dropout_rate": "DR",
    "batch_norm": "BN",
    "latent_size": "LS",
    "total_items": "TI",
    "optimizer": "O",
    "seed": "",
    "batch_size": "B",
    "initial_learning_rate": "lr",
    "decay_rate": "dr",
    "beta_cyclical": "betaC",
    "skip_strength_decay_power": "SSDP",
    "l2_bottleneck_decay_power": "l2DP"
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
        for p in VAE_HPARAMS + EXTRA_PARAMS:
            if len(args_dict[p]) > 1:
                # this way, numpy does not cast int to int64 or float to float32
                args_dict[p] = args_dict[p][np.random.choice(range(len(args_dict[p])))]
                nicknames.append(PARAMS_NICKNAME[p])
                params.append(args_dict[p])
        param_str = "_" + "_".join([f"{nickname}{param}" for nickname, param in zip(nicknames, params)])
        args_dict.update({"logname": args.logname_prefixe + "_" + f"{gridsearch_id:03d}" + param_str + "_" + DATE})
        yield new_args


def exhaustive_grid_search(args):
    """
    Lexicographic ordering of given parameter lists, up to n_models deep.
    """
    from itertools import product
    grid_params = []
    for p in VAE_HPARAMS + EXTRA_PARAMS:
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
        for p in VAE_HPARAMS + EXTRA_PARAMS:
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
        params_dict = {k: v for k, v in vars(run_args).items() if k in VAE_HPARAMS + EXTRA_PARAMS}
        params_dict.update({
            "experiment_id": run_args.logname,
            "train_cost": history["train_cost"][-1],
            "val_cost": history["val_cost"][-1],
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
    parser.add_argument("--n_models",               default=10,     type=int,       help="Models to train")
    parser.add_argument("--datasets",               required=True, nargs="+",       help="Path to kappa directories, with tfrecords files")
    parser.add_argument("--compression_type",       default=None,                   help="Compression type used to write data. Default assumes no compression.")
    parser.add_argument("--strategy",               default="uniform",              help="Allowed startegies are 'uniform' and 'exhaustive'.")


    # Model params
    parser.add_argument("--layers",                 default=4,          nargs="+",      type=int,       help="Number of layer in encoder/decoder")
    parser.add_argument("--res_blocks_in_layer",    default=2,          nargs="+",      type=int,       help="List of res block per layers. If single number is given, assume uniform structure")
    parser.add_argument("--conv_layers_per_block",  default=2,          nargs="+",      type=int,       help="Number of convolution layers in a block")
    parser.add_argument("--filter_scaling",         default=2,          nargs="+",      type=float,     help="Filter scaling after each layers")
    parser.add_argument("--filters",                default=8,          nargs="+",      type=int,       help="Number of filters in the first layer")
    parser.add_argument("--kernel_size",            default=3,          nargs="+",      type=int)
    parser.add_argument("--res_architecture",       default="bare",     nargs="+",                      help="One of ['bare', 'original', 'bn_after_addition', 'relu_before_addition', 'relu_only_pre_activation', 'full_pre_activation', 'full_pre_activation_rescale']")
    parser.add_argument("--kernel_reg_amp",         default=1e-4,                       type=float,     help="L2 kernel regularization amplitude")
    parser.add_argument("--bias_reg_amp",           default=1e-4,                       type=float,     help="L2 bias regularizer amplitude")
    parser.add_argument("--activation",             default="relu",     nargs="+",                      help="Name of activation function, on of ['relu', 'leaky_relu', 'bipolar_relu', 'bipolar_leaky_relu', 'bipolar_elu', 'gelu', etc.]")
    parser.add_argument("--dropout_rate",           default=None,       nargs="+",      type=float,     help="2D spatial dropout rate (drop entire feature map to help them become independent)")
    parser.add_argument("--batch_norm",             default=0,          nargs="+",      type=int,       help="0: False, do no use batch norm. 1: True, use batch norm beforce activation")
    parser.add_argument("--latent_size",            default=16,         nargs="+",      type=int,       help="Twice the size of the latent code vector z")

    # Training set params
    parser.add_argument("--batch_size",             default=10,     nargs="+",  type=int,   help="Number of images in a batch. ")
    parser.add_argument("--train_split",            default=0.8,                type=float,             help="Fraction of the training set.")
    parser.add_argument("--total_items",            required=True,  nargs="+",  type=int,   help="Total images in an epoch.")
    # ... for tfrecord dataset
    parser.add_argument("--cache_file",             default=None,                           help="Path to cache file, useful when training on server. Use ${SLURM_TMPDIR}/cache")
    parser.add_argument("--block_length",           default=1,                  type=int,   help="Number of example to read from each files.")

    # Optimization params
    parser.add_argument("-e", "--epochs",                   default=10,     type=int,                   help="Number of epochs for training.")
    parser.add_argument("--optimizer",                      default="Adam", nargs="+",                  help="Class name of the optimizer (e.g. 'Adam' or 'Adamax')")
    parser.add_argument("--initial_learning_rate",          default=1e-3,   nargs="+",  type=float,     help="Initial learning rate.")
    parser.add_argument("--decay_rate",                     default=1.,     nargs="+",  type=float,     help="Exponential decay rate of learning rate (1=no decay).")
    parser.add_argument("--decay_steps",                    default=1000,               type=int,       help="Decay steps of exponential decay of the learning rate.")
    parser.add_argument("--beta_init",                      default=0.,                 type=float,     help="Initial value of the beta schedule")
    parser.add_argument("--beta_end_value",                 default=1.,                 type=float,     help="End value of the beta schedule")
    parser.add_argument("--beta_decay_power",               default=1.,                 type=float,     help="Power of the Polynomial schedule")
    parser.add_argument("--beta_decay_steps",               default=1000,               type=int,       help="Number of steps until end of schedule is reached")
    parser.add_argument("--beta_cyclical",                  default=0,      nargs="+",  type=int,       help="Make beta schedule cyclical if 1. 0: Monotone schedule.")
    parser.add_argument("--skip_strength_init",             default=1.,                 type=float,     help="Initial value of the skip_strength schedule")
    parser.add_argument("--skip_strength_end_value",        default=0.,                 type=float,     help="End value of the skip_strength schedule")
    parser.add_argument("--skip_strength_decay_power",      default=0.5,    nargs="+",  type=float,     help="Power of the Polynomial schedule")
    parser.add_argument("--skip_strength_decay_steps",      default=1000,               type=int,       help="Number of steps until end of schedule is reached")
    parser.add_argument("--l2_bottleneck_init",             default=1.,                 type=float,     help="Initial value of the l2_bottleneck schedule")
    parser.add_argument("--l2_bottleneck_end_value",        default=0.,                 type=float,     help="End value of the l2_bottleneck schedule")
    parser.add_argument("--l2_bottleneck_decay_power",      default=0.5,    nargs="+",  type=float,     help="Power of the Polynomial schedule")
    parser.add_argument("--l2_bottleneck_decay_steps",      default=1000,               type=int,       help="Number of steps until end of schedule is reached")
    parser.add_argument("--staircase",                      action="store_true",            help="Learning rate schedule only change after decay steps if enabled.")
    parser.add_argument("--clipping",                       action="store_true",            help="Clip backprop gradients between -10 and 10.")
    parser.add_argument("--patience",                       default=np.inf, type=int,       help="Number of step at which training is stopped if no improvement is recorder.")
    parser.add_argument("--tolerance",                      default=0,      type=float,     help="Current score <= (1 - tolerance) * best score => reset patience, else reduce patience.")
    parser.add_argument("--track_train",                    action="store_true",            help="Track training metric instead of validation metric, in case we want to overfit")
    parser.add_argument("--max_time",                       default=np.inf, type=float,     help="Time allowed for the training, in hours.")

    # logs
    parser.add_argument("--logdir",                  default="None",                help="Path of logs directory. Default if None, no logs recorded.")
    parser.add_argument("--logname",                 default=None,                  help="Overwrite name of the log with this argument")
    parser.add_argument("--logname_prefixe",         default="KappaVAE",            help="If name of the log is not provided, this prefix is prepended to the date")
    parser.add_argument("--model_dir",               default="None",                help="Path to the directory where to save models checkpoints.")
    parser.add_argument("--checkpoints",             default=10,    type=int,       help="Save a checkpoint of the models each {%} iteration.")
    parser.add_argument("--max_to_keep",             default=3,     type=int,       help="Max model checkpoint to keep.")
    parser.add_argument("--n_residuals",             default=1,     type=int,       help="Number of residual plots to save. Add overhead at the end of an epoch only.")

    # Reproducibility params
    parser.add_argument("--seed",                   default=None,   nargs="+",   type=int,       help="Random seed for numpy and tensorflow.")

    # Keep these as default, they need to be in Namespace but we dont use them for this script
    parser.add_argument("--model_id",                       default="None",              help="Start training from previous "
                                                                                          "checkpoint of this model if provided")
    parser.add_argument("--load_checkpoint",                default="best",              help="One of 'best', 'lastest' or the specific checkpoint index")
    parser.add_argument("--json_override",                  default=None,               help="A json filepath that will override every command line parameters. "
                                                                                           "Useful for reproducibility")
    args = parser.parse_args()
    distributed_strategy(args)