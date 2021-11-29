import numpy as np
import os
from datetime import datetime
from scripts.train_rim_shared_unetv4 import main
import pandas as pd
import json

# total number of slurm workers detected
# defaults to 1 if not running under SLURM
N_WORKERS = int(os.getenv('SLURM_ARRAY_TASK_COUNT', 1))

# this worker's array index. Assumes slurm array job is zero-indexed
# defaults to zero if not running under SLURM
THIS_WORKER = int(os.getenv('SLURM_ARRAY_TASK_ID', 0)) ## it starts from 1!!

DATE = datetime.now().strftime("%y%m%d%H%M%S")

RIM_HPARAMS = [
    "adam",
    "rmsprop",
    "steps",
    "kappalog",
    "kappa_normalize",
    "source_link",
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


def distributed_strategy(args):
    assert N_WORKERS == len(args.models)
    model = args.models[THIS_WORKER - 1]
    model_dir = os.path.join(os.getenv('CENSAI_PATH'), "models")
    vars(args)["model_dir"] = model_dir
    vars(args)["model_id"] = model
    with open(os.path.join(model_dir, model, "unet_hparams.json")) as f:
        unet_hparams = json.load(f)
    with open(os.path.join(model_dir, model, "rim_hparams.json")) as f:
        rim_hparams = json.load(f)
    with open(os.path.join(model_dir, model, "script_params.json")) as f:
        script_hparams = json.load(f)
    keys = ["kappa_residual_weights", "source_residual_weights", "source_link", "flux_lagrange_multiplier"]
    optim_params = {key: value for key, value in zip(keys, [script_hparams[k] for k in keys])}
    if "rmsprop" in script_hparams.keys():
        vars(args)["rmsprop"] = script_hparams["rmsprop"]
    else:
        vars(args)["rmsprop"] = 0
    vars(args).update(unet_hparams)
    vars(args).update(rim_hparams)
    vars(args).update(optim_params)
    vars(args)["logname"] = f"continue_lr{args.initial_learning_rate}_" + DATE
    history, best_score = main(args)
    params_dict = {k: v for k, v in vars(args).items() if k in RIM_HPARAMS + UNET_MODEL_HPARAMS + EXTRA_PARAMS}
    logname = args.model_id + '_' + args.logname
    params_dict.update({
        "experiment_id": logname,
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
    df = pd.DataFrame(params_dict, index=[THIS_WORKER-1])
    grid_csv_path = os.path.join(os.getenv("CENSAI_PATH"), "results", f"{args.logname_prefixe}.csv")
    this_run_csv_path = os.path.join(os.getenv("CENSAI_PATH"), "results", f"{logname}.csv")
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
    parser.add_argument("--models",                 required=True,  nargs="+",       help="List of models to continue training")
    parser.add_argument("--datasets",               required=True,  nargs="+",      help="Path to directories that contains tfrecords of dataset. Can be multiple inputs (space separated)")
    parser.add_argument("--val_datasets",           default=None,   nargs="+",      help="Validation dataset path")
    parser.add_argument("--compression_type",       default=None,                   help="Compression type used to write data. Default assumes no compression.")

    # Physical model hyperparameter
    parser.add_argument("--forward_method",         default="conv2d",               help="One of ['conv2d', 'fft', 'unet']. If the option 'unet' is chosen, the parameter "
                                                                                         "'--raytracer' must be provided and point to model checkpoint directory.")
    parser.add_argument("--raytracer",              default=None,                   help="Path to raytracer checkpoint dir if method 'unet' is used.")

    # Training set params
    parser.add_argument("--batch_size",             default=1,      type=int,       help="Number of images in a batch. ")
    parser.add_argument("--train_split",            default=0.8,    type=float,     help="Fraction of the training set.")
    parser.add_argument("--total_items",            required=True,  type=int,  help="Total images in an epoch.")
    # ... for tfrecord dataset
    parser.add_argument("--num_parallel_reads",     default=10,     type=int,       help="TFRecord dataset number of parallel reads when loading data.")
    parser.add_argument("--cache_file",             default=None,                   help="Path to cache file, useful when training on server. Use ${SLURM_TMPDIR}/cache")
    parser.add_argument("--cycle_length",           default=4,      type=int,       help="Number of files to read concurrently.")
    parser.add_argument("--block_length",           default=1,      type=int,       help="Number of example to read from each files.")

    # Optimization params
    parser.add_argument("-e", "--epochs",            default=10,     type=int,      help="Number of epochs for training.")
    parser.add_argument("--optimizer",               default="Adam",                help="Class name of the optimizer (e.g. 'Adam' or 'Adamax')")
    parser.add_argument("--initial_learning_rate",   default=1e-3,   type=float,     help="Initial learning rate.")
    parser.add_argument("--decay_rate",              default=1.,     type=float,     help="Exponential decay rate of learning rate (1=no decay).")
    parser.add_argument("--decay_steps",             default=1000,   type=int,       help="Decay steps of exponential decay of the learning rate.")
    parser.add_argument("--staircase",               action="store_true",            help="Learning rate schedule only change after decay steps if enabled.")
    parser.add_argument("--patience",                default=np.inf, type=int,       help="Number of step at which training is stopped if no improvement is recorder.")
    parser.add_argument("--tolerance",               default=0,      type=float,     help="Current score <= (1 - tolerance) * best score => reset patience, else reduce patience.")
    parser.add_argument("--track_train",             action="store_true",            help="Track training metric instead of validation metric, in case we want to overfit")
    parser.add_argument("--max_time",                default=np.inf, type=float,     help="Time allowed for the training, in hours.")
    parser.add_argument("--buffer_size",             default=1000,   type=int,       help="Buffer size for shuffling at each epoch.")
    parser.add_argument("--time_weights",            default="uniform",              help="uniform: w_t=1 for all t, linear: w_t~t, quadratic: w_t~t^2")
    parser.add_argument("--unroll_time_steps",       action="store_true",            help="Unroll time steps of RIM in GPU usinf tf.function")
    parser.add_argument("--reset_optimizer_states",  action="store_true",            help="When training from pre-trained weights, reset states of optimizer.")
    parser.add_argument("--kappa_residual_weights",  default="uniform",              help="Options are ['uniform', 'linear', 'quadratic', 'sqrt']")
    parser.add_argument("--source_residual_weights", default="uniform",              help="Options are ['uniform', 'linear', 'quadratic']")

    # logs
    parser.add_argument("--logdir",                  default="None",                help="Path of logs directory. Default if None, no logs recorded.")
    parser.add_argument("--logname_prefixe",         default="RIMUnet512",          help="If name of the log is not provided, this prefix is prepended to the date")
    parser.add_argument("--checkpoints",             default=10,    type=int,       help="Save a checkpoint of the models each {%} iteration.")
    parser.add_argument("--max_to_keep",             default=3,     type=int,       help="Max model checkpoint to keep.")
    parser.add_argument("--n_residuals",             default=1,     type=int,       help="Number of residual plots to save. Add overhead at the end of an epoch only.")

    # Make sure each model train on the same dataset
    parser.add_argument("--seed",                   default=42,     type=int, help="Random seed for numpy and tensorflow.")

    # Keep these as default, they need to be in Namespace but we dont use them for this script
    parser.add_argument("--filter_cap",             default=1024,   type=int,       help="Put there for legacy scripts, make sure more recent scripts have this cap as well")
    parser.add_argument("--model_id",               default="None",                 help="Start training from previous checkpoint of this model if provided")
    parser.add_argument("--json_override",          default=None,   nargs="+",      help="A json filepath that will override every command line parameters. Useful for reproducibility")

    args = parser.parse_args()
    distributed_strategy(args)
