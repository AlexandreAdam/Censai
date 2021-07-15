import tensorflow as tf
import numpy as np
from censai import PhysicalModel, RIMSharedUnet
from censai.models import SharedUnetModel, RayTracer
from censai.data.lenses_tng import decode_train, decode_physical_model_info
from censai.utils import nullwriter, rim_residual_plot as residual_plot, plot_to_image
import os, glob, time, json
from datetime import datetime

""" # NOTE ON THE USE OF MULTIPLE GPUS #
Double the number of gpus will not speed up the code. In fact, doubling the number of gpus and mirroring 
the ops accross replicas means the code is TWICE as slow.

In fact, using multiple gpus means one should at least multiply the batch size by the number of gpus introduced, 
and optimize hyperparameters accordingly (learning rate should be scaled similarly).
"""
gpus = tf.config.list_physical_devices('GPU')
if len(gpus) == 1:
    STRATEGY = tf.distribute.OneDeviceStrategy(device="/gpu:0")
elif len(gpus) > 1:
    STRATEGY = tf.distribute.MirroredStrategy()

RIM_HPARAMS = [
    "adam",
    "steps",
    "kappalog",
    "kappa_normalize",
    "kappa_init",
    "source_init"
]
UNET_MODEL_HPARAMS = [
    "filters",
    "filter_scaling",
    "kernel_size",
    "layers",
    "block_conv_layers",
    "strides",
    "bottleneck_kernel_size",
    "bottleneck_filters",
    "resampling_kernel_size",
    "gru_kernel_size",
    "upsampling_interpolation",
    "kernel_regularizer_amp",
    "bias_regularizer_amp",
    "activation",
    "alpha",
    "initializer",
    "kappa_resize_filters",
    "kappa_resize_method",
    "kappa_resize_conv_layers",
    "kappa_resize_strides",
    "kappa_resize_kernel_size",
    "kappa_resize_separate_grad_downsampling"
]


def is_power_of_two(n):
    while n > 1:
        n /= 2
        if int(n) != n:
            return False
        elif n == 2:
            return True


def main(args):
    files = []
    for dataset in args.datasets:
        files.extend(glob.glob(os.path.join(dataset, "*.tfrecords")))
    np.random.shuffle(files)
    # Read concurrently from multiple records
    files = tf.data.Dataset.from_tensor_slices(files)
    dataset = files.interleave(lambda x: tf.data.TFRecordDataset(x, num_parallel_reads=args.num_parallel_reads, compression_type=args.compression_type),
                               cycle_length=args.cycle_length, block_length=args.block_length)
    # Read off global parameters from first example in dataset
    for physical_params in dataset.map(decode_physical_model_info):
        break
    dataset = dataset.map(decode_train).batch(args.batch_size)
    if args.cache_file is not None:
        dataset = dataset.cache(args.cache_file).prefetch(tf.data.experimental.AUTOTUNE)
    else:  # do not cache if no file is provided, dataset is huge and does not fit in GPU or RAM
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    train_dataset = dataset.take(int(args.train_split * args.total_items) // args.batch_size) # dont forget to divide by batch size!
    val_dataset = dataset.skip(int(args.train_split * args.total_items) // args.batch_size)
    val_dataset = val_dataset.take(int((1 - args.train_split) * args.total_items) // args.batch_size)
    train_dataset = STRATEGY.experimental_distribute_dataset(train_dataset)
    val_dataset = STRATEGY.experimental_distribute_dataset(val_dataset)
    if args.raytracer is not None:
        with open(os.path.join(args.raytracer, "ray_tracer_hparams.json"), "r") as f:
            raytracer_hparams = json.load(f)
    with STRATEGY.scope():  # Replicate ops accross gpus
        if args.raytracer is not None:
            raytracer = RayTracer(**raytracer_hparams)
            # load last checkpoint in the checkpoint directory
            checkpoint = tf.train.Checkpoint(net=raytracer)
            manager = tf.train.CheckpointManager(checkpoint, directory=args.raytracer, max_to_keep=3)
            checkpoint.restore(manager.latest_checkpoint).expect_partial()
        else:
            raytracer = None
        phys = PhysicalModel(
            pixels=physical_params["kappa pixels"].numpy(),
            src_pixels=physical_params["src pixels"].numpy(),
            image_fov=physical_params["image fov"].numpy(),
            kappa_fov=physical_params["kappa fov"].numpy(),
            src_fov=physical_params["source fov"].numpy(),
            method=args.forward_method,
            noise_rms=physical_params["noise rms"].numpy(),
            raytracer=raytracer
        )
        assert is_power_of_two(phys.pixels)
        assert is_power_of_two(phys.src_pixels)
        kappa_resize_layers = (phys.pixels // phys.src_pixels) // args.kappa_resize_strides
        vars(args).update({"kappa_resize_layers": int(kappa_resize_layers)})

        unet = SharedUnetModel(
            kappa_resize_layers=args.kappa_resize_layers,
            filters=args.filters,
            filter_scaling=args.filter_scaling,
            kernel_size=args.kernel_size,
            layers=args.layers,
            block_conv_layers=args.block_conv_layers,
            strides=args.strides,
            bottleneck_kernel_size=args.bottleneck_kernel_size,
            bottleneck_filters=args.bottleneck_filters,
            resampling_kernel_size=args.resampling_kernel_size,
            gru_kernel_size=args.gru_kernel_size,
            upsampling_interpolation=args.upsampling_interpolation,
            kernel_regularizer_amp=args.kernel_regularizer_amp,
            bias_regularizer_amp=args.bias_regularizer_amp,
            activation=args.activation,
            alpha=args.alpha,
            initializer=args.initializer,
            kappa_resize_filters=args.kappa_resize_filters,
            kappa_resize_strides=args.kappa_resize_strides,
            kappa_resize_conv_layers=args.kappa_resize_conv_layers,
            kappa_resize_kernel_size=args.kappa_resize_kernel_size,
            kappa_resize_method=args.kappa_resize_method,
            kappa_resize_separate_grad_downsampling=args.kappa_resize_separate_grad_downsampling
        )
        rim = RIMSharedUnet(
            physical_model=phys,
            unet=unet,
            steps=args.steps,
            adam=args.adam,
            kappalog=args.kappalog,
            source_link=args.source_link,
            kappa_normalize=args.kappa_normalize,
            kappa_init=args.kappa_init,
            source_init=args.source_init
        )
        learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=args.initial_learning_rate,
            decay_rate=args.decay_rate,
            decay_steps=args.decay_steps,
            staircase=args.staircase
        )
        optim = tf.keras.optimizers.deserialize(
            {
                "class_name": args.optimizer,
                'config': {"learning_rate": learning_rate_schedule}
            }
        )

    # ==== Take care of where to write logs and stuff =================================================================
    if args.model_id.lower() != "none":
        logname = args.model_id
    elif args.logname is not None:
        logname = args.logname
    else:
        logname = args.logname_prefixe + "_" + datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    if args.logdir.lower() != "none":
        logdir = os.path.join(args.logdir, logname)
        if not os.path.isdir(logdir):
            os.mkdir(logdir)
        writer = tf.summary.create_file_writer(logdir)
    else:
        writer = nullwriter()
    # ===== Make sure directory and checkpoint manager are created to save model ===================================
    if args.model_dir.lower() != "none":
        checkpoints_dir = os.path.join(args.model_dir, logname)
        if not os.path.isdir(checkpoints_dir):
            os.mkdir(checkpoints_dir)
            with open(os.path.join(checkpoints_dir, "script_params.json"), "w") as f:
                json.dump(vars(args), f, indent=4)
            with open(os.path.join(checkpoints_dir, "unet_hparams.json"), "w") as f:
                hparams_dict = {key: vars(args)[key] for key in UNET_MODEL_HPARAMS}
                json.dump(hparams_dict, f, indent=4)
            with open(os.path.join(checkpoints_dir, "rim_hparams.json"), "w") as f:
                hparams_dict = {key: vars(args)[key] for key in RIM_HPARAMS}
                json.dump(hparams_dict, f, indent=4)
        ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optim, net=rim.unet)
        checkpoint_manager = tf.train.CheckpointManager(ckpt, checkpoints_dir, max_to_keep=args.max_to_keep)
        save_checkpoint = True
        # ======= Load model if model_id is provided ===============================================================
        if args.model_id.lower() != "none":
            if args.load_checkpoint == "lastest":
                checkpoint_manager.checkpoint.restore(checkpoint_manager.latest_checkpoint)
            elif args.load_checkpoint == "best":
                scores = np.loadtxt(os.path.join(checkpoints_dir, "score_sheet.txt"))
                _checkpoint = scores[np.argmin(scores[:, 1]), 0]
                checkpoint = checkpoint_manager.checkpoints[_checkpoint]
                checkpoint_manager.checkpoint.restore(checkpoint)
            else:
                checkpoint = checkpoint_manager.checkpoints[int(args.load_checkpoint)]
                checkpoint_manager.checkpoint.restore(checkpoint)
    else:
        save_checkpoint = False
    # =================================================================================================================

    def train_step(inputs):
        X, source, kappa = inputs
        with tf.GradientTape() as tape:
            tape.watch(rim.unet.trainable_variables)
            cost, chi_squared = rim.cost_function(X, source, kappa, outer_tape=tape, reduction=False)
            cost = tf.reduce_sum(cost) / args.batch_size  # Reduce by the global batch size, not the replica batch size
        gradient = tape.gradient(cost, rim.unet.trainable_variables)
        if args.clipping:
            gradient = [tf.clip_by_value(grad, -10, 10) for grad in gradient]
        optim.apply_gradients(zip(gradient, rim.unet.trainable_variables))
        chi_squared = tf.reduce_sum(chi_squared) / args.batch_size
        return cost, chi_squared

    @tf.function
    def distributed_train_step(dist_inputs):
        per_replica_losses, per_replica_chi_squared = STRATEGY.run(train_step, args=(dist_inputs,))
        # Replica losses are aggregated by summing them
        global_loss = STRATEGY.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
        global_chi_squared = STRATEGY.reduce(tf.distribute.ReduceOp.SUM, per_replica_chi_squared, axis=None)
        return global_loss, global_chi_squared

    def test_step(inputs):
        X, source, kappa = inputs
        cost, chi_squared = rim.cost_function(X, source, kappa, reduction=False)
        cost = tf.reduce_sum(cost) / args.batch_size
        chi_squared = tf.reduce_sum(chi_squared) / args.batch_size
        return cost, chi_squared

    @tf.function
    def distributed_test_step(dist_inputs):
        per_replica_losses, per_replica_chi_squared = STRATEGY.run(test_step, args=(dist_inputs,))
        # Replica losses are aggregated by summing them
        global_loss = STRATEGY.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
        global_chi_squared = STRATEGY.reduce(tf.distribute.ReduceOp.SUM, per_replica_chi_squared, axis=None)
        return global_loss, global_chi_squared

    # ====== Training loop ============================================================================================
    epoch_loss = tf.metrics.Mean()
    time_per_step = tf.metrics.Mean()
    val_loss = tf.metrics.Mean()
    epoch_chi_squared = tf.metrics.Mean()
    val_chi_squared = tf.metrics.Mean()
    history = {  # recorded at the end of an epoch only
        "train_cost": [],
        "train_chi_squared": [],
        "val_cost": [],
        "val_chi_squared": [],
        "learning_rate": [],
        "time_per_step": []
    }
    best_loss = np.inf
    patience = args.patience
    step = 0
    lastest_checkpoint = 1
    for epoch in range(args.epochs):
        epoch_loss.reset_states()
        epoch_chi_squared.reset_states()
        time_per_step.reset_states()
        with writer.as_default():
            for batch, distributed_inputs in enumerate(train_dataset):
                start = time.time()
                cost, chi_squared = distributed_train_step(distributed_inputs)
        # ========== Summary and logs ==================================================================================
                _time = time.time() - start
                tf.summary.scalar("Time per step", _time, step=step)
                tf.summary.scalar("MSE", cost, step=step)
                tf.summary.scalar("Chi Squared", chi_squared, step=step)
                time_per_step.update_state([_time])
                epoch_loss.update_state([cost])
                epoch_chi_squared.update_state([chi_squared])
                step += 1
            # last batch we make a summary of residuals
            for res_idx in range(min(args.n_residuals, args.batch_size)):
                lens_true = distributed_inputs[0][res_idx, ...]
                source_true = distributed_inputs[1][res_idx, ...]
                kappa_true = distributed_inputs[2][res_idx, ...]
                source_pred, kappa_pred, chi_squared = rim.predict(lens_true[None, ...])
                lens_pred = phys.forward(source_pred[-1], kappa_pred[-1])[0, ...]
                tf.summary.image(f"Residuals {res_idx}",
                                 plot_to_image(
                                     residual_plot(
                                         lens_true, source_true, kappa_true, lens_pred, source_pred[-1][0, ...],
                                         kappa_pred[-1][0, ...], chi_squared[-1][0]
                                     )), step=step)

            # ========== Validation set ===================
            val_loss.reset_states()
            val_chi_squared.reset_states()
            for distributed_inputs in val_dataset:
                cost, chi_squared = distributed_test_step(distributed_inputs)
                val_loss.update_state([cost])
                val_chi_squared.update_state([chi_squared])

            for res_idx in range(min(args.n_residuals, args.batch_size)):
                lens_true = distributed_inputs[0][res_idx, ...]
                source_true = distributed_inputs[1][res_idx, ...]
                kappa_true = distributed_inputs[2][res_idx, ...]
                source_pred, kappa_pred, chi_squared = rim.predict(lens_true[None, ...])
                lens_pred = phys.forward(source_pred[-1], kappa_pred[-1])[0, ...]
                tf.summary.image(f"Val Residuals {res_idx}",
                                 plot_to_image(
                                     residual_plot(
                                         lens_true, source_true, kappa_true, lens_pred, source_pred[-1][0, ...],
                                         kappa_pred[-1][0, ...], chi_squared[-1][0]
                                     )), step=step)
        val_cost = val_loss.result().numpy()
        val_chi_sq = val_chi_squared.result().numpy()
        train_cost = epoch_loss.result().numpy()
        train_chi_sq = epoch_chi_squared.result().numpy()
        tf.summary.scalar("Val MSE", val_cost, step=step)
        tf.summary.scalar("Learning Rate", optim.lr(step), step=step)
        tf.summary.scalar("Val Chi Squared", val_chi_sq, step=step)
        print(f"epoch {epoch} | train loss {train_cost:.3e} | val loss {val_cost:.3e} "
              f"| learning rate {optim.lr(step).numpy():.2e} | time per step {time_per_step.result().numpy():.2e} s")
        history["train_cost"].append(train_cost)
        history["val_cost"].append(val_cost)
        history["learning_rate"].append(optim.lr(step).numpy())
        history["train_chi_squared"].append(train_chi_sq)
        history["val_chi_squared"].append(val_chi_sq)
        history["time_per_step"].append(time_per_step.result().numpy())

        cost = train_cost if args.track_train else val_cost
        if np.isnan(cost):
            print("Training broke the Universe")
            break
        if cost < (1 - args.tolerance) * best_loss:
            best_loss = cost
            patience = args.patience
        else:
            patience -= 1
        if save_checkpoint:
            checkpoint_manager.checkpoint.step.assign_add(1) # a bit of a hack
            if epoch % args.checkpoints == 0 or patience == 0 or epoch == args.epochs - 1:
                with open(os.path.join(checkpoints_dir, "score_sheet.txt"), mode="a") as f:
                    np.savetxt(f, np.array([[lastest_checkpoint, cost]]))
                lastest_checkpoint += 1
                checkpoint_manager.save()
                print("Saved checkpoint for step {}: {}".format(int(checkpoint_manager.checkpoint.step), checkpoint_manager.latest_checkpoint))
        if patience == 0:
            print("Reached patience")
            break
    return history, best_loss


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--model_id",               default="None",                 help="Start from this model id checkpoint. None means start from scratch")
    parser.add_argument("--load_checkpoint",        default="best",                 help="One of 'best', 'lastest' or the specific checkpoint index.")
    parser.add_argument("--datasets",               required=True,  nargs="+",      help="Path to directories that contains tfrecords of dataset. Can be multiple inputs (space separated)")
    parser.add_argument("--compression_type",       default=None,                   help="Compression type used to write data. Default assumes no compression.")

    # RIM hyperparameters
    parser.add_argument("--steps",              default=16,     type=int,       help="Number of time steps of RIM")
    parser.add_argument("--adam",               action="store_true",            help="ADAM update for the log-likelihood gradient.")
    parser.add_argument("--kappalog",           action="store_true")
    parser.add_argument("--kappa_normalize",    action="store_true")
    parser.add_argument("--source_link",        default="identity",             help="One of 'exp', 'source' or 'identity' (default).")
    parser.add_argument("--kappa_init",         default=1e-1,   type=float,     help="Initial value of kappa for RIM")
    parser.add_argument("--source_init",        default=1e-3,   type=float,     help="Initial value of source for RIM")

    # Shared Unet params
    parser.add_argument("--filters",                                    default=32,     type=int)
    parser.add_argument("--filter_scaling",                             default=1,      type=int)
    parser.add_argument("--kernel_size",                                default=3,      type=int)
    parser.add_argument("--layers",                                     default=2,      type=int)
    parser.add_argument("--block_conv_layers",                          default=2,      type=int)
    parser.add_argument("--strides",                                    default=2,      type=int)
    parser.add_argument("--bottleneck_kernel_size",                     default=None,   type=int)
    parser.add_argument("--bottleneck_filters",                         default=None,   type=int)
    parser.add_argument("--resampling_kernel_size",                     default=None,   type=int)
    parser.add_argument("--gru_kernel_size",                            default=None,   type=int)
    parser.add_argument("--upsampling_interpolation",                   action="store_true")
    parser.add_argument("--kernel_regularizer_amp",                     default=1e-4,   type=float)
    parser.add_argument("--bias_regularizer_amp",                       default=1e-4,   type=float)
    parser.add_argument("--activation",                                 default="leaky_relu")
    parser.add_argument("--alpha",                                      default=0.1,    type=float)
    parser.add_argument("--initializer",                                default="glorot_normal")
    parser.add_argument("--kappa_resize_filters",                       default=4,      type=int)
    parser.add_argument("--kappa_resize_method",                        default="bilinear")
    parser.add_argument("--kappa_resize_conv_layers",                   default=1,      type=int)
    parser.add_argument("--kappa_resize_strides",                       default=2,      type=int)
    parser.add_argument("--kappa_resize_kernel_size",                   default=3,      type=int)
    parser.add_argument("--kappa_resize_separate_grad_downsampling",    action="store_true")

    # Physical model hyperparameter
    parser.add_argument("--forward_method",         default="conv2d",               help="One of ['conv2d', 'fft', 'unet']. If the option 'unet' is chosen, the parameter "
                                                                                         "'--raytracer' must be provided and point to model checkpoint directory.")
    parser.add_argument("--raytracer",              default=None,                   help="Path to raytracer checkpoint dir if method 'unet' is used.")

    # Training set params
    parser.add_argument("-b", "--batch_size",       default=1,      type=int,       help="Number of images in a batch. ")
    parser.add_argument("--train_split",            default=0.8,    type=float,     help="Fraction of the training set.")
    parser.add_argument("--total_items",            required=True,  type=int,       help="Total images in an epoch.")
    # ... for tfrecord dataset
    parser.add_argument("--num_parallel_reads",     default=10,     type=int,       help="TFRecord dataset number of parallel reads when loading data.")
    parser.add_argument("--cache_file",             default=None,                   help="Path to cache file, useful when training on server. Use ${SLURM_TMPDIR}/cache")
    parser.add_argument("--cycle_length",           default=4,      type=int,       help="Number of files to read concurrently.")
    parser.add_argument("--block_length",           default=1,      type=int,       help="Number of example to read from each files.")

    # Optimization params
    parser.add_argument("-e", "--epochs",           default=10,     type=int,       help="Number of epochs for training.")
    parser.add_argument("--optimizer",              default="Adam",                 help="Class name of the optimizer (e.g. 'Adam' or 'Adamax')")
    parser.add_argument("--initial_learning_rate",  default=1e-3,   type=float,     help="Initial learning rate.")
    parser.add_argument("--decay_rate",             default=1.,     type=float,     help="Exponential decay rate of learning rate (1=no decay).")
    parser.add_argument("--decay_steps",            default=1000,   type=int,       help="Decay steps of exponential decay of the learning rate.")
    parser.add_argument("--staircase",              action="store_true",            help="Learning rate schedule only change after decay steps if enabled.")
    parser.add_argument("--clipping",               action="store_true",            help="Clip backprop gradients between -10 and 10.")
    parser.add_argument("--patience",               default=np.inf, type=int,       help="Number of step at which training is stopped if no improvement is recorder.")
    parser.add_argument("--tolerance",              default=0,      type=float,     help="Current score <= (1 - tolerance) * best score => reset patience, else reduce patience.")
    parser.add_argument("--track_train",                    action="store_true",    help="Track training metric instead of validation metric, in case we want to overfit")

    # logs
    parser.add_argument("--logdir",                  default="None",                help="Path of logs directory. Default if None, no logs recorded.")
    parser.add_argument("--logname",                 default=None,                  help="Overwrite name of the log with this argument")
    parser.add_argument("--logname_prefixe",         default="RIMUnet512",          help="If name of the log is not provided, this prefix is prepended to the date")
    parser.add_argument("--model_dir",               default="None",                help="Path to the directory where to save models checkpoints.")
    parser.add_argument("--checkpoints",             default=10,    type=int,       help="Save a checkpoint of the models each {%} iteration.")
    parser.add_argument("--max_to_keep",             default=3,     type=int,       help="Max model checkpoint to keep.")
    parser.add_argument("--n_residuals",             default=1,     type=int,       help="Number of residual plots to save. Add overhead at the end of an epoch only.")

    # Reproducibility params
    parser.add_argument("--seed",                   default=None,   type=int,       help="Random seed for numpy and tensorflow.")
    parser.add_argument("--json_override",          default=None,                   help="A json filepath that will override every command line parameters. "
                                                                                         "Useful for reproducibility")

    args = parser.parse_args()
    if args.seed is not None:
        tf.random.set_seed(args.seed)
        np.random.seed(args.seed)
    if args.json_override is not None:
        with open(args.json_override, "r") as f:
            json_override = json.load(f)
        args_dict = vars(args)
        args_dict.update(json_override)

    main(args)
