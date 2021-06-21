import tensorflow as tf
from censai import RayTracer512 as RayTracer
from censai.data import NISGenerator
from censai.data.alpha_tng import decode_train
from censai.utils import nullwriter
import os, glob
import numpy as np
try:
    import wandb
    wandb.init(project="censai_ray_tracer", entity="adam-alexandre01123", sync_tensorboard=True)
    wndb = True
except ImportError:
    wndb = False
    print("wandb not installed, package ignored")


def main(args):
    if wndb:
        config = wandb.config
        config.update(vars(args))
    if args.dataset == "NIS":
        train_dataset = NISGenerator(int(args.train_split * args.total_items), batch_size=args.batch_size, pixels=args.pixels)
        val_dataset = NISGenerator(int((1 - args.train_split) * args.total_items), batch_size=args.batch_size, pixels=args.pixels)
    else:
        files = glob.glob(os.path.join(args.dataset, "*.tfrecords"))
        dataset = tf.data.TFRecordDataset(files, num_parallel_reads=args.num_parallel_reads).map(decode_train)
        dataset = dataset.batch(args.batch_size).cache(args.cache_file).prefetch(tf.data.experimental.AUTOTUNE)
        train_dataset = dataset.take(int(args.train_split * args.total_items))
        val_dataset = dataset.skip(int(args.train_split * args.total_items))
    ray_tracer = RayTracer(
        initializer=args.initializer,
        bottleneck_kernel_size=args.bottleneck_kernel_size,
        bottleneck_strides=args.bottleneck_strides,
        pre_bottleneck_kernel_size=args.pre_bottleneck_kernel_size,
        decoder_encoder_kernel_size=args.decoder_encoder_kernel_size,
        decoder_encoder_filters=args.decoder_encoder_filters,
        upsampling_interpolation=args.upsampling_interpolation,  # use strided transposed convolution if false
        kernel_regularizer_amp=args.kernel_regularizer_amp,
        activation=args.activation,
        filter_scaling=args.filter_scaling
    )
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        args.initial_learning_rate,
        decay_steps=args.decay_steps,
        decay_rate=args.decay_rate,
        staircase=True)
    optim = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999)

    # setup tensorboard writer (nullwriter in case we do not want to sync)
    if args.logdir.lower() != "none":
        logdir = os.path.join(args.logdir, args.logname)
        traindir = os.path.join(logdir, "train")
        testdir = os.path.join(logdir, "test")
        if not os.path.isdir(logdir):
            os.mkdir(logdir)
        if not os.path.isdir(traindir):
            os.mkdir(traindir)
        if not os.path.isdir(testdir):
            os.mkdir(testdir)
        train_writer = tf.summary.create_file_writer(traindir)
        test_writer = tf.summary.create_file_writer(testdir)
    else:
        train_writer = nullwriter()
        test_writer = nullwriter()

    if args.model_dir.lower() != "none":
        models_dir = os.path.join(args.model_dir, args.logname)
        if not os.path.isdir(models_dir):
            os.mkdir(models_dir)
        checkpoints_dir = os.path.join(models_dir, "source_checkpoints")
        if not os.path.isdir(checkpoints_dir):
            os.mkdir(checkpoints_dir)
            # save script parameter for future reference
            import json
            with open(os.path.join(checkpoints_dir, "script_params.json"), "w") as f:
                json.dump(vars(args), f)
        ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optim, net=ray_tracer)
        checkpoint_manager = tf.train.CheckpointManager(ckpt, checkpoints_dir, max_to_keep=args.max_to_keep)
        save_checkpoint = True
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
    epoch_loss = tf.metrics.Mean()
    val_loss = tf.metrics.Mean()
    best_loss = np.inf
    patience = args.patience
    step = 1
    lastest_checkpoint = 1
    for epoch in range(1, args.epochs + 1):
        epoch_loss.reset_states()
        with train_writer.as_default():
            for batch, (kappa, alpha) in enumerate(train_dataset):
                # ========== Forward and backprop ==========
                with tf.GradientTape(watch_accessed_variables=True) as tape:
                    tape.watch(ray_tracer.trainable_weights)
                    cost = tf.reduce_mean(tf.square(ray_tracer(kappa) - alpha))
                    cost += tf.reduce_sum(ray_tracer.losses)  # add L2 regularizer loss
                gradient = tape.gradient(cost, ray_tracer.trainable_weights)
                if args.clipping:
                    clipped_gradient = [tf.clip_by_value(grad, -10, 10) for grad in gradient]
                else:
                    clipped_gradient = gradient
                optim.apply_gradients(zip(clipped_gradient, ray_tracer.trainable_variables))
        # ========== Summary and logs ==========
                epoch_loss.update_state([cost])
                tf.summary.scalar("MSE", cost, step=step)
                step += 1
        with test_writer.as_default():
            val_loss.reset_states()
            for kappa, alpha in val_dataset:
                cost = tf.reduce_mean(tf.square(ray_tracer(kappa) - alpha))
                cost += tf.reduce_sum(ray_tracer.losses)
                val_loss.update_state([cost])
        train_cost = epoch_loss.result().numpy()
        val_cost = val_loss.result().numpy()
        print(f"epoch {epoch} | train loss {train_cost:.3e} | val loss {val_cost:.3e} | learning rate {optim.lr(step).numpy():.2e}")
        if val_cost < best_loss * (1 - args.tolerance):
            best_loss = val_cost
            patience = args.patience
        else:
            patience -= 1

        if save_checkpoint:
            checkpoint_manager.checkpoint.step.assign_add(1)  # a bit of a hack
            if epoch % args.checkpoints == 0 or patience == 0 or epoch == args.epochs - 1:
                with open(os.path.join(checkpoints_dir, "score_sheet.txt"), mode="a") as f:
                    np.savetxt(f, np.array([lastest_checkpoint, val_cost]))
                lastest_checkpoint += 1
                checkpoint_manager.save()
                print("Saved checkpoint for step {}: {}".format(int(checkpoint_manager.checkpoint.step), checkpoint_manager.latest_checkpoint))
        if patience == 0:
            print("Reached patience")
            break


if __name__ == "__main__":
    from argparse import ArgumentParser
    from datetime import datetime
    import json
    date = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    parser = ArgumentParser()
    parser.add_argument("--model_id",        default="None", help="Start training from previous "
                                                           "checkpoint of this model if provided")
    parser.add_argument("--load_checkpoint", default="best", help="One of 'best', 'lastest' or the specific checkpoint index")

    # Model hyper parameters
    parser.add_argument("--initializer",                    default="glorot_uniform", type=str,   help="Weight initializer")
    parser.add_argument("--decoder_encoder_kernel_size",    default=3,                type=int,   help="Main kernel size")
    parser.add_argument("--pre_bottleneck_kernel_size",     default=6,                type=int,   help="Kernel size of layer before bottleneck")
    parser.add_argument("--bottleneck_kernel_size",         default=16,               type=int,   help="Kernel size of bottleneck layr, should be twice bottleneck feature map size")
    parser.add_argument("--bottleneck_strides",             default=4,                type=int,   help="Strided of the downsampling convolutional layer before bottleneck")
    parser.add_argument("--decoder_encoder_filters",        default=32,               type=int,   help="Number of filters of conv layers")
    parser.add_argument("--filter_scaling",                 default=1,                type=float, help="Scaling of the number of filters at each layers (1=no scaling)")
    parser.add_argument("--upsampling_interpolation",       default=False,            type=bool,  help="True: Use Bilinear interpolation for upsampling, False use Fractional Striding Convolution")
    parser.add_argument("--activation",                     default="linear",         type=str,   help="Non-linearity of layers")
    parser.add_argument("--kernel_regularizer_amp",         default=1e-3,             type=float, help="l2 regularization on weights")
    parser.add_argument("--kappalog",                       default=True,             type=bool,  help="Input is log of kappa")
    parser.add_argument("--normalize",                      default=False,            type=bool,  help="Normalize log of kappa with max and minimum values defined in definitions.py")

    # Training set params
    parser.add_argument("-b", "--batch_size",               default=10,  type=int, help="Number of images in a batch")
    parser.add_argument("--dataset",                        default="NIS",    help="Dataset to use, either path to directory that contains alpha labels tfrecords "
                                                                                   "or the name of the dataset tu use. Options are ['NIS'].")
    parser.add_argument("--train_split",                    default=0.8, type=float, help="Fraction of the training set")
    parser.add_argument("--total_items", required=True,                  type=int, help="Total images in an epoch.")

    # For NIS dataset
    parser.add_argument("--pixels",                         default=512, type=int, help="When using NIS, size of the image to generate")

    # for tfrecord dataset
    parser.add_argument("--num_parallel_reads",             default=10, type=int, help="TFRecord dataset number of parallel reads when loading data")
    parser.add_argument("--cache_file",                     default=None, help="Path to cache file, useful when training on server. Use ${SLURM_TMPDIR}/cache")

    # Logs
    parser.add_argument("--logdir",                         default="None",        help="Path of logs directory.")
    parser.add_argument("--logname",                        default="RayTracer_" + date,  help="Name of the logs, default is 'RT_' + date")
    parser.add_argument("--model_dir",                      default="None",        help="Directory where to save model weights")
    parser.add_argument("--checkpoints",                    default=10, type=int,  help="Save a checkpoint of the models each {%} iteration")
    parser.add_argument("--max_to_keep",                    default=3, type=int,   help="Max model checkpoint to keep")

    # Optimization params
    parser.add_argument("-e", "--epochs",                   default=10,     type=int,   help="Number of epochs for training.")
    parser.add_argument("--initial_learning_rate",          default=1e-3,   type=float, help="Initial learning rate.")
    parser.add_argument("--decay_rate",                     default=1.,     type=float, help="Exponential decay rate of learning rate (1=no decay).")
    parser.add_argument("--decay_steps",                    default=1000,   type=int,   help="Decay steps of exponential decay of the learning rate.")
    parser.add_argument("--clipping",                       default=True,   type=bool,  help="Clip backprop gradients between -10 and 10")
    parser.add_argument("--patience",                       default=np.inf, type=int, help="Number of step at which training is stopped if no improvement is recorder")
    parser.add_argument("--tolerance",                      default=0,      type=float, help="Current score <= (1 - tolerance) * best score => reset patience, else reduce patience.")

    # Reproducibility params
    parser.add_argument("--seed", default=None, type=int, help="Random seed for numpy and tensorflow")
    parser.add_argument("--json_override", default=None,
                        help="A json filepath that will override every command line parameters. "
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
