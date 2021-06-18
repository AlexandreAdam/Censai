import tensorflow as tf
from censai import RayTracer512 as RayTracer
from censai.data_generator import NISGenerator
from censai.utils import nullwriter
import os
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
        config.update(args)
    gen = NISGenerator(args.total_items, args.batch_size, pixels=args.pixels)
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
    best_loss = np.inf
    patience = args.patience
    step = 1
    lastest_checkpoint = 1
    if args.kappalog:
        link = lambda x: tf.math.log(x) / tf.math.log(10.)
    else:
        link = lambda x: x
    for epoch in range(1, args.epochs + 1):
        epoch_loss.reset_states()
        with train_writer.as_default():
            for batch, (kappa, alpha) in enumerate(gen):
                # ========== Forward and backprop ==========
                with tf.GradientTape(watch_accessed_variables=True) as tape:
                    tape.watch(ray_tracer.trainable_weights)
                    cost = tf.reduce_mean(tf.square(ray_tracer(link(kappa)) - alpha)) # also MSE
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
        train_cost = epoch_loss.result().numpy()
        print(f"epoch {epoch} | train loss {train_cost:.3e} | learning rate {optim.lr(step).numpy():.2e}")
        if train_cost < best_loss * (1 - args.tolerance):
            best_loss = train_cost
            patience = args.patience
        else:
            patience -= 1

        if save_checkpoint:
            checkpoint_manager.checkpoint.step.assign_add(1)  # a bit of a hack
            if epoch % args.checkpoints == 0 or patience == 0 or epoch == args.epochs - 1:
                with open(os.path.join(checkpoints_dir, "score_sheet.txt"), mode="a") as f:
                    np.savetxt(f, np.array([lastest_checkpoint, train_cost]))
                lastest_checkpoint += 1
                checkpoint_manager.save()
                print("Saved checkpoint for step {}: {}".format(int(checkpoint_manager.checkpoint.step), checkpoint_manager.latest_checkpoint))
        if patience == 0:
            print("Reached patience")
            break


if __name__ == "__main__":
    from argparse import ArgumentParser
    from datetime import datetime
    date = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    parser = ArgumentParser()
    parser.add_argument("--model_id",        default="None", help="Start training from previous "
                                                           "checkpoint of this model if provided")
    parser.add_argument("--load_checkpoint", default="best", help="One of 'best', 'lastest' or the specific checkpoint index")

    # Model hyper parameters
    parser.add_argument("--initializer",                    default="random_uniform", type=str,   help="Weight initializer")
    parser.add_argument("--decoder_encoder_kernel_size",    default=3,                type=int,   help="Main kernel size")
    parser.add_argument("--pre_bottleneck_kernel_size",     default=6,                type=int,   help="Kernel size of layer before bottleneck")
    parser.add_argument("--bottleneck_kernel_size",         default=16,               type=int,   help="Kernel size of bottleneck layr, should be twice bottleneck feature map size")
    parser.add_argument("--decoder_encoder_filters",        default=32,               type=int,   help="Number of filters of conv layers")
    parser.add_argument("--filter_scaling",                 default=1,                type=float, help="Scaling of the number of filters at each layers (1=no scaling)")
    parser.add_argument("--upsampling_interpolation",       default=False,            type=bool,  help="True: Use Bilinear interpolation for upsampling, False use Fractional Striding Convolution")
    parser.add_argument("--activation",                     default="linear",         type=str,   help="Non-linearity of layers")
    parser.add_argument("--kernel_regularizer_amp",         default=1e-3,             type=float, help="l2 regularization on weights")
    parser.add_argument("--kappalog",                       default=True,             type=bool,  help="Input is log of kappa")
    parser.add_argument("--normalize",                      default=False,            type=bool,  help="Normalize log of kappa with max and minimum values defined in definitions.py")

    # Training set params
    parser.add_argument("-t", "--total_items",              default=100, type=int, help="Total images in an epoch")
    parser.add_argument("-b", "--batch_size",               default=10,  type=int, help="Number of images in a batch")

    # Logs
    parser.add_argument("--logdir",                         default="None",        help="Path of logs directory.")
    parser.add_argument("--logname",                        default="RT_" + date,  help="Name of the logs, default is the local date + time")
    parser.add_argument("--model_dir",                      default="None",        help="Directory where to save model weights")
    parser.add_argument("--checkpoints",                    default=10, type=int,  help="Save a checkpoint of the models each {%} iteration")
    parser.add_argument("--max_to_keep",                    default=3, type=int,   help="Max model checkpoint to keep")

    # Optimization params
    parser.add_argument("-e", "--epochs",                   default=10,     type=int,   help="Number of epochs for training.")
    parser.add_argument("--initial_learning_rate",          default=1e-3,   type=float, help="Initial learning rate.")
    parser.add_argument("--decay_rate",                     default=1.,     type=float, help="Exponential decay rate of learning rate (1=no decay).")
    parser.add_argument("--decay_steps",                    default=1000,   type=int,   help="Decay steps of exponential decay of the learning rate.")
    parser.add_argument("--clipping",                       default=True,   type=bool,  help="Clip backprop gradients between -10 and 10")
    parser.add_argument("--patience",                       default=np.inf, type=float, help="Number of step at which training is stopped if no improvement is recorder")
    parser.add_argument("--tolerance",                      default=0,      type=float, help="Current score <= (1 - tolerance) * best score => reset patience, else reduce patience.")

    args = parser.parse_args()
    main(args)
