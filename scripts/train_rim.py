import tensorflow as tf
import numpy as np

# that won't work, network needs a few Gb of memory to lead its weight in the GPU
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
#
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
    # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
        print(e)
from censai.physical_model import PhysicalModel
from censai.data_generator import NISGenerator
from censai.rim_unet import RIM
from censai.utils import nullwriter
import os
from datetime import datetime
# try:
#     import wandb
#     wandb.init(project="censai_rim", entity="adam-alexandre01123", sync_tensorboard=True)
# except ImportError:
#     print("wandb not installed, package ignored")


def main(args):
    gen = NISGenerator(args.total_items, args.batch_size, model="rim", pixels=args.pixels)
    gen_test = NISGenerator(args.batch_size, args.batch_size, train=False, model="rim", pixels=args.pixels)
    phys = PhysicalModel(pixels=args.pixels, noise_rms=args.noise_rms)
    rim = RIM(phys, args.batch_size, args.time_steps, args.pixels, adam=args.adam, strides=args.strides)
    learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=args.learning_rate,
        decay_rate=args.decay_rate,
        decay_steps=args.decay_steps,
        staircase=args.staircase
    )
    optim = tf.optimizers.Adam(learning_rate=learning_rate_schedule)
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
        test_writer = nullwriter()
        train_writer = nullwriter()
    if args.model_dir.lower() != "none":
        models_dir = os.path.join(args.model_dir, args.logname)
        if not os.path.isdir(models_dir):
            os.mkdir(models_dir)
        source_checkpoints_dir = os.path.join(models_dir, "source_checkpoints")
        if not os.path.isdir(source_checkpoints_dir):
            os.mkdir(source_checkpoints_dir)
        kappa_checkpoints_dir = os.path.join(models_dir, "kappa_checkpoints")
        if not os.path.isdir(kappa_checkpoints_dir):
            os.mkdir(kappa_checkpoints_dir)
        source_ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optim, net=rim.source_model)
        source_checkpoint_manager = tf.train.CheckpointManager(source_ckpt, source_checkpoints_dir, max_to_keep=3)
        kappa_ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optim, net=rim.kappa_model)
        kappa_checkpoint_manager = tf.train.CheckpointManager(kappa_ckpt, kappa_checkpoints_dir, max_to_keep=3)
        save_checkpoint = True
        if args.model_id.lower() != "none":
            kappa_checkpoint_manager.checkpoint.restore(kappa_checkpoint_manager.latest_checkpoint)
            source_checkpoint_manager.restore(source_checkpoint_manager.latest_checkpoint)
    else:
        save_checkpoint = False



    epoch_loss = tf.metrics.Mean()
    best_loss = np.inf
    patience = args.patience
    step = 0
    for epoch in range(args.epochs):
        epoch_loss.reset_states()
        with train_writer.as_default():
            for batch, (X, source, kappa) in enumerate(gen):
                with tf.GradientTape(persistent=True, watch_accessed_variables=True) as tape:
                    tape.watch(rim.source_model.trainable_variables)
                    tape.watch(rim.kappa_model.trainable_variables)
                    cost = rim.cost_function(X, source, kappa)
                gradient1 = tape.gradient(cost, rim.source_model.trainable_variables)
                gradient2 = tape.gradient(cost, rim.kappa_model.trainable_variables)
                # clipped_gradient = [tf.clip_by_value(grad, -10, 10) for grad in gradient]
                optim.apply_gradients(zip(gradient1, rim.source_model.trainable_variables)) # backprop
                optim.apply_gradients(zip(gradient2, rim.kappa_model.trainable_variables))

                #========== Summary and logs ==========
                epoch_loss.update_state([cost])
                tf.summary.scalar("MSE", cost, step=step)
                step += 1
            tf.summary.scalar("Learning Rate", optim.lr(step), step=step)
        with test_writer.as_default():
            for (X, source, kappa) in gen_test:
                test_cost = rim.cost_function(X, source,  tf.math.log(kappa + 1e-10) / tf.math.log(10.))
            tf.summary.scalar("MSE", test_cost, step=step)
        print(f"epoch {epoch} | train loss {epoch_loss.result().numpy():.3e} | val loss {test_cost.numpy():.3e} "
              f"| learning rate {optim.lr(step).numpy():.2e}")
        if test_cost < (1 - args.tolerance) * best_loss:
            best_loss = test_cost
            patience = args.patience
        else:
            patience -= 1
        if save_checkpoint:
            source_checkpoint_manager.checkpoint.step.assign_add(1) # a bit of a hack
            kappa_checkpoint_manager.checkpoint.step.assign_add(1)
            if epoch % args.checkpoints == 0 or patience == 0 or epoch == args.epochs - 1:
                source_checkpoint_manager.save()
                kappa_checkpoint_manager.save()
                print("Saved checkpoint for step {}: {}".format(int(source_checkpoint_manager.checkpoint.step),
                                                                source_checkpoint_manager.latest_checkpoint))
        if patience == 0:
            print("Reached patience")
            break


if __name__ == "__main__":
    from argparse import ArgumentParser
    date = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    parser = ArgumentParser()
    parser.add_argument("--model_id", type=str, default="None",
                        help="Start from this model id checkpoint. None means start from scratch")
    parser.add_argument("--pixels", required=False, default=64, type=int, help="Number of pixels on a side")

    # training params
    parser.add_argument("-t", "--total_items", default=100, type=int, required=False, help="Total images in an epoch")
    parser.add_argument("-b", "--batch_size", default=10, type=int, required=False, help="Number of images in a batch")
    parser.add_argument("-e", "--epochs", required=False, default=1, type=int, help="Number of epochs for training")
    parser.add_argument("--patience", required=False, default=np.inf, type=float, help="Number of epoch at which "
                                                                "training is stop if no improvement have been made")
    parser.add_argument("--tolerance", required=False, default=0, type=float,
                        help="Percentage [0-1] of improvement required for patience to reset. The most lenient "
                                                        "value is 0 (any improvement reset patience)")

    # hyperparameters
    parser.add_argument("--learning_rate", required=False, default=1e-4, type=float)
    parser.add_argument("--decay_rate", type=float, default=1,
                        help="Decay rate of the exponential decay schedule of the learning rate. 1=no decay")
    parser.add_argument("--decay_steps", type=int, default=100)
    parser.add_argument("--staircase", action="store_true", help="Learning schedule is a staircase "
                                                                 "function if added to arguments")
    parser.add_argument("--noise_rms", required=False, default=1e-3, type=float, help="Pixel value rms of lensed image")
    parser.add_argument("--time_steps", required=False, default=16, type=int, help="Number of time steps of RIM")
    parser.add_argument("--kappalog", required=False, default=True, type=bool)
    parser.add_argument("--adam", required=False, default=True, type=bool,
                        help="ADAM update for the log-likelihood gradient")
    parser.add_argument("--strides", required=False, default=2, type=int, help="Value of the stride parameter in the 3 "
                                                    "downsampling and upsampling layers")
    # logs
    parser.add_argument("--logdir", required=False, default="None",
                        help="Path of logs directory. Default if None, no logs recorded")
    parser.add_argument("--logname", required=False, default=date,
                        help="Name of the logs, default is the local date + time")
    parser.add_argument("--model_dir", required=False, default="None",
                        help="Path to the directory where to save models checkpoints")
    parser.add_argument("--checkpoints", required=False, default=10,
                        help="Save a checkpoint of the models each {%} iteration")
    parser.add_argument("--max_to_keep", required=False, default=3,
                        help="Max model checkpoint to keep")
    args = parser.parse_args()

    main(args)
