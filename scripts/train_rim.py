import tensorflow as tf
import numpy as np
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
    gen_test = NISGenerator(args.validation, args.validation, train=False, model="rim", pixels=args.pixels)
    phys = PhysicalModel(pixels=args.pixels, noise_rms=args.noise_rms)
    rim = RIM(phys, args.batch_size, args.time_steps, args.pixels)
    optim = tf.optimizers.Adam(args.lr)
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

    epoch_loss = tf.metrics.Mean()
    best_loss = np.inf
    patience = args.patience
    if args.kappalog:
        link = lambda x: tf.math.log(x + 1e-10) / np.log(10.)
    else:
        link = lambda x: x
    step = 1
    for epoch in range(args.epochs):
        epoch_loss.reset_states()
        with train_writer.as_default():
            for batch, (X, source, kappa) in enumerate(gen):
                with tf.GradientTape(persistent=True, watch_accessed_variables=True) as tape:
                    tape.watch(rim.model_1.trainable_variables)
                    tape.watch(rim.model_2.trainable_variables)
                    cost = rim.cost_function(X, source, link(kappa))
                gradient1 = tape.gradient(cost, rim.model_1.trainable_variables)
                gradient2 = tape.gradient(cost, rim.model_2.trainable_variables)
                # clipped_gradient = [tf.clip_by_value(grad, -10, 10) for grad in gradient]
                optim.apply_gradients(zip(gradient1, rim.model_1.trainable_variables)) # backprop
                optim.apply_gradients(zip(gradient2, rim.model_2.trainable_variables))

                #========== Summary and logs ==========
                epoch_loss.update_state([cost])
                tf.summary.scalar("MSE", cost, step=step)
                step += 1
            # tf.summary.scalar("Learning Rate", optim.lr(step), step=step)
        with test_writer.as_default():
            for (X, source, kappa) in gen_test:
                test_cost = rim.cost_function(X, source,  tf.math.log(kappa + 1e-10) / tf.math.log(10.))
            tf.summary.scalar("MSE", test_cost, step=step)
        print(f"epoch {epoch} | train loss {epoch_loss.result().numpy():.3e} | val loss {test_cost.numpy():.3e}") #| learning rate {optim.lr(step).numpy():.2e}")
        if test_cost < best_loss - args.tolerance:
            best_loss = test_cost
            patience = args.patience
        else:
            patience -= 1
        if patience == 0:
            print("Reached patience")
            break


if __name__ == "__main__":
    from argparse import ArgumentParser
    date = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    parser = ArgumentParser()
    parser.add_argument("-t", "--total_items", default=1, type=int, required=False, help="Total images in an epoch")
    parser.add_argument("-b", "--batch_size", default=1, type=int, required=False, help="Number of images in a batch")
    parser.add_argument("--validation", required=False, default=1, type=int, help="Number of images in the validation set")
    parser.add_argument("--logdir", required=False, default="logs", help="Path of logs directory. Default assumes script is" \
            "run from the base directory of censai. For no logs, use None")
    parser.add_argument("--logname", required=False, default=date, help="Name of the logs, default is the local date + time")
    parser.add_argument("-e", "--epochs", required=False, default=1, help="Number of epochs for training")
    parser.add_argument("--lr", required=False, default=1e-3, type=float, help="Learning rate")
    parser.add_argument("--pixels", required=False, default=128, type=int, help="Number of pixels on a side") # cannot change that for the moment
    parser.add_argument("--noise_rms", required=False, default=1e-3, type=float, help="Pixel value rms of lensed image")
    parser.add_argument("--time_steps", required=False, default=8, type=int, help="Number of time steps of RIM")
    args = parser.parse_args()
    # ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt, net=rim.model)
    # manager = tf.train.CheckpointManager(ckpt, models_dir, max_to_keep=3)
    main(args)
