import tensorflow as tf
from censai import RayTracer512 as RayTracer
from censai.data_generator import NISGenerator
from censai.utils import nullwriter
import os
from datetime import datetime
# try:
#     import wandb
#     wandb.init(project="censai", entity="adam-alexandre01123", sync_tensorboard=True)
# except ImportError:
#     print("wandb not installed, package ignored")


def main(args):
    gen = NISGenerator(args.total_items, args.batch_size)
    gen_test = NISGenerator(args.validation, args.validation, train=False)
    ray_tracer = RayTracer()
    optim = tf.optimizers.Adam(lr=args.lr)

    # setup tensorboard writer (nullwriter in case we do not want to sync)
    if args.model_id.lower() != "none":
        logname = args.model_id
    else:
        logname = args.logname
    if args.logdir.lower() != "none":
        logdir = os.path.join(args.logdir, logname)
        traindir = os.path.join(logdir, "train")
        if not os.path.isdir(logdir):
            os.mkdir(logdir)
        if not os.path.isdir(traindir):
            os.mkdir(traindir)
        train_writer = tf.summary.create_file_writer(traindir)
    else:
        train_writer = nullwriter()

    step = 1
    for epoch in range(args.epochs):
        with train_writer.as_default():
            for batch, (kappa, alpha) in enumerate(gen):
                with tf.GradientTape() as tape:
                    tape.watch(ray_tracer.trainable_variables)
                    cost = ray_tracer.cost(kappa, alpha) # call + MSE loss function
                    cost += tf.reduce_sum(ray_tracer.losses) # add regularizer loss
                gradient = tape.gradient(cost, ray_tracer.trainable_variables)
                clipped_gradient = [tf.clip_by_value(grad, -10, 10) for grad in gradient]
                optim.apply_gradients(zip(clipped_gradient, ray_tracer.trainable_variables)) # backprop

                #========== Summary and logs ==========
                tf.summary.scalar("MSE", cost, step=step)
                step += 1

if __name__ == "__main__":
    from argparse import ArgumentParser
    date = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    parser = ArgumentParser()
    parser.add_argument("-t", "--total_items", default=100, type=int, required=False, help="Total images in an epoch")
    parser.add_argument("-b", "--batch_size", default=10, type=int, required=False, help="Number of images in a batch")
    parser.add_argument("--validation", required=False, default=20, type=int, help="Number of images in the validation set")
    parser.add_argument("--logdir", required=False, default="logs", help="Path of logs directory. Default assumes script is" \
            "run from the base directory of censai. For no logs, use None")
    parser.add_argument("--logname", required=False, default="RT_" + date, help="Name of the logs, default is the local date + time")
    parser.add_argument("-e", "--epochs", required=False, default=10, help="Number of epochs for training")
    parser.add_argument("--lr", required=False, default=1e-3, type=float, help="Learning rate")
    args = parser.parse_args()
    main(args)


