import tensorflow as tf
import numpy as np
from censai.models import Autoencoder
from censai.cosmos_utils import decode as decode_cosmos
from censai.utils import nullwriter
import os
from datetime import datetime
import re
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


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def main(args):
    filenames = os.listdir(args.data)
    filenames.sort(key=natural_keys)
    # keep the n last files as a test set
    filenames = filenames[:-args.test_shards]
    filenames = [os.path.join(args.data, file) for file in filenames]
    train_size = len(filenames) * args.examples_per_shard  # estimate the length of the dataset
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=args.num_parallel_reads)
    dataset = dataset.map(decode_cosmos)
    dataset = dataset.shuffle()
    train_dataset = dataset.take(int(train_size * args.split))
    test_dataset = dataset.skip(int(train_size * (1 - args.split)))
    train_dataset = train_dataset.batch(args.batch_size, drop_remainder=False)
    train_dataset = train_dataset.enumerate()
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=args.learning_rate,
        decay_rate=args.decay_rate,
        decay_steps=args.decay_steps,
        staircase=args.staircase
    )

    optim = tf.optimizers.Adam(learning_rate=learning_rate_schedule)
    AE = Autoencoder(
        pixels=args.pixels,
        res_layers=args.res_layers,
        conv_layers_in_resblock=args.conv_layers_in_res_block,
        filter_scaling=args.filter_scaling,
        filter_init=args.filter_init,
        kernel_size=args.kernel_size,
        res_architecture=args.res_architecture,
        kernel_reg_amp=args.kernel_reg_amp,
        bias_reg_amp=args.bias_reg_amp,
        alpha=args.relu_alpha,
        resblock_dropout_rate=args.resblock_dropout_rate,
        latent_size=args.latent_size
    )

    # setup tensorboard writer (nullwriter in case we do not want to sync)
    if args.model_id.lower() != "none":
        logname = args.model_id
    else:
        logname = args.logname
    if args.logdir.lower() != "none":
        logdir = os.path.join(args.logdir, logname)
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
        models_dir = os.path.join(args.model_dir, logname)
        if not os.path.isdir(models_dir):
            os.mkdir(models_dir)
        encoder_checkpoints_dir = os.path.join(models_dir, "encoder_checkpoints")
        if not os.path.isdir(encoder_checkpoints_dir):
            os.mkdir(encoder_checkpoints_dir)
        decoder_checkpoints_dir = os.path.join(models_dir, "decoder_checkpoints")
        if not os.path.isdir(decoder_checkpoints_dir):
            os.mkdir(decoder_checkpoints_dir)
        encoder_ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optim, net=AE.encoder)
        encoder_checkpoint_manager = tf.train.CheckpointManager(encoder_ckpt, encoder_checkpoints_dir, max_to_keep=3)
        decoder_ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optim, net=AE.decoder)
        decoder_checkpoint_manager = tf.train.CheckpointManager(decoder_ckpt, decoder_checkpoints_dir, max_to_keep=3)
        save_checkpoint = True
        if args.model_id.lower() != "none":
            encoder_checkpoint_manager.checkpoint.restore(encoder_checkpoint_manager.latest_checkpoint)
            decoder_checkpoint_manager.checkpoint.restore(decoder_checkpoint_manager.latest_checkpoint)
    else:
        save_checkpoint = False

    epoch_loss = tf.metrics.Mean()
    best_loss = np.inf
    patience = args.patience
    step = 0
    for epoch in range(args.epochs):
        epoch_loss.reset_states()
        with train_writer.as_default():
            for batch, (X, PSF, PS) in train_dataset:
                with tf.GradientTape() as tape:
                    tape.watch(AE.trainable_variables)
                    cost = tf.reduce_mean(AE.training_cost_function(X, PSF, PS))
                    cost += tf.reduce_sum(AE.losses)  # Add layer specific regularizer losses (L2 in definitions)
                gradient = tape.gradient(cost, AE.trainable_variables)
                # clipped_gradient = [tf.clip_by_value(grad, -10, 10) for grad in gradient]
                optim.apply_gradients(zip(gradient, AE.trainable_variables)) # backprop

                #========== Summary and logs ==========
                epoch_loss.update_state([cost])
                tf.summary.scalar("MSE", cost, step=step)
                step += 1
            tf.summary.scalar("Learning Rate", optim.lr(step), step=step)
        with test_writer.as_default():
            for batch, (X, PSF, PS) in train_dataset:
                test_cost = tf.reduce_mean(AE.training_cost_function(X, PSF, PS))
            tf.summary.scalar("MSE", test_cost, step=step)
        print(f"epoch {epoch} | train loss {epoch_loss.result().numpy():.3e} | val loss {test_cost.numpy():.3e} "
              f"| learning rate {optim.lr(step).numpy():.2e}")
        if test_cost < (1 - args.tolerance) * best_loss:
            best_loss = test_cost
            patience = args.patience
        else:
            patience -= 1
        if save_checkpoint:
            encoder_checkpoint_manager.checkpoint.step.assign_add(1) # a bit of a hack
            decoder_checkpoint_manager.checkpoint.step.assign_add(1)
            if epoch % args.checkpoints == 0 or patience == 0 or epoch == args.epochs - 1:
                encoder_checkpoint_manager.save()
                decoder_checkpoint_manager.save()
                print("Saved checkpoint for step {}: {}".format(int(encoder_checkpoint_manager.checkpoint.step),
                                                                encoder_checkpoint_manager.latest_checkpoint))
        if patience == 0:
            print("Reached patience")
            break


if __name__ == "__main__":
    date = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--model_id", type=str, default="None",
                        help="Start from this model id checkpoint. None means start from scratch")
    parser.add_argument("--pixels", required=False, default=128, type=int, help="Number of pixels on a side, should be fixed for a given cosmos tfrecord")
    parser.add_argument("--num_parallel_read", default=1, type=int, help="TF dataset number of parallel processes loading the data while training")
    parser.add_argument("--data", required=True, help="Path to the data root directory, containing tf records files")

    # training params
    parser.add_argument("--split", default=0.8, type=float, help="Training split, number in the range [0.5, 1)")
    parser.add_argument("--test_shards", default=2, type=int, help="Number of shards to keep as a test set. The largest shard index are kept")
    parser.add_argument("--examples_per_shard", default=1000, type=int,
                        help="Number of example on a given COSMO shard. Should match the parameter of cosmo_to_tfrecords with which it was generated")
    parser.add_argument("-b", "--batch_size", default=100, type=int, required=False, help="Number of images in a batch")
    parser.add_argument("-e", "--epochs", required=False, default=1, type=int, help="Number of epochs for training")
    parser.add_argument("--patience", required=False, default=np.inf, type=float, help="Number of epoch at which "
                                                                "training is stop if no improvement have been made")
    parser.add_argument("--tolerance", required=False, default=0, type=float,
                        help="Percentage [0-1] of improvement required for patience to reset. The most lenient "
                                                        "value is 0 (any improvement reset patience)")
    parser.add_argument("--learning_rate", required=False, default=1e-4, type=float)
    parser.add_argument("--decay_rate", type=float, default=1,
                        help="Decay rate of the exponential decay schedule of the learning rate. 1=no decay")
    parser.add_argument("--decay_steps", type=int, default=100)
    parser.add_argument("--staircase", action="store_true", help="Learning schedule is a staircase "
                                                                 "function if added to arguments")

    # model hyperparameters
    parser.add_argument("--res_layers", default=7, type=int,
                        help="Number of downsampling block in encoder (symmetric in decoder")
    parser.add_argument("--conv_layers_in_res_block", default=2, type=int,
                        help="Number of conv layers in a Residual block")
    parser.add_argument("--filter_scaling", default=2, type=float,
                        help="Filters scale by {filter_scaling}^{res_layer_index}, generally number between (1, 2]")
    parser.add_argument("--filer_init", default=8, type=int,
                        help="Number of filters in the first residual block (before last for decoder)")
    parser.add_argument("--kernel_size", default=3, type=int,
                        help="Size of the kernels throughout model")
    parser.add_argument("--kernel_reg_amp", default=1e-2, type=float,
                        help="Amplitude of l2 regularization for kernel weights in the model")
    parser.add_argument("--bias_reg_amp", default=1e-2, type=float,
                        help="Amplitude of l2 regularization for bias variables in the model")
    parser.add_argument("--relu_alpha", default=0.1, type=float,
                        help="Slope of LeakyReLu in the negative plane")
    parser.add_argument("--resblock_dropout_rate", default=None, type=float,
                        help="Number between [0, 1), number of filters to drop at each call. Default is to not use dropout")
    parser.add_argument("--latent_size", default=16, type=int,
                        help="Size of the latent vector space")

    # logs
    parser.add_argument("--logdir", required=False, default="None",
                        help="Path of logs directory. Default if None, no logs recorded")
    parser.add_argument("--model_dir", required=False, default="None",
                        help="Path to the directory where to save models checkpoints")
    parser.add_argument("--checkpoints", required=False, default=10, type=int,
                        help="Save a checkpoint of the models each {%} iteration")
    parser.add_argument("--max_to_keep", required=False, default=3, type=int,
                        help="Max model checkpoint to keep")
    parser.add_argument("--logname", required=False, default="cosmosAE_" + date,
                        help="Name of the logs, default is the local date + time")

    args = parser.parse_args()

    main(args)
