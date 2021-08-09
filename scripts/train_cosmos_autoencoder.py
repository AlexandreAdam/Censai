import tensorflow as tf
import numpy as np
from censai.models import CosmosAutoencoder
from censai.data.cosmos import decode, preprocess, decode_shape
from censai.utils import nullwriter, vae_residual_plot as residual_plot
from censai.definitions import PolynomialSchedule
import os
from datetime import datetime
import math, glob


def main(args):
    files = []
    for dataset in args.datasets:
        files.extend(glob.glob(os.path.join(dataset, "*.tfrecords")))
    np.random.shuffle(files)
    # Read concurrently from multiple records
    files = tf.data.Dataset.from_tensor_slices(files)
    dataset = files.interleave(lambda x: tf.data.TFRecordDataset(x, compression_type=args.compression_type),
                               block_length=args.block_length, num_parallel_calls=tf.data.AUTOTUNE)

    # Read off global parameters from first example in dataset
    for pixels in dataset.map(decode_shape):
        break
    dataset = dataset.map(decode).map(preprocess).batch(args.batch_size)
    if args.cache_file is not None:
        dataset = dataset.cache(args.cache_file).prefetch(tf.data.experimental.AUTOTUNE)
    else:
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    train_dataset = dataset.take(math.floor(args.train_split * args.total_items / args.batch_size)) # dont forget to divide by batch size!
    val_dataset = dataset.skip(math.floor(args.train_split * args.total_items / args.batch_size))
    val_dataset = val_dataset.take(math.ceil((1 - args.train_split) * args.total_items / args.batch_size))

    learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=args.learning_rate,
        decay_rate=args.decay_rate,
        decay_steps=args.decay_steps,
        staircase=args.staircase
    )
    skip_strength_schedule = PolynomialSchedule(
        initial_value=args.skip_strength,
        end_value=0.,
        power=args.skip_strength_decay_power,
        decay_steps=args.skip_strength_decay_steps
    )
    l2_bottleneck_schedule = PolynomialSchedule(
        initial_value=args.l2_bottleneck,
        end_value=0.,
        power=args.l2_bottleneck_decay_power,
        decay_steps=args.l2_bottleneck_decay_steps
    )

    optim = tf.optimizers.Adam(learning_rate=learning_rate_schedule)
    AE = CosmosAutoencoder(
        pixels=pixels,
        conv_layers=args.conv_layers,
        layers=args.layers,
        filter_scaling=args.filter_scaling,
        filters=args.filters,
        kernel_size=args.kernel_size,
        kernel_reg_amp=args.kernel_reg_amp,
        bias_reg_amp=args.bias_reg_amp,
        activation=args.activation,
        latent_size=args.latent_size
    )

    # ==== Take care of where to write logs and stuff =================================================================
    if args.model_id.lower() != "none":
        logname = args.model_id
    else:
        logname = args.logname
    if args.logdir.lower() != "none":
        logdir = os.path.join(args.logdir, logname)
        if not os.path.isdir(logdir):
            os.mkdir(logdir)
        writer = tf.summary.create_file_writer(logdir)
    else:
        writer = nullwriter()
    # ===== Make sure directory and checkpoint manager are created to save model ===================================
    if args.model_dir.lower() != "none":
        model_dir = os.path.join(args.model_dir, logname)
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optim, net=AE)
        checkpoint_manager = tf.train.CheckpointManager(ckpt, model_dir, max_to_keep=3)
        save_checkpoint = True
        # ======= Load model if model_id is provided ===============================================================
        if args.model_id.lower() != "none":
            checkpoint_manager.checkpoint.restore(checkpoint_manager.latest_checkpoint)
    else:
        save_checkpoint = False
    # =================================================================================================================

    def train_step(X, PSF, PS, step):
        with tf.GradientTape() as tape:
            tape.watch(AE.trainable_variables)
            chi_squared, bottleneck_l2_cost, apo_loss, tv_loss = AE.training_cost_function(
                image=X,
                psf=PSF,
                ps=PS,
                skip_strength=skip_strength_schedule(step),
                l2_bottleneck=l2_bottleneck_schedule(step),
                apodization_alpha=args.apodization_alpha,
                apodization_factor=args.apodization_factor,
                tv_factor=args.tv_factorcost
            )
            cost = tf.reduce_mean(chi_squared + bottleneck_l2_cost + apo_loss + tv_loss)
            cost += tf.reduce_sum(AE.losses)  # Add layer specific regularizer losses (L2 in definitions)
        gradient = tape.gradient(cost, AE.trainable_variables)
        if args.clipping:
            gradient = [tf.clip_by_value(grad, -10, 10) for grad in gradient]
        optim.apply_gradients(zip(gradient, AE.trainable_variables))  # backprop
        bottleneck_l2_cost = tf.reduce_mean(bottleneck_l2_cost)
        apo_loss = tf.reduce_mean(apo_loss)
        tv_loss = tf.reduce_mean(tv_loss)
        return cost, bottleneck_l2_cost, apo_loss, tv_loss

    def test_step(X, PSF, PS, step):
        chi_squared, bottleneck_l2_cost, apo_loss, tv_loss = AE.training_cost_function(
            image=X,
            psf=PSF,
            ps=PS,
            skip_strength=skip_strength_schedule(step),
            l2_bottleneck=l2_bottleneck_schedule(step),
            apodization_alpha=args.apodization_alpha,
            apodization_factor=args.apodization_factor,
            tv_factor=args.tv_factor
        )
        cost = tf.reduce_mean(chi_squared + bottleneck_l2_cost + apo_loss + tv_loss)
        cost += tf.reduce_sum(AE.losses)  # Add layer specific regularizer losses (L2 in definitions)
        bottleneck_l2_cost = tf.reduce_mean(bottleneck_l2_cost)
        apo_loss = tf.reduce_mean(apo_loss)
        tv_loss = tf.reduce_mean(tv_loss)
        return cost, bottleneck_l2_cost, apo_loss, tv_loss

    # ====== Training loop ============================================================================================
    epoch_loss = tf.metrics.Mean()
    test_loss = tf.metrics.Mean()
    best_loss = np.inf
    patience = args.patience
    step = 0
    for epoch in range(args.epochs):
        epoch_loss.reset_states()
        test_loss.reset_states()
        with writer.as_default():
            for batch, (X, PSF, PS) in train_dataset:
                cost, bottleneck_l2_cost, apo_loss, tv_loss = train_step(X, PSF, PS, step)

                #========== Summary and logs ==========
                epoch_loss.update_state([cost])
                tf.summary.scalar("MSE", cost, step=step)
                step += 1
            tf.summary.scalar("Learning Rate", optim.lr(step), step=step)
            for X, PSF, PS in val_dataset:
                cost, bottleneck_l2_cost, apo_loss, tv_loss = test_step(X, PSF, PS, step)
                test_loss.update_state([cost])
            tf.summary.scalar("Val MSE", test_loss.result(), step=step)
        print(f"epoch {epoch} | train loss {epoch_loss.result().numpy():.3e} | val loss {test_loss.result().numpy():.3e} "
              f"| learning rate {optim.lr(step).numpy():.2e}")
        if epoch_loss.result() < (1 - args.tolerance) * best_loss:
            best_loss = epoch_loss.result()
            patience = args.patience
        else:
            patience -= 1
        if save_checkpoint:
            checkpoint_manager.checkpoint.step.assign_add(1) # a bit of a hack
            if epoch % args.checkpoints == 0 or patience == 0 or epoch == args.epochs - 1:
                checkpoint_manager.save()
                print("Saved checkpoint for step {}: {}".format(int(checkpoint_manager.checkpoint.step), checkpoint_manager.latest_checkpoint))
        if patience == 0:
            print("Reached patience")
            break


if __name__ == "__main__":
    date = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--model_id",                   default="None",             help="Start from this model id checkpoint. None means start from scratch")
    parser.add_argument("--datasets",                   required=True, nargs="+",   help="Path to the data root directory, containing tf records files")
    parser.add_argument("--compression_type",           default=None,               help="Compression type used to write data. Default assumes no compression.")

    # training params
    parser.add_argument("--train_split",                default=0.9,    type=float, help="Training split, number in the range [0.5, 1)")
    parser.add_argument("--total_items",                defualt=1000,   type=int,   help="Total item sin the dataset")
    parser.add_argument("--cache_file",                 default=None,               help="Path to cache file, useful when training on server. Use ${SLURM_TMPDIR}/cache")
    parser.add_argument("--block_length",               default=1,      type=int,   help="Number of example to read from each files at a given batch.")
    parser.add_argument("-b", "--batch_size",           default=100,    type=int,   help="Number of images in a batch")
    parser.add_argument("-e", "--epochs",               default=1,      type=int,   help="Number of epochs for training")
    parser.add_argument("--patience",                   default=np.inf, type=float, help="Number of epoch at which "
                                                                                         "training is stop if no improvement have been made")
    parser.add_argument("--tolerance",                  default=0,      type=float, help="Percentage [0-1] of improvement required for patience to reset. The most lenient "
                                                                                         "value is 0 (any improvement reset patience)")
    parser.add_argument("--learning_rate",              default=1e-4,   type=float, help="Initial value of the learning rate")
    parser.add_argument("--decay_rate",                 default=1,      type=float, help="Decay rate of the exponential decay schedule of the learning rate. 1=no decay")
    parser.add_argument("--decay_steps",                default=100,    type=int)
    parser.add_argument("--staircase",                  action="store_true",        help="Learning schedule is a staircase function if added to arguments")
    parser.add_argument("--clipping",                   action="store_true",            help="Clip backprop gradients between -10 and 10.")
    parser.add_argument("--apodization_alpha",          default=0.4,     type=float, help="Shape parameter of the Tukey window (Tapered cosine Window), "
                                                                                         "representing the fraction of the window inside the cosine tapered region."
                                                                                         "If zero, the Tukey window is equivalent to a rectangular window (no apodization) "
                                                                                         "If one, the Tukey window is equivalent to a Hann window.")
    parser.add_argument("--apodization_factor",         default=1e-2,   type=float, help="Lagrange multiplier of apodization loss")
    parser.add_argument("--tv_factor",                  default=1e-2,   type=float, help="Lagrange multiplier of Total Variation (TV) loss. Penalize high spatial frequency "
                                                                                         "components in the predicted image")
    parser.add_argument("--l2_bottleneck",              default=1,      type=float, help="Initial value of l2 penalty in bottleneck identity "
                                                                                         "map of encoder/decoder latent representation")
    parser.add_argument("--l2_bottleneck_decay_steps",  default=1000,   type=int,   help="Number of steps until l2 bottleneck penalty factor reaches 0")
    parser.add_argument("--l2_bottleneck_decay_power",  default=0.2,    type=float, help="Control the shape of the decay of l2_bottlenck schedule (0.5=square root decay, etc.)")
    parser.add_argument("--skip_strength",              default=0.5,    type=float, help="Initial value of the multiplicative factor in front of the Unet additive skip between "
                                                                                         "encoder and decoder layers.")
    parser.add_argument("--skip_strength_decay_steps",  default=1000,   type=int,   help="Number of steps until skip_strength reaches 0")
    parser.add_argument("--skip_strength_decay_power",  default=0.5,    type=float, help="Control the shape of the decay for skip_strength schedule")


    # model hyperparameters
    parser.add_argument("--layers",                     default=7,      type=int,   help="Number of downsampling block in encoder (symmetric in decoder")
    parser.add_argument("--conv_layers",                default=2,      type=int,   help="Number of conv layers in a Residual block")
    parser.add_argument("--filter_scaling",             default=2,      type=float, help="Filters scale by {filter_scaling}^{res_layer_index}, generally number between (1, 2]")
    parser.add_argument("--filters",                    default=8,      type=int,   help="Number of filters in the first residual block (before last for decoder)")
    parser.add_argument("--kernel_size",                default=3,      type=int,   help="Size of the kernels throughout model")
    parser.add_argument("--kernel_reg_amp",             default=1e-4,   type=float, help="Amplitude of l2 regularization for kernel weights in the model")
    parser.add_argument("--bias_reg_amp",               default=1e-4,   type=float, help="Amplitude of l2 regularization for bias variables in the model")
    parser.add_argument("--latent_size",                default=32,     type=int,   help="Size of the latent vector space")
    parser.add_argument("--activation",                 default="leaky_relu")


    # logs
    parser.add_argument("--logdir",                     default="None",             help="Path of logs directory. Default if None, no logs recorded")
    parser.add_argument("--model_dir",                  default="None",             help="Path to the directory where to save models checkpoints")
    parser.add_argument("--checkpoints",                default=10,     type=int,   help="Save a checkpoint of the models each {%} iteration")
    parser.add_argument("--max_to_keep",                default=3,      type=int,   help="Max model checkpoint to keep")
    parser.add_argument("--logname",                    default="cosmosAE_" + date, help="Name of the logs, default is the local date + time")

    args = parser.parse_args()

    main(args)
