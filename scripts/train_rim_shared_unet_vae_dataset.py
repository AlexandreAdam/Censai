import tensorflow as tf
import numpy as np
from censai import PhysicalModel, RIMSharedUnet
from censai.models import SharedUnetModel, RayTracer, VAE, VAESecondStage
from censai.utils import nullwriter, rim_residual_plot as residual_plot, plot_to_image
import os, time, json
from datetime import datetime


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

VAE_HPARAMS = [
    "pixels",
    "layers",
    "conv_layers",
    "filter_scaling",
    "filters",
    "kernel_size",
    "kernel_reg_amp",
    "bias_reg_amp",
    "activation",
    "dropout_rate",
    "batch_norm",
    "latent_size"
]

VAE2_HPARAMS = [
    "hidden_layers",
    "kernel_reg_amp",
    "bias_reg_amp",
    "activation",
    "latent_size",
    "units",
    "output_size"
]


def main(args):
    if args.raytracer is not None:
        with open(os.path.join(args.raytracer, "ray_tracer_hparams.json"), "r") as f:
            raytracer_hparams = json.load(f)
    if args.raytracer is not None:
        raytracer = RayTracer(**raytracer_hparams)
        # load last checkpoint in the checkpoint directory
        checkpoint = tf.train.Checkpoint(net=raytracer)
        manager = tf.train.CheckpointManager(checkpoint, directory=args.raytracer, max_to_keep=3)
        checkpoint.restore(manager.latest_checkpoint).expect_partial()
    else:
        raytracer = None
    
    # =============== kappa vae ========================================
    # Load first stage and freeze weights
    with open(os.path.join(args.kappa_first_stage_vae, "model_hparams.json"), "r") as f:
        kappa_vae_hparams = json.load(f)
    kappa_vae = VAE(**kappa_vae_hparams)
    ckpt1 = tf.train.Checkpoint(step=tf.Variable(1), net=kappa_vae)
    checkpoint_manager1 = tf.train.CheckpointManager(ckpt1, args.kappa_first_stage_vae, 1)
    checkpoint_manager1.checkpoint.restore(checkpoint_manager1.latest_checkpoint).expect_partial()
    kappa_vae.trainable = False
    kappa_vae.encoder.trainable = False
    kappa_vae.decoder.trainable = False
    
    # Setup sampling from second stage if provided
    if args.kappa_second_stage_vae is not None:
        with open(os.path.join(args.kappa_second_stage_vae, "model_hparams.json"), "r") as f:
            kappa_vae2_hparams = json.load(f)
        kappa_vae2 = VAESecondStage(**kappa_vae2_hparams)
        ckpt1 = tf.train.Checkpoint(step=tf.Variable(1), net=kappa_vae2)
        checkpoint_manager1 = tf.train.CheckpointManager(ckpt1, args.kappa_second_stage_vae, 1)
        checkpoint_manager1.checkpoint.restore(checkpoint_manager1.latest_checkpoint).expect_partial()
        kappa_vae2.trainable = False
        kappa_vae2.encoder.trainable = False
        kappa_vae2.decoder.trainable = False
        kappa_sampling_function = lambda batch_size: 10 ** kappa_vae.decode(kappa_vae2.sample(batch_size))
    else:
        kappa_sampling_function = lambda batch_size: 10 ** kappa_vae.sample(batch_size)

    # =============== source vae ========================================
    # Load first stage and freeze weights
    with open(os.path.join(args.source_first_stage_vae, "model_hparams.json"), "r") as f:
        source_vae_hparams = json.load(f)
    source_vae = VAE(**source_vae_hparams)
    ckpt1 = tf.train.Checkpoint(step=tf.Variable(1), net=source_vae)
    checkpoint_manager1 = tf.train.CheckpointManager(ckpt1, args.source_first_stage_vae, 1)
    checkpoint_manager1.checkpoint.restore(checkpoint_manager1.latest_checkpoint).expect_partial()
    source_vae.trainable = False
    source_vae.encoder.trainable = False
    source_vae.decoder.trainable = False

    # Setup sampling from second stage if provided
    if args.source_second_stage_vae is not None:
        with open(os.path.join(args.source_second_stage_vae, "model_hparams.json"), "r") as f:
            source_vae2_hparams = json.load(f)
        source_vae2 = VAESecondStage(**source_vae2_hparams)
        ckpt1 = tf.train.Checkpoint(step=tf.Variable(1), net=source_vae2)
        checkpoint_manager1 = tf.train.CheckpointManager(ckpt1, args.source_second_stage_vae, 1)
        checkpoint_manager1.checkpoint.restore(checkpoint_manager1.latest_checkpoint).expect_partial()
        source_vae2.trainable = False
        source_vae2.encoder.trainable = False
        source_vae2.decoder.trainable = False
        source_sampling_function = lambda batch_size: source_vae.decode(source_vae2.sample(batch_size))
    else:
        source_sampling_function = lambda batch_size: source_vae.sample(batch_size)

    phys = PhysicalModel(
        pixels=args.image_pixels,
        kappa_pixels=kappa_vae_hparams["pixels"],
        src_pixels=source_vae_hparams["pixels"],
        image_fov=args.image_fov,
        kappa_fov=args.kappa_fov,
        src_fov=args.source_fov,
        method=args.forward_method,
        noise_rms=args.noise_rms,
        raytracer=raytracer,
        psf_sigma=args.psf_sigma
    )

    unet = SharedUnetModel(
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

    def train_step(X, source, kappa):
        with tf.GradientTape() as tape:
            tape.watch(rim.unet.trainable_variables)
            source_series, kappa_series, chi_squared = rim.call(X, outer_tape=tape)
            source_cost = tf.reduce_mean(tf.square(source_series - rim.source_inverse_link(source)), axis=(0, 2, 3, 4))
            kappa_cost = tf.reduce_mean(tf.square(kappa_series - rim.kappa_inverse_link(kappa)), axis=(0, 2, 3, 4))
            cost = tf.reduce_sum(kappa_cost + source_cost) / args.batch_size
        gradient = tape.gradient(cost, rim.unet.trainable_variables)
        if args.clipping:
            gradient = [tf.clip_by_value(grad, -10, 10) for grad in gradient]
        optim.apply_gradients(zip(gradient, rim.unet.trainable_variables))
        chi_squared = tf.reduce_sum(chi_squared) / args.batch_size
        source_cost = tf.reduce_sum(source_cost) / args.batch_size
        kappa_cost = tf.reduce_sum(kappa_cost) / args.batch_size
        return cost, chi_squared, source_cost, kappa_cost

    # ====== Training loop ============================================================================================
    epoch_loss = tf.metrics.Mean()
    time_per_step = tf.metrics.Mean()
    epoch_chi_squared = tf.metrics.Mean()
    epoch_source_cost = tf.metrics.Mean()
    epoch_kappa_cost = tf.metrics.Mean()
    history = {  # recorded at the end of an epoch only
        "cost": [],
        "chi_squared": [],
        "source_cost": [],
        "kappa_cost": [],
        "learning_rate": [],
        "time_per_step": [],
        "step": []
    }
    best_loss = np.inf
    patience = args.patience
    step = 0
    global_start = time.time()
    estimated_time_for_epoch = 0
    out_of_time = False
    lastest_checkpoint = 1
    for epoch in range(args.epochs):
        if (time.time() - global_start) > args.max_time*3600 - estimated_time_for_epoch:
            break
        epoch_start = time.time()
        epoch_loss.reset_states()
        epoch_chi_squared.reset_states()
        epoch_source_cost.reset_states()
        epoch_kappa_cost.reset_states()
        time_per_step.reset_states()
        with writer.as_default():
            for batch in range(args.total_items // args.batch_size):
                start = time.time()
                kappa = kappa_sampling_function(args.batch_size)
                source = source_sampling_function(args.batch_size)
                source /= tf.reduce_max(source, axis=(1, 2, 3), keepdims=True)  # normalize source
                X = tf.nn.relu(phys.noisy_forward(source, kappa, noise_rms=args.noise_rms))
                X /= tf.reduce_max(X, axis=(1, 2, 3), keepdims=True)  # normalize lens
                cost, chi_squared, source_cost, kappa_cost = train_step(X, source, kappa)

        # ========== Summary and logs ==================================================================================
                _time = time.time() - start
                tf.summary.scalar("Time per step", _time, step=step)
                tf.summary.scalar("MSE", cost, step=step)
                tf.summary.scalar("Chi Squared", chi_squared, step=step)
                tf.summary.scalar("Source Cost", source_cost, step=step)
                tf.summary.scalar("Kappa Cost", kappa_cost, step=step)
                time_per_step.update_state([_time])
                epoch_loss.update_state([cost])
                epoch_chi_squared.update_state([chi_squared])
                epoch_source_cost.update_state([source_cost])
                epoch_kappa_cost.update_state([kappa_cost])
                step += 1
            if args.n_residuals > 0:
                kappa_true = kappa_sampling_function(args.n_residuals)
                source_true = source_sampling_function(args.n_residuals)
                lens_true = tf.nn.relu(phys.noisy_forward(source_true, kappa_true, noise_rms=args.noise_rms))
                source_pred, kappa_pred, chi_squared = rim.predict(lens_true)
                lens_pred = phys.forward(source_pred[-1], kappa_pred[-1])
            for res_idx in range(args.n_residuals):
                try:
                    tf.summary.image(f"Residuals {res_idx}",
                                     plot_to_image(
                                         residual_plot(
                                             lens_true[res_idx],
                                             source_true[res_idx, ...],
                                             kappa_true[res_idx, ...],
                                             lens_pred[res_idx],
                                             source_pred[-1][res_idx, ...],
                                             kappa_pred[-1][res_idx, ...],
                                             chi_squared[-1][res_idx]
                                         )), step=step)
                except ValueError:
                    continue

            train_cost = epoch_loss.result().numpy()
            train_chi_sq = epoch_chi_squared.result().numpy()
            train_s_cost = epoch_source_cost.result().numpy()
            train_k_cost = epoch_kappa_cost.result().numpy()
            tf.summary.scalar("Learning Rate", optim.lr(step), step=step)
        print(f"step {step} | train loss {train_cost:.3e} | chi sq {train_chi_sq:.3e}"
              f"| learning rate {optim.lr(step).numpy():.2e} | time per step {time_per_step.result().numpy():.2e} s")
        history["cost"].append(train_cost)
        history["learning_rate"].append(optim.lr(step).numpy())
        history["chi_squared"].append(train_chi_sq)
        history["time_per_step"].append(time_per_step.result().numpy())
        history["kappa_cost"].append(train_k_cost)
        history["source_cost"].append(train_s_cost)
        history["step"].append(step)

        cost = train_cost
        if np.isnan(cost):
            print("Training broke the Universe")
            break
        if cost < (1 - args.tolerance) * best_loss:
            best_loss = cost
            patience = args.patience
        else:
            patience -= 1
        if (time.time() - global_start) > args.max_time * 3600:
            out_of_time = True
        if save_checkpoint:
            checkpoint_manager.checkpoint.step.assign_add(1) # a bit of a hack
            if epoch % args.checkpoints == 0 or patience == 0 or epoch == args.epochs - 1 or out_of_time:
                with open(os.path.join(checkpoints_dir, "score_sheet.txt"), mode="a") as f:
                    np.savetxt(f, np.array([[lastest_checkpoint, cost]]))
                lastest_checkpoint += 1
                checkpoint_manager.save()
                print("Saved checkpoint for step {}: {}".format(int(checkpoint_manager.checkpoint.step), checkpoint_manager.latest_checkpoint))
        if patience == 0:
            print("Reached patience")
            break
        if out_of_time:
            break
        if epoch > 0:  # First epoch is always very slow and not a good estimate of an epoch time.
            estimated_time_for_epoch = time.time() - epoch_start
    print(f"Finished training after {(time.time() - global_start)/3600:.3f} hours.")
    return history, best_loss


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--model_id",                   default="None",                 help="Start from this model id checkpoint. None means start from scratch")
    parser.add_argument("--load_checkpoint",            default="best",                 help="One of 'best', 'lastest' or the specific checkpoint index.")
    parser.add_argument("--kappa_first_stage_vae",      required=True)
    parser.add_argument("--kappa_second_stage_vae",     default=None)
    parser.add_argument("--source_first_stage_vae",     required=True)
    parser.add_argument("--source_second_stage_vae",    default=None)

    # RIM hyperparameters
    parser.add_argument("--steps",              default=16,     type=int,       help="Number of time steps of RIM")
    parser.add_argument("--adam",               action="store_true",            help="ADAM update for the log-likelihood gradient.")
    parser.add_argument("--kappalog",           action="store_true")
    parser.add_argument("--kappa_normalize",    action="store_true")
    parser.add_argument("--source_link",        default="identity",             help="One of 'exp', 'source', 'relu' or 'identity' (default).")
    parser.add_argument("--kappa_init",         default=1e-1,   type=float,     help="Initial value of kappa for RIM")
    parser.add_argument("--source_init",        default=1e-3,   type=float,     help="Initial value of source for RIM")

    # Shared Unet params
    parser.add_argument("--filters",                                    default=32,     type=int)
    parser.add_argument("--filter_scaling",                             default=1,      type=float)
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

    # Physical model hyperparameter
    parser.add_argument("--forward_method",         default="conv2d",               help="One of ['conv2d', 'fft', 'unet']. If the option 'unet' is chosen, the parameter "
                                                                                         "'--raytracer' must be provided and point to model checkpoint directory.")
    parser.add_argument("--raytracer",              default=None,                   help="Path to raytracer checkpoint dir if method 'unet' is used.")
    parser.add_argument("--image_pixels",           default=512,    type=int,       help="Number of pixels on a side of the lensed image")
    parser.add_argument("--image_fov",              default=20,     type=float,     help="Field of view of lensed image in arcsec")
    parser.add_argument("--kappa_fov",              default=18,     type=float,     help="Field of view of kappa map (in lens plane), in arcsec")
    parser.add_argument("--source_fov",             default=3,      type=float,     help="Field of view of source map, in arcsec")
    parser.add_argument("--noise_rms",              default=1e-2,   type=float,     help="RMS of white noise added to lensed image")
    parser.add_argument("--psf_sigma",              default=0.08,   type=float,     help="Size, in arcseconds, of the gaussian blurring PSF")

    # Training set params
    parser.add_argument("-b", "--batch_size",       default=1,      type=int,       help="Number of images in a batch. ")
    parser.add_argument("--total_items",            required=True,  type=int,       help="Total images in an epoch.")

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
    parser.add_argument("--max_time",               default=np.inf, type=float,     help="Time allowed for the training, in hours.")

    # logs
    parser.add_argument("--logdir",                  default="None",                help="Path of logs directory. Default if None, no logs recorded.")
    parser.add_argument("--logname",                 default=None,                  help="Overwrite name of the log with this argument")
    parser.add_argument("--logname_prefixe",         default="RIMSUwVAE",           help="If name of the log is not provided, this prefix is prepended to the date")
    parser.add_argument("--model_dir",               default="None",                help="Path to the directory where to save models checkpoints.")
    parser.add_argument("--checkpoints",             default=10,    type=int,       help="Save a checkpoint of the models each {%} iteration.")
    parser.add_argument("--max_to_keep",             default=3,     type=int,       help="Max model checkpoint to keep.")
    parser.add_argument("--n_residuals",             default=1,     type=int,       help="Number of residual plots to save. Add overhead at the end of an epoch only.")

    # Reproducibility params
    parser.add_argument("--seed",                   default=None,   type=int,       help="Random seed for numpy and tensorflow.")
    parser.add_argument("--json_override",          default=None, nargs="+",        help="A json filepath that will override every command line parameters. Useful for reproducibility")

    args = parser.parse_args()
    if args.seed is not None:
        tf.random.set_seed(args.seed)
        np.random.seed(args.seed)
    if args.json_override is not None:
        if isinstance(args.json_override, list):
            files = args.json_override
        else:
            files = [args.json_override,]
        for file in files:
            with open(file, "r") as f:
                json_override = json.load(f)
            args_dict = vars(args)
            args_dict.update(json_override)

    main(args)
