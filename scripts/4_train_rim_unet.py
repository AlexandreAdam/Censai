import tensorflow as tf
import numpy as np
from censai import PhysicalModel, RIMUnet
from censai.data.lenses_tng import decode_train, decode_physical_model_info
from censai.data import NISGenerator
from censai.utils import nullwriter
import os, glob
from datetime import datetime
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    if len(gpus) == 2:
        PHYSICAL_MODEL_DEVICE = tf.device("/device:GPU:1")
    else:
        PHYSICAL_MODEL_DEVICE = tf.device("/device:GPU:0")
try:
    import wandb
    wandb.init(project="censai_unet_rim", entity="adam-alexandre01123", sync_tensorboard=True)
    wndb = True
except ImportError:
    wndb = False
    print("wandb not installed, package ignored")


def main(args):
    if wndb:
        config = wandb.config
        config.update(vars(args))
    if args.dataset == "NIS":
        train_dataset = NISGenerator(int(args.train_split * args.total_items), batch_size=args.batch_size, pixels=args.pixels, model="rim")
        val_dataset = NISGenerator(int((1 - args.train_split) * args.total_items), batch_size=args.batch_size, pixels=args.pixels, model="rim")
        phys = PhysicalModel(pixels=args.pixels, noise_rms=args.noise_rms, method=args.forward_method, device=PHYSICAL_MODEL_DEVICE)
    else:
        files = glob.glob(os.path.join(args.dataset, "*.tfrecords"))
        dataset = tf.data.TFRecordDataset(files, num_parallel_reads=args.num_parallel_reads)
        # Read off global parameters from first example in dataset
        for params in dataset.map(decode_physical_model_info):
            break
        dataset = dataset.map(decode_train).batch(args.batch_size)
        if args.cache_file is not None:
            dataset = dataset.cache(args.cache_file).prefetch(tf.data.experimental.AUTOTUNE)
        else:  # do not cache if no file is provided, dataset is huge and does not fit in GPU or RAM
            dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        train_dataset = dataset.take(int(args.train_split * args.total_items))
        val_dataset = dataset.skip(int(args.train_split * args.total_items))
        if args.raytracer_hparams is not None:
            import json
            with open(args.raytracer_hparams, "r") as f:
                raytracer_hparams = json.load(f)
        else:
            raytracer_hparams = {}
        phys = PhysicalModel(
            pixels=params["kappa pixels"].numpy(),
            src_pixels=params["src pixels"].numpy(),
            image_fov=params["image fov"].numpy(),
            kappa_fov=params["kappa fov"].numpy(),
            method=args.forward_method,
            noise_rms=params["noise rms"].numpy(),
            logkappa=args.logkappa,
            checkpoint_path=args.raytracer,
            device=PHYSICAL_MODEL_DEVICE,
            **raytracer_hparams
        )
    rim = RIMUnet(phys, args.batch_size, args.time_steps, args.pixels, adam=args.adam,
                  kappalog=args.kappalog, normalize=args.normalize,
                  **{"source": {"strides": args.source_strides},
                     "kappa": {"strides": args.kappa_strides}}
                  )
    learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=args.initial_learning_rate,
        decay_rate=args.decay_rate,
        decay_steps=args.decay_steps,
        staircase=args.staircase
    )
    optim = tf.optimizers.Adam(learning_rate=learning_rate_schedule)
    if args.model_id.lower() != "none":
        logname = args.model_id
    elif args.logname is not None:
        logname = args.logname
    else:
        logname = "UnetRIM_" + datetime.now().strftime("%y-%m-%d_%H-%M-%S")
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
            import json
            with open(os.path.join(models_dir, "script_params.json"), "w") as f:
                json.dump(vars(args), f)
        source_checkpoints_dir = os.path.join(models_dir, "source_checkpoints")
        if not os.path.isdir(source_checkpoints_dir):
            os.mkdir(source_checkpoints_dir)
        kappa_checkpoints_dir = os.path.join(models_dir, "kappa_checkpoints")
        if not os.path.isdir(kappa_checkpoints_dir):
            os.mkdir(kappa_checkpoints_dir)
        source_ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optim, net=rim.source_model)
        source_checkpoint_manager = tf.train.CheckpointManager(source_ckpt, source_checkpoints_dir, max_to_keep=args.max_to_keep)
        kappa_ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optim, net=rim.kappa_model)
        kappa_checkpoint_manager = tf.train.CheckpointManager(kappa_ckpt, kappa_checkpoints_dir, max_to_keep=args.max_to_keep)
        save_checkpoint = True
        if args.model_id.lower() != "none":
            if args.load_checkpoint == "lastest":
                kappa_checkpoint_manager.checkpoint.restore(kappa_checkpoint_manager.latest_checkpoint)
                source_checkpoint_manager.checkpoint.restore(source_checkpoint_manager.latest_checkpoint)
            elif args.load_checkpoint == "best":
                kappa_scores = np.loadtxt(os.path.join(kappa_checkpoints_dir, "score_sheet.txt"))
                source_scores = np.loadtxt(os.path.join(source_checkpoints_dir, "score_sheet.txt"))
                _kappa_checkpoint = kappa_scores[np.argmin(kappa_scores[:, 1]), 0]
                _source_checkpoint = source_scores[np.argmin(source_scores[:, 1]), 0]
                kappa_checkpoint = kappa_checkpoint_manager.checkpoints[_kappa_checkpoint]
                kappa_checkpoint_manager.checkpoint.restore(kappa_checkpoint)
                source_checkpoint = kappa_checkpoint_manager.checkpoints[_source_checkpoint]
                source_checkpoint_manager.checkpoint.restore(source_checkpoint)
            else:
                kappa_checkpoint = kappa_checkpoint_manager.checkpoints[int(args.load_checkpoint)]
                source_checkpoint = source_checkpoint_manager.checkpoints[int(args.load_checkpoint)]
                kappa_checkpoint_manager.checkpoint.restore(kappa_checkpoint)
                source_checkpoint_manager.checkpoint.restore(source_checkpoint)
    else:
        save_checkpoint = False

    epoch_loss = tf.metrics.Mean()
    val_loss = tf.metrics.Mean()
    best_loss = np.inf
    patience = args.patience
    step = 0
    lastest_checkpoint = 1
    for epoch in range(args.epochs):
        epoch_loss.reset_states()
        with train_writer.as_default():
            for batch, (X, source, kappa) in enumerate(train_dataset):
                with tf.GradientTape(persistent=True, watch_accessed_variables=True) as tape:
                    tape.watch(rim.source_model.trainable_variables)
                    tape.watch(rim.kappa_model.trainable_variables)
                    cost = rim.cost_function(X, source, kappa)
                gradient1 = tape.gradient(cost, rim.source_model.trainable_variables)
                gradient2 = tape.gradient(cost, rim.kappa_model.trainable_variables)
                if args.clipping:
                    gradient1 = [tf.clip_by_value(grad, -10, 10) for grad in gradient1]
                    gradient2 = [tf.clip_by_value(grad, -10, 10) for grad in gradient2]
                optim.apply_gradients(zip(gradient1, rim.source_model.trainable_variables)) # backprop
                optim.apply_gradients(zip(gradient2, rim.kappa_model.trainable_variables))

                #========== Summary and logs ==========
                epoch_loss.update_state([cost])
                tf.summary.scalar("MSE", cost, step=step)
                step += 1
            tf.summary.scalar("Learning Rate", optim.lr(step), step=step)
        with test_writer.as_default():
            val_loss.reset_states()
            for X, source, kappa in val_dataset:
                test_cost = rim.cost_function(X, source,  kappa)
                val_loss.update_state([test_cost])
            tf.summary.scalar("MSE", test_cost, step=step)
        val_cost = val_loss.result().numpy()
        print(f"epoch {epoch} | train loss {epoch_loss.result().numpy():.3e} | val loss {val_cost:.3e} "
              f"| learning rate {optim.lr(step).numpy():.2e}")
        if val_cost < (1 - args.tolerance) * best_loss:
            best_loss = val_cost
            patience = args.patience
        else:
            patience -= 1
        if save_checkpoint:
            source_checkpoint_manager.checkpoint.step.assign_add(1) # a bit of a hack
            kappa_checkpoint_manager.checkpoint.step.assign_add(1)
            if epoch % args.checkpoints == 0 or patience == 0 or epoch == args.epochs - 1:
                with open(os.path.join(kappa_checkpoints_dir, "score_sheet.txt"), mode="a") as f:
                    np.savetxt(f, np.array([lastest_checkpoint, val_cost]))
                with open(os.path.join(source_checkpoints_dir, "score_sheet.txt"), mode="a") as f:
                    np.savetxt(f, np.array([lastest_checkpoint, val_cost]))
                lastest_checkpoint += 1
                source_checkpoint_manager.save()
                kappa_checkpoint_manager.save()
                print("Saved checkpoint for step {}: {}".format(int(source_checkpoint_manager.checkpoint.step),
                                                                source_checkpoint_manager.latest_checkpoint))
        if patience == 0:
            print("Reached patience")
            break


if __name__ == "__main__":
    from argparse import ArgumentParser
    import json
    parser = ArgumentParser()
    parser.add_argument("--model_id", type=str, default="None",
                        help="Start from this model id checkpoint. None means start from scratch")
    parser.add_argument("--load_checkpoint", default="best", help="One of 'best', 'lastest' or the specific checkpoint index.")

    # RIM hyperparameters
    parser.add_argument("--time_steps",         default=16,     type=int, help="Number of time steps of RIM")
    parser.add_argument("--adam",               default=True,   type=bool,
                        help="ADAM update for the log-likelihood gradient.")
    # ... for kappa model
    parser.add_argument("--logkappa",           default=True,   type=bool)
    parser.add_argument("--normalize",          default=False,  type=bool)
    parser.add_argument("--kappa_strides",      default=4,      type=int,
                        help="Value of the stride parameter in the 3 downsampling and upsampling layers "
                             "for the kappa model.")
    # ... for the source model
    parser.add_argument("--source_strides",     default=2,      type=int,
                        help="Value of the stride parameter in the 3 downsampling and upsampling layers "
                             "for the source model.")
    # ... for the physical model
    parser.add_argument("--forward_method",     default="conv2d",
                        help="One of ['conv2d', 'fft', 'unet']. If the option 'unet' is chosen, the parameter "
                             "'--raytracer' must be provided and point to model checkpoint directory.")
    parser.add_argument("--raytracer",          default=None,
                        help="Path to raytracer checkpoint dir if method 'unet' is used.")
    parser.add_argument("--raytracer_hparams",  default=None,
                        help="Path to raytracer json that describe hyper parameters")

    # Training set params
    parser.add_argument("-b", "--batch_size",   default=10,     type=int,   help="Number of images in a batch.")
    parser.add_argument("--dataset",            default="NIS",
                        help="Dataset to use, either path to directory that contains alpha labels tfrecords "
                             "or the name of the dataset tu use. Options are ['NIS'].")
    parser.add_argument("--train_split",        default=0.8,    type=float, help="Fraction of the training set.")
    parser.add_argument("--total_items",        required=True,  type=int,   help="Total images in an epoch.")
    # ... for NIS dataset
    parser.add_argument("--pixels",             default=512,    type=int,   help="When using NIS, size of the image to generate.")
    parser.add_argument("--noise_rms",          default=1e-3,   type=float, help="Pixel value rms of lensed image.")
    # ... for tfrecord dataset
    parser.add_argument("--num_parallel_reads", default=10,     type=int,
                        help="TFRecord dataset number of parallel reads when loading data.")
    parser.add_argument("--cache_file",         default=None,
                        help="Path to cache file, useful when training on server. Use ${SLURM_TMPDIR}/cache")

    # Optimization params
    parser.add_argument("-e", "--epochs",           default=10,     type=int,   help="Number of epochs for training.")
    parser.add_argument("--initial_learning_rate",  default=1e-3,   type=float, help="Initial learning rate.")
    parser.add_argument("--decay_rate",             default=1.,     type=float,
                        help="Exponential decay rate of learning rate (1=no decay).")
    parser.add_argument("--decay_steps",            default=1000,   type=int,
                        help="Decay steps of exponential decay of the learning rate.")
    parser.add_argument("--staircase",              action="store_true",        help="Learning rate schedule only change after decay steps if enabled.")
    parser.add_argument("--clipping",               default=True,   type=bool, help="Clip backprop gradients between -10 and 10.")
    parser.add_argument("--patience",               default=np.inf, type=int,
                        help="Number of step at which training is stopped if no improvement is recorder.")
    parser.add_argument("--tolerance",              default=0,      type=float,
                        help="Current score <= (1 - tolerance) * best score => reset patience, else reduce patience.")

    # logs
    parser.add_argument("--logdir",                  default="None",
                        help="Path of logs directory. Default if None, no logs recorded.")
    parser.add_argument("--logname",                 default=None,
                        help="Name of the logs, default is 'RT_' + date")
    parser.add_argument("--model_dir",               default="None",
                        help="Path to the directory where to save models checkpoints.")
    parser.add_argument("--checkpoints",             default=10,    type=int,
                        help="Save a checkpoint of the models each {%} iteration.")
    parser.add_argument("--max_to_keep",             default=3,     type=int,
                        help="Max model checkpoint to keep.")

    # Reproducibility params
    parser.add_argument("--seed",                   default=None,   type=int,
                        help="Random seed for numpy and tensorflow.")
    parser.add_argument("--json_override",          default=None,
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
