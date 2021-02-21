import tensorflow as tf
from censai.definitions import RayTracer
from censai.data_generator import NISGenerator
try:
    import wandb
    wandb.init(project="censai", sync_tensorboard=True)
except ImportError:
    print("wandb not installed, package ignored")

# if logdir is not None:
    # if not os.path.isdir(os.path.join(logdir, "train")):
        # os.mkdir(os.path.join(logdir, "train"))
    # if not os.path.isdir(os.path.join(logdir, "test")):
        # os.mkdir(os.path.join(logdir, "test"))
    # train_writer = tf.summary.create_file_writer(os.path.join(logdir, "train"))
    # test_writer = tf.summary.create_file_writer(os.path.join(logdir, "test"))
# else:
    # test_writer = nullwriter()
    # train_writer = nullwriter()

