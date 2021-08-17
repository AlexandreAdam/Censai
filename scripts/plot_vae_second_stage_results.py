from censai.models import VAE, VAESecondStage
from censai.utils import sampling_plot
import numpy as np
import tensorflow as tf
import os, glob
import json


def main(args):
    model_list = glob.glob(os.path.join(os.getenv("CENSAI_PATH"), "models", args.model_prefixe +"*"))
    for model in model_list:
        if "second_stage" in model:
            continue
        with open(os.path.join(model, "model_hparams.json"), "r") as f:
            vae_hparams = json.load(f)

        # load weights of first stage
        vae = VAE(**vae_hparams)
        ckpt1 = tf.train.Checkpoint(net=vae)
        checkpoint_manager1 = tf.train.CheckpointManager(ckpt1, model, 1)
        checkpoint_manager1.checkpoint.restore(checkpoint_manager1.latest_checkpoint).expect_partial()

        model_name = os.path.split(model)[-1]
        print(model_name)
        second_stages = [file for file in model_list if "second_stage" in file and model_name in file]
        for second_stage in second_stages:
            second_stage_name = os.path.split(second_stage)[-1]
            print(second_stage_name)
            with open(os.path.join(second_stage, "model_hparams.json"), "r") as f:
                vae2_hparams = json.load(f)
            vae2 = VAESecondStage(**vae2_hparams)
            ckpt2 = tf.train.Checkpoint(net=vae2)
            checkpoint_manager2 = tf.train.CheckpointManager(ckpt2, second_stage, 1)
            checkpoint_manager2.checkpoint.restore(checkpoint_manager2.latest_checkpoint).expect_partial()

            for n in range(args.n_plots):
                z = vae2.sample(args.sampling_size)
                y_pred = vae.decode(z)
                fig = sampling_plot(y_pred)
                fig.suptitle(model_name)
                fig.savefig(os.path.join(os.getenv("CENSAI_PATH"), "results", "vae2_sampling_" + second_stage_name + args.output_postfixe + f"_{n:02d}.png"))


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--model_prefixe",      required=True)
    parser.add_argument("--output_postfixe",    default="",                             help="phrase to append to result filename")
    parser.add_argument("--sampling_size",      default=81,         type=int,           help="Number of images to sample for a single figure")
    parser.add_argument("--n_plots",            default=1,          type=int,           help="Number of plot to make for a given model")
    parser.add_argument("--seed",               default=None,       type=int)

    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    main(args)
