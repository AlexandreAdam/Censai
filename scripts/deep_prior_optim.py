import numpy as np
import matplotlib.pyplot as plt
import os, glob, re, json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm, SymLogNorm, CenteredNorm
from censai.data.lenses_tng import decode_train, decode_physical_model_info
from censai import PhysicalModel
from argparse import Namespace
from argparse import Namespace
import math, json
import matplotlib.pylab as pylab
import tensorflow as tf
import h5py
from tqdm import tqdm

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 2000)
pd.set_option('display.max_colwidth', 200)

result_dir = os.path.join(os.getenv("CENSAI_PATH"), "results")
data_path = os.path.join(os.getenv("CENSAI_PATH"), "data")
models_path = os.path.join(os.getenv("CENSAI_PATH"), "models")

params = {'legend.fontsize': 'x-large',
#           'figure.figsize': (10, 10),
         'axes.labelsize': 'x-large',
         'axes.titlesize': 30,#'x-large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)
# plt.style.use("science")



# model = "RIMSU512_k128_NIE2nsvdO_033_TS10_F16_L5_IK11_NLrelu_al0.04_GAplus_42_B10_lr0.0005_dr0.8_ds5000_TWquadratic_210923032150"
model = "RIMSU128_hTNG2nsvdO_Sinit1_001_F16_IK11_NLleaky_relu_128_211012121338"
path = os.path.join(os.getenv("CENSAI_PATH"), "data", "rim_predictions", model, "prediction.h5")
hf = h5py.File(path, 'r')
k = 8

lens = np.array(hf[f"data{k:03d}/lens"]).squeeze()
source = np.array(hf[f"data{k:03d}/source"]).squeeze()
kappa = np.array(hf[f"data{k:03d}/kappa"]).squeeze()
lens_pred = np.array(hf[f"data{k:03d}/lens_pred"]).squeeze()
source_pred = np.array(hf[f"data{k:03d}/source_pred"]).squeeze()[-1]
kappa_pred = np.array(hf[f"data{k:03d}/kappa_pred"]).squeeze()[-1]
chi_squared = np.array(hf[f"data{k:03d}/chi_squared"]).squeeze()[-1]

hf.close()

DTYPE = tf.float32
STEPS = 100
SAVE = 10

phys = PhysicalModel(
    pixels=128,
    image_fov=17.4,
    kappa_fov=17.4,
    src_fov=10,
    method="fft",
    noise_rms=0.01,
    psf_sigma=0.2
)



optim = tf.keras.optimizers.Adam(lr=1e-4)
source_o = tf.Variable(tf.identity(source_pred)[None, ..., None], DTYPE)
kappa_o = tf.Variable((tf.math.log(tf.identity(kappa_pred)[None, ..., None], DTYPE) + 1e-6) / tf.math.log(10.))
lensed_image = lens



source_series = tf.TensorArray(DTYPE, size=STEPS // SAVE + 1)
kappa_series = tf.TensorArray(DTYPE, size=STEPS // SAVE + 1)
chi_squared_series = tf.TensorArray(DTYPE, size=STEPS // SAVE + 1)
lens_series = tf.TensorArray(DTYPE, size=STEPS // SAVE + 1)
for current_step in tqdm(range(STEPS)):
    with tf.GradientTape() as g:
        g.watch(source_o)
        g.watch(kappa_o)
        y_pred = phys.forward(source=source_o, kappa=10 ** kappa_o)
        #         lam = tf.reduce_sum(y_pred * lensed_image[None, ..., None]) / tf.reduce_sum(lensed_image**2)
        log_likelihood = 0.5 * tf.reduce_sum(tf.square(lensed_image - y_pred) / phys.noise_rms ** 2)
    source_grad, kappa_grad = g.gradient(log_likelihood, [source_o, kappa_o])
    optim.apply_gradients(zip([source_grad, kappa_grad], [source_o, kappa_o]))

    #     with tf.GradientTape() as g:
    #         g.watch(kappa_o)
    #         y_pred = phys.forward(source=source_o, kappa=kappa_o)
    #         log_likelihood = 0.5 * tf.reduce_mean(tf.square(lensed_image - y_pred)/phys.noise_rms**2, axis=(1, 2, 3))
    #     kappa_grad = g.gradient(cost, kappa_o)
    #     optim.apply_gradients(zip([kappa_grad], [kappa_o]))

    if current_step % SAVE == 0:
        source_series = source_series.write(index=current_step // SAVE, value=source_o)
        kappa_series = kappa_series.write(index=current_step // SAVE, value=kappa_o)
        chi_squared_series = chi_squared_series.write(index=current_step // SAVE, value=log_likelihood)
        lens_series = lens_series.write(index=current_step // SAVE, value=y_pred)

source_series = source_series.write(index=STEPS // SAVE, value=source_o)
kappa_series = kappa_series.write(index=STEPS // SAVE, value=kappa_o)
chi_squared_series = chi_squared_series.write(index=STEPS // SAVE, value=log_likelihood)
lens_series = lens_series.write(index=STEPS // SAVE, value=y_pred)
s, k, c, y = source_series.stack(), kappa_series.stack(), chi_squared_series.stack(), lens_series.stack()