from censai.physical_model import PhysicalModel, AnalyticalPhysicalModel
from censai.cosmos_utils import decode, preprocess
from astropy.visualization import LogStretch, ImageNormalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tensorflow as tf
import numpy as np
import os


def create_model(l2=1e-2, alpha=0.1, dropout_rate=0.5):
    common_params = {"padding": "SAME",
                     "activation": tf.keras.layers.LeakyReLU(alpha=alpha),
                     "kernel_regularizer": tf.keras.regularizers.l2(l2=l2),
                     "bias_regularizer": tf.keras.regularizers.l2(l2=l2),
                     "data_format": "channels_last"
                     }
    _model = tf.keras.Sequential(
        layers=[
            tf.keras.layers.InputLayer(input_shape=[128, 128, 1]),
            tf.keras.layers.Conv2D(filters=8, kernel_size=5, **common_params),
            tf.keras.layers.MaxPool2D(pool_size=(4, 4), padding="SAME", data_format="channels_last"),
            tf.keras.layers.Conv2D(filters=16, kernel_size=3, **common_params),
            tf.keras.layers.MaxPool2D(pool_size=(4, 4), padding="SAME", data_format="channels_last"),
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, **common_params),
            tf.keras.layers.MaxPool2D(pool_size=(4, 4), padding="SAME", data_format="channels_last"),
            tf.keras.layers.Flatten(data_format="channels_last"),
            tf.keras.layers.Dense(64, activation=tf.keras.layers.ReLU(),
                                  kernel_regularizer= tf.keras.regularizers.l2(l2=l2),
                                  bias_regularizer=tf.keras.regularizers.l2(l2=l2)
                                 ),
            tf.keras.layers.Dropout(rate=dropout_rate),
            tf.keras.layers.Dense(16, activation=tf.keras.layers.ReLU(),
                                  kernel_regularizer=tf.keras.regularizers.l2(l2=l2),
                                  bias_regularizer=tf.keras.regularizers.l2(l2=l2)
                                 ),
            tf.keras.layers.Dense(2, activation=tf.keras.activations.sigmoid)
        ]
    )
    return _model


def image_dataset(datapath):
    tf.data.TFRecordDataset(datapath).map(decode).map(preprocess)
