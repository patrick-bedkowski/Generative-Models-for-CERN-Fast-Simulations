import wandb
import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))

wandb.login(key="d53387a3b34fda2a3caaf861b5fad88cb4ec99ef")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model, decomposition, manifold, preprocessing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, roc_curve, auc, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from tensorflow.keras import layers
import pickle
import time
from numpy import load
from matplotlib import pyplot
import pickle
import argparse

import tensorflow as tf
from tensorflow import keras
print(tf.__version__)

from tensorflow.compat.v1.keras.layers import Input, Dense, LeakyReLU, Conv2D, MaxPooling2D, UpSampling2D,  Concatenate
from tensorflow.compat.v1.keras.models import Model
from tensorflow.compat.v1.keras.layers import Dense, Reshape, Flatten
from tensorflow.compat.v1.keras.layers import Dropout,BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import mse, binary_crossentropy, logcosh
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

from scipy.stats import wasserstein_distance
import pandas as pd
from utils import sum_channels_parallel as sum_channels_parallel
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras import layers
import time

import sklearn
from sklearn.preprocessing import StandardScaler
from datetime import datetime

data = pd.read_pickle('/net/tscratch/people/plgpbedkowski/data/data_photonsum_proton_18_2312.pkl')
print('Loaded: ', data.shape, "max:", data.max())

# Data containing particle conditional data from particle having responses with proton photon sum in interval [70, 2312] without taking into consideration photon sums of neutron responses.
data_cond = pd.read_pickle('/net/tscratch/people/plgpbedkowski/data/data_cond_photonsum_proton_18_2312.pkl')
print('Loaded cond: ', data_cond.shape, "max:", data_cond.values.max(), "min:", data_cond.values.min())

proton_photon_sum = data_cond['proton_photon_sum']
std_proton = data_cond['std']

data_cond.drop(columns=['proton_photon_sum', 'std'], inplace=True)

data = np.log(data+1)
data = np.float32(data)
print("data max", data.max(), "min", data.min())

scaler = StandardScaler()
data_cond = np.float32(data_cond)
data_cond = scaler.fit_transform(data_cond)
print("cond max", data_cond.max(), "min", data_cond.min())

_, x_test, _, y_test, _, std_test, = train_test_split(data, data_cond, std_proton, test_size=0.2, shuffle=False)
print(x_test.shape, y_test.shape, std_test.shape)

from utils import sum_channels_parallel

org = np.exp(x_test)-1
ch_org = np.array(org).reshape(-1, 56, 30)
ch_org = pd.DataFrame(sum_channels_parallel(ch_org)).values

org.shape
del org

vae = tf.keras.models.load_model("/net/tscratch/people/plgpbedkowski/best_models_h5/18_proton/gen_vae_320.h5", compile=False)
gan_4 = tf.keras.models.load_model("/net/tscratch/people/plgpbedkowski/best_models_h5/18_proton/gen_gan_98.h5", compile=False)
gan_5 = tf.keras.models.load_model("/net/tscratch/people/plgpbedkowski/best_models_h5/18_proton/gen_gan_224.h5", compile=False)
sdigan_4 = tf.keras.models.load_model("/net/tscratch/people/plgpbedkowski/best_models_h5/18_proton/gen_sdi-gan_48.h5", compile=False)
sdigan_5 = tf.keras.models.load_model("/net/tscratch/people/plgpbedkowski/best_models_h5/18_proton/gen_sdi-gan_107.h5", compile=False)
sdigan_reg = tf.keras.models.load_model("/net/tscratch/people/plgpbedkowski/best_models_h5/18_proton/gen_sin-gan_113.h5", compile=False)
sdigan_reg_aux = tf.keras.models.load_model("/net/tscratch/people/plgpbedkowski/best_models_h5/18_proton/gen_sin-gan-aux-reg-arch-2_105.h5", compile=False)

sdigan_reg_aux_1 = tf.keras.models.load_model("/net/tscratch/people/plgpbedkowski/best_models_h5/1_proton/gen_sin-gan-aux-reg-arch-2_215.h5", compile=False)
sdigan_reg_aux_5 = tf.keras.models.load_model("/net/tscratch/people/plgpbedkowski/best_models_h5/5_proton/gen_sin-gan-aux-reg-arch-2_81.h5", compile=False)
joint = tf.keras.models.load_model("/net/tscratch/people/plgpbedkowski/best_models_h5/joint/gen_sin-gan-join-aux-reg-arch-2_331.h5", compile=False)


def get_number_of_empty_responses(model, noise_dim = 10, joint=False):
    list_n_empty_responses = []
    list_n_empty_responses_p = []
    list_n_empty_responses_n = []
    batch_size = 200
    n_batches = y_test.shape[0] // batch_size

    for i in range(n_batches):
        start_idx = i*batch_size
        end_idx = start_idx + batch_size
        # for _ in range(n_executions):
        seed = tf.random.normal([batch_size, noise_dim])
        input_data = [seed, y_test[start_idx:end_idx]]

        if joint:
            outputs = model(input_data, training=False)
            outputs_p, outputs_n = outputs[:,:,:,0], outputs[:,:,:,1]
            pixel_sum_p = tf.reduce_sum(outputs_p, axis=(1, 2))
            pixel_sum_n = tf.reduce_sum(outputs_n, axis=(1, 2))

            indices_of_empty_responses_p = tf.where(tf.equal(pixel_sum_p, 0))
            indices_of_empty_responses_n = tf.where(tf.equal(pixel_sum_n, 0))

            n_empty_responses_p = len(indices_of_empty_responses_p.numpy())
            n_empty_responses_n = len(indices_of_empty_responses_n.numpy())

            list_n_empty_responses_p.append(n_empty_responses_p)
            list_n_empty_responses_n.append(n_empty_responses_n)
        else:
            outputs = model(input_data, training=False)
            pixel_sum = tf.reduce_sum(outputs[:,:,:,0], axis=(1, 2))
            indices_of_empty_responses = tf.where(tf.equal(pixel_sum, 0))
            n_empty_responses = len(indices_of_empty_responses.numpy())
            list_n_empty_responses.append(n_empty_responses)

    if joint:
        return sum(list_n_empty_responses_p), sum(list_n_empty_responses_n)
    else:
        return sum(list_n_empty_responses)


def model_diversity_error(generator,
                    y_test, std_test,
                    n_calc=10,
                    noise_std=1):
    """
    Generated diversity loss for model. Performs n_calculations for the same x_test, y_test data.
    """
    n_samples = y_test.shape[0]
    images_data = None
    for j in range(n_calc):
        z = np.random.normal(0, noise_std, (n_samples, 10))
        z_c = y_test
        results = generator.predict([z, z_c])
        results = np.exp(results) - 1

        # 1. flatten responses
        flatten_responses = results.reshape(len(results), -1)  # (n_samples, 1680)
        flatten_responses = flatten_responses.reshape(n_samples, -1, 1)
        if images_data is None:
            images_data = flatten_responses
        else:
            images_data = np.append(images_data, flatten_responses, axis=2)
    stddevs = np.std(images_data, axis=2).sum(axis=1)
    normalized_stddevs = stddevs/max(stddevs)
    return mean_absolute_error(normalized_stddevs, std_test.values)

def calculate_ws_ch(generator,
                    x_test, y_test,
                    ch_org, data_shape,
                    n_calc=5,
                    scale=1, noise_std=1):
    """
    Calculates ws distance for each channel separately.
    """
    ws = [0, 0, 0, 0, 0]
    for j in range(n_calc):
        z = np.random.normal(0, noise_std, (x_test.shape[0], 10))
        z_c = y_test
        results = generator.predict([z, z_c])
        results = np.exp(results) - 1
        results = results * scale

        ch_gen = np.array(results).reshape(data_shape)
        ch_gen = pd.DataFrame(sum_channels_parallel(ch_gen)).values
        for i in range(5):
            ws[i] = ws[i] + wasserstein_distance(ch_org[:, i], ch_gen[:, i])
        ws = np.array(ws)

    ws = ws / n_calc
    return ws.sum() / 5
    # print("\n", "-" * 30, "\n")
    # print("ws mean", f'{ws.sum() / 5:.2f}', end=" ")
    # for n, score in enumerate(ws):
    #     print("ch" + str(n + 1), f'{score:.2f}', end=" ")

print('====== Empty Responses ======')

n_empty_responses_vae = get_number_of_empty_responses(vae)
print(f"VAE: {n_empty_responses_vae}")

n_empty_responses_gan = get_number_of_empty_responses(gan_4)
print(f"GAN 1e-4: {n_empty_responses_gan}")

n_empty_responses_gan = get_number_of_empty_responses(gan_5)
print(f"GAN 1e-5: {n_empty_responses_gan}")

n_empty_responses_sdigan = get_number_of_empty_responses(sdigan_4)
print(f"SDI-GAN 1e-4: {n_empty_responses_sdigan}")

n_empty_responses_sdigan = get_number_of_empty_responses(sdigan_5)
print(f"SDI-GAN 1e-5: {n_empty_responses_sdigan}")

n_empty_responses_sdigan_reg = get_number_of_empty_responses(sdigan_reg)
print(f"SDI-GAN + reg: {n_empty_responses_sdigan_reg}")

n_empty_responses_sdigan_reg_aux = get_number_of_empty_responses(sdigan_reg_aux)
print(f"SDI-GAN + reg + aux: {n_empty_responses_sdigan_reg_aux}")

get_number_of_empty_responses_joint = get_number_of_empty_responses(joint, joint=True)
print(f"JOINT: {get_number_of_empty_responses_joint}")


print('====== Diversity Error ======')

n_empty_responses_vae = model_diversity_error(vae, y_test, std_test, n_calc=10)
print(f"VAE: {n_empty_responses_vae}")

n_empty_responses_gan = model_diversity_error(gan_4, y_test, std_test, n_calc=10)
print(f"GAN 1e-4: {n_empty_responses_gan}")

n_empty_responses_gan = model_diversity_error(gan_5, y_test, std_test, n_calc=10)
print(f"GAN 1e-5: {n_empty_responses_gan}")

n_empty_responses_sdigan = model_diversity_error(sdigan_4, y_test, std_test, n_calc=10)
print(f"SDI-GAN 1e-4: {n_empty_responses_sdigan}")

n_empty_responses_sdigan = model_diversity_error(sdigan_5, y_test, std_test, n_calc=10)
print(f"SDI-GAN 1e-5: {n_empty_responses_sdigan}")

n_empty_responses_sdigan_reg = model_diversity_error(sdigan_reg, y_test, std_test, n_calc=10)
print(f"SDI-GAN + reg: {n_empty_responses_sdigan_reg}")

n_empty_responses_sdigan_reg_aux = model_diversity_error(sdigan_reg_aux, y_test, std_test, n_calc=10)
print(f"SDI-GAN + reg + aux: {n_empty_responses_sdigan_reg_aux}")

print('====== WS ======')

ws_vae = calculate_ws_ch(vae, x_test, y_test, ch_org, (-1, 56, 30))
print(f"VAE: {ws_vae}")

ws_gan = calculate_ws_ch(gan_4, x_test, y_test, ch_org, (-1, 56, 30))
print(f"GAN 1e-4: {ws_gan}")

ws_gan = calculate_ws_ch(gan_5, x_test, y_test, ch_org, (-1, 56, 30))
print(f"GAN 1e-5: {ws_gan}")

ws_sdigan = calculate_ws_ch(sdigan_4, x_test, y_test, ch_org, (-1, 56, 30))
print(f"SDI-GAN 1e-4: {ws_sdigan}")

ws_sdigan = calculate_ws_ch(sdigan_5, x_test, y_test, ch_org, (-1, 56, 30))
print(f"SDI-GAN 1e-5: {ws_sdigan}")

ws_sdigan_reg = calculate_ws_ch(sdigan_reg, x_test, y_test, ch_org, (-1, 56, 30))
print(f"SDI-GAN + reg: {ws_sdigan_reg}")

ws_sdigan_reg_aux = calculate_ws_ch(sdigan_reg_aux, x_test, y_test, ch_org, (-1, 56, 30))
print(f"SDI-GAN + reg + aux: {ws_sdigan_reg_aux}")

ws_sdigan_reg_aux = calculate_ws_ch(sdigan_reg_aux_1, x_test, y_test, ch_org, (-1, 56, 30))
print(f"SDI-GAN 1 + reg + aux: {ws_sdigan_reg_aux}")

ws_sdigan_reg_aux = calculate_ws_ch(sdigan_reg_aux_5, x_test, y_test, ch_org, (-1, 56, 30))
print(f"SDI-GAN 5 + reg + aux: {ws_sdigan_reg_aux}")
