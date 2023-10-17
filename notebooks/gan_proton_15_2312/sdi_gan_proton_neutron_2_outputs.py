import wandb

import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))

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
from datetime import datetime

SAVE_EXPERIMENT_DATA = True

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time

from IPython import display
import sklearn
from sklearn.preprocessing import StandardScaler
from datetime import datetime

data_proton = pd.read_pickle('data_proton_photonsum_p_15_2133_n_15_3273.pkl')
print('Loaded: ',  data_proton.shape, "max:", data_proton.max())

data_neutron = pd.read_pickle('data_neutron_photonsum_p_15_2133_n_15_3273.pkl')
print('Loaded: ',  data_neutron.shape, "max:", data_neutron.max())

data_cond = pd.read_pickle('data_cond_stddev_photonsum_p_15_2133_n_15_3273.pkl')
print('Loaded cond: ',  data_cond.shape, "max:", data_cond.values.max(), "min:", data_cond.values.min())

# calculate min max proton, neutron sum
photon_sum_proton_min, photon_sum_proton_max = data_cond.proton_photon_sum.min(), data_cond.proton_photon_sum.max()
photon_sum_neutron_min, photon_sum_neutron_max = data_cond.neutron_photon_sum.min(), data_cond.neutron_photon_sum.max()

data_cond.drop(columns=['proton_photon_sum', 'neutron_photon_sum', 'group_number_proton', 'group_number_neutron'], inplace=True)

STRENGTH = 0.1

DATE_STR = datetime.now().strftime("%d_%m_%Y_%H_%M")

NAME = "sdi-gan-padded-2-outputs"

wandb_run_name = f"{int(photon_sum_proton_min)}_{int(photon_sum_proton_max)}_{DATE_STR}"

EXPERIMENT_DIR_NAME = f"experiments/{NAME}_{int(photon_sum_proton_min)}_{int(photon_sum_proton_max)}_{DATE_STR}"

print("Experiment DIR: ", EXPERIMENT_DIR_NAME)

def create_dir(path):
    if SAVE_EXPERIMENT_DATA:
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
            
def save_scales(model_name, scaler_means, scaler_scales):
    out_fnm = f"{model_name}_scales.txt"
    res = "#means"
    for mean_ in scaler_means:
        res += "\n" + str(mean_)
    res += "\n\n#scales"
    for scale_ in scaler_scales:
        res += "\n" + str(scale_)

    if SAVE_EXPERIMENT_DATA:
        filepath = f"../../{EXPERIMENT_DIR_NAME}/scales/"
        create_dir(filepath)
        with open(filepath+out_fnm, mode="w") as f:
            f.write(res)
            
data_cond["cond"] = data_cond["Energy"].astype(str) +"|"+ data_cond["Vx"].astype(str) +"|"+  data_cond["Vy"].astype(str) +"|"+ data_cond["Vz"].astype(str) +"|"+  data_cond["Px"].astype(str) +"|"+  data_cond["Py"].astype(str) +"|"+ data_cond["Pz"].astype(str) +"|"+  data_cond["mass"].astype(str) +"|"+  data_cond["charge"].astype(str)

data_cond_id = data_cond[["cond"]].reset_index()
data_cond_id

# select a random index of the same conditional data
# shuffle the data and merge it according to the conditional data. Pick the first index of the grouped conditional data
# if some unique conditional data has only one index of sample, then pair it with the same index 
ids = data_cond_id.merge(data_cond_id.sample(frac=1), on=["cond"], how="inner").groupby("index_x").first()
ids = ids["index_y"]
ids

from sklearn.preprocessing import MinMaxScaler

data_proton = np.log(data_proton+1)
data_proton = np.float32(data_proton)
print("data max", data_proton.max(), "min", data_proton.min())

data_proton_2 = data_proton[ids]

data_neutron = np.log(data_neutron+1)
data_neutron = np.float32(data_neutron)
print("data max", data_neutron.max(), "min", data_neutron.min())

data_neutron_2 = data_neutron[ids]

data_cond = data_cond.drop(columns="cond")

scaler_proton = MinMaxScaler()
std_proton = data_cond["std_proton"].values.reshape(-1,1)
std_proton = np.float32(std_proton)
std_proton = scaler_proton.fit_transform(std_proton)
print("std max", std_proton.max(), "min", std_proton.min())

scaler_neutron = MinMaxScaler()
std_neutron = data_cond["std_neutron"].values.reshape(-1,1)
std_neutron = np.float32(std_neutron)
std_neutron = scaler_neutron.fit_transform(std_neutron)
print("std max", std_neutron.max(), "min", std_neutron.min())

scaler = StandardScaler()
data_cond = np.float32(data_cond.drop(columns=["std_proton", "std_neutron"]))
data_cond = scaler.fit_transform(data_cond)
print("cond max", data_cond.max(), "min", data_cond.min())

x_train_p, x_test_p, x_train_p_2, x_test_p_2, x_train_n, x_test_n, x_train_n_2, x_test_n_2, y_train, y_test, std_proton_train, std_proton_test, std_neutron_train, std_neutron_test = train_test_split(
data_proton, data_proton_2, data_neutron, data_neutron_2, data_cond, std_proton, std_neutron, test_size=0.2, shuffle=False)
print(x_train_p.shape, x_test_p.shape, x_train_p_2.shape, x_test_p_2.shape, x_train_n.shape, x_test_n.shape, x_train_n_2.shape, x_test_n_2.shape)
print(y_train.shape, y_test.shape, std_proton_train.shape, std_proton_test.shape, std_neutron_train.shape, std_neutron_test.shape)

#save scales
if SAVE_EXPERIMENT_DATA:
    save_scales("Proton", scaler.mean_, scaler.scale_)
    
BATCH_SIZE = 128

# Training dataset

# datasets that in each index contain two samples from the same conditional data
dataset_p = tf.data.Dataset.from_tensor_slices(x_train_p).batch(batch_size=BATCH_SIZE)
dataset_p_2 = tf.data.Dataset.from_tensor_slices(x_train_p_2).batch(batch_size=BATCH_SIZE)

dataset_n = tf.data.Dataset.from_tensor_slices(x_train_n).batch(batch_size=BATCH_SIZE)
dataset_n_2 = tf.data.Dataset.from_tensor_slices(x_train_n_2).batch(batch_size=BATCH_SIZE)

# conditional data
dataset_cond = tf.data.Dataset.from_tensor_slices(y_train).batch(batch_size=BATCH_SIZE)

# standard deviation for each conditional data samples
dataset_std_proton = tf.data.Dataset.from_tensor_slices(std_proton_train).batch(batch_size=BATCH_SIZE)
dataset_std_neutron = tf.data.Dataset.from_tensor_slices(std_neutron_train).batch(batch_size=BATCH_SIZE)

# shuffled conditional data
fake_cond = tf.data.Dataset.from_tensor_slices(y_train).shuffle(12800).batch(batch_size=BATCH_SIZE)

# zipped data
dataset_with_cond = tf.data.Dataset.zip((dataset_p, dataset_p_2,
                                         dataset_n, dataset_n_2,
                                         dataset_cond, dataset_std_proton, dataset_std_neutron, fake_cond)).shuffle(12800)

# Validation dataset

val_dataset_p = tf.data.Dataset.from_tensor_slices(x_test_p).batch(batch_size=BATCH_SIZE)
val_dataset_p_2 = tf.data.Dataset.from_tensor_slices(x_test_p_2).batch(batch_size=BATCH_SIZE)

val_dataset_n = tf.data.Dataset.from_tensor_slices(x_test_n).batch(batch_size=BATCH_SIZE)
val_dataset_n_2 = tf.data.Dataset.from_tensor_slices(x_test_n_2).batch(batch_size=BATCH_SIZE)

val_dataset_cond = tf.data.Dataset.from_tensor_slices(y_test).batch(batch_size=BATCH_SIZE)
val_dataset_std_proton = tf.data.Dataset.from_tensor_slices(std_proton_test).batch(batch_size=BATCH_SIZE)
val_dataset_std_neutron = tf.data.Dataset.from_tensor_slices(std_neutron_test).batch(batch_size=BATCH_SIZE)
val_fake_cond =  tf.data.Dataset.from_tensor_slices(y_test).shuffle(12800).batch(batch_size=BATCH_SIZE)

val_dataset_with_cond = tf.data.Dataset.zip((val_dataset_p, val_dataset_p_2,
                                             val_dataset_n, val_dataset_n_2,
                                             val_dataset_cond, val_dataset_std_proton, val_dataset_std_neutron, val_fake_cond)).shuffle(12800)

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

from tensorflow import keras

# OLD Architecture

latent_dim = 8
cond_dim = 9

############################ generator ############################

x = Input(shape=(latent_dim,))
cond = Input(shape=(cond_dim,))
inputs = Concatenate(axis=1)([x, cond])


# PROTON HEAD
layer_1 = Dense(128*2)(inputs)
layer_1_bd = Dropout(0.2)(BatchNormalization()(layer_1))
layer_1_a = LeakyReLU(alpha=0.1)(layer_1_bd)

layer_2 = Dense(128*20*12)(layer_1_a)
layer_2_bd = Dropout(0.2)(BatchNormalization()(layer_2))
layer_2_a = LeakyReLU(alpha=0.1)(layer_2_bd)

reshaped = Reshape((20,12,128))(layer_2_a)
reshaped_s = UpSampling2D(size=(3,2))(reshaped)

# PROTON HEAD
conv1 = Conv2D(256, kernel_size=(2, 6), padding='valid')(reshaped_s)
conv1_bd = Dropout(0.2)(BatchNormalization()(conv1))
conv1_a = LeakyReLU(alpha=0.1)(conv1_bd)
conv1_a_s = UpSampling2D(size=(1,2))(conv1_a)

conv2 = Conv2D(128, kernel_size=(2, 6))(conv1_a_s)
conv2_bd = Dropout(0.2)(BatchNormalization()(conv2))
conv2_a = LeakyReLU(alpha=0.1)(conv2_bd)

conv3 = Conv2D(64, kernel_size=(2, 4))(conv2_a)
conv3_bd = Dropout(0.2)(BatchNormalization()(conv3))
conv3_a = LeakyReLU(alpha=0.1)(conv3_bd)

# NEUTRON HEAD
conv4 = Conv2D(256, kernel_size=(6, 2), padding='valid')(reshaped_s)
conv4_bd = Dropout(0.2)(BatchNormalization()(conv4))
conv4_a = LeakyReLU(alpha=0.1)(conv4_bd)
conv4_a_s = UpSampling2D(size=(1,2))(conv4_a)

conv5 = Conv2D(128, kernel_size=(6, 2))(conv4_a_s)
conv5_bd = Dropout(0.2)(BatchNormalization()(conv5))
conv5_a = LeakyReLU(alpha=0.1)(conv5_bd)

conv6 = Conv2D(64, kernel_size=(6, 2))(conv5_a)
conv6_bd = Dropout(0.2)(BatchNormalization()(conv6))
conv6_a = LeakyReLU(alpha=0.1)(conv6_bd)

outputs_proton = Conv2D(1, kernel_size=(2, 1), activation='relu')(conv3_a)
outputs_neutron = Conv2D(1, kernel_size=(2, 1), activation='relu')(conv6_a)

generator = Model([x, cond], [outputs_proton, outputs_neutron], name='generator')
generator.summary()

############################ discriminator ############################

# PROTON IMAGE
input_img_proton = Input(shape=[56, 30, 1], name='input_img_proton')
conv1 = Conv2D(32, kernel_size=3)(input_img_proton)
conv1_bd = Dropout(0.2)(BatchNormalization()(conv1))
conv1_a = LeakyReLU(alpha=0.1)(conv1_bd)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1_a)

conv2 = Conv2D(16, kernel_size=3)(pool1)
conv2_bd = Dropout(0.2)(BatchNormalization()(conv2))
conv2_a = LeakyReLU(alpha=0.1)(conv2_bd)
pool2 = MaxPooling2D(pool_size=(2, 1))(conv2_a)

flat_proton = Flatten()(pool2)

# NEUTRON IMAGE
input_img_neutron = Input(shape=[44, 44, 1], name='input_img_neutron')
conv3 = Conv2D(32, kernel_size=3)(input_img_neutron)
conv3_bd = Dropout(0.2)(BatchNormalization()(conv3))
conv3_a = LeakyReLU(alpha=0.1)(conv3_bd)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3_a)

conv4 = Conv2D(16, kernel_size=3)(pool3)
conv4_bd = Dropout(0.2)(BatchNormalization()(conv4))
conv4_a = LeakyReLU(alpha=0.1)(conv4_bd)
pool4 = MaxPooling2D(pool_size=(2, 1))(conv4_a)

flat_neutron = Flatten()(pool4)

# CONDITIONAL
cond = Input(shape=(cond_dim,))

inputs2 = Concatenate(axis=1)([flat_proton, flat_neutron, cond])

layer_1 = Dense(128)(inputs2)
layer_1_bd = Dropout(0.2)(BatchNormalization()(layer_1))
layer_1_a = LeakyReLU(alpha=0.1)(layer_1_bd)

layer_2 = Dense(64)(layer_1_a)
layer_2_bd = Dropout(0.2)(BatchNormalization()(layer_2))
layer_2_a = LeakyReLU(alpha=0.1)(layer_2_bd)
outputs = Dense(2, activation='sigmoid')(layer_2_a)

discriminator = Model([input_img_proton, input_img_neutron, cond], [outputs, layer_2_a], name='discriminator')
discriminator.summary()

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    
    # update state of accuracy of real and false images
    d_acc_r.update_state(tf.ones_like(real_output), real_output)
    d_acc_f.update_state(tf.zeros_like(fake_output), fake_output)
    return total_loss

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
d_acc_r = keras.metrics.BinaryAccuracy(name="d_acc_r", threshold=0.5)
d_acc_f = keras.metrics.BinaryAccuracy(name="d_acc_r", threshold=0.5)
g_acc = keras.metrics.BinaryAccuracy(name="g_acc_g", threshold=0.5)

def generator_loss(step, fake_output,
                   fake_latent, fake_latent_2,
                   noise, noise_2,
                   std_proton, std_neutron):

    g_acc.update_state(tf.ones_like(fake_output), fake_output)

    crossentropy_loss = cross_entropy(tf.ones_like(fake_output), fake_output)

    div = tf.math.divide(tf.reduce_mean(tf.abs(fake_latent - fake_latent_2),(1)), tf.reduce_mean(tf.abs(noise-noise_2),(1)))

    div_loss_proton = std_proton * STRENGTH / (div + 1e-5)
    div_loss_neutron = std_neutron * STRENGTH / (div + 1e-5)

    div_loss_proton = tf.reduce_mean(tf.math.multiply(tf.reduce_mean(std_proton,(1)), div_loss_proton))
    div_loss_neutron = tf.reduce_mean(tf.math.multiply(tf.reduce_mean(std_neutron,(1)), div_loss_neutron))

    # average diversity loss
    div_loss = div_loss_proton + div_loss_neutron
    return crossentropy_loss + div_loss, div_loss

EPOCHS = 200
noise_dim = latent_dim
num_examples_to_generate = 16

START_GENERATING_IMG_FROM_IDX = 20
# Seed to reuse for generating samples for comparison during training
seed = tf.random.normal([num_examples_to_generate, noise_dim])
seed_cond = y_test[START_GENERATING_IMG_FROM_IDX:START_GENERATING_IMG_FROM_IDX+num_examples_to_generate]

wandb.login(key="d53387a3b34fda2a3caaf861b5fad88cb4ec99ef")

wandb.finish()
# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="Generative Models for CERN Fast Simulations",
    name=wandb_run_name,
    # track hyperparameters and run metadata
    config={
    "Model": NAME,
    "dataset": "proton_neutron_data",
    "epochs": EPOCHS,
    "Date": DATE_STR,
    "latent_dimension": latent_dim,
    "Proton_min": photon_sum_proton_min,
    "Proton_max": photon_sum_proton_max,
    "Experiment_dir_name": EXPERIMENT_DIR_NAME,
    "batch_size": BATCH_SIZE
    },
    tags=[f"proton_min_{photon_sum_proton_min}",
          f"proton_max_{photon_sum_proton_max}",
          f"neutron_min_{photon_sum_neutron_min}",
          f"neutron_max_{photon_sum_neutron_max}",
          f"gan_strength_{STRENGTH}", "sdi-gan", "two-channels-gen-disc"]
)

from scipy.stats import wasserstein_distance
import pandas as pd
from utils import sum_channels_parallel
from sklearn.metrics import mean_absolute_error

org_p = np.exp(x_test_p)-1
ch_org_p = np.array(org_p).reshape(-1,56,30)
ch_org_p = pd.DataFrame(sum_channels_parallel(ch_org_p)).values
del org_p

org_n = np.exp(x_test_n)-1
ch_org_n = np.array(org_n).reshape(-1,44,44)
ch_org_n = pd.DataFrame(sum_channels_parallel(ch_org_n)).values
del org_n


def calculate_ws_ch(n_calc):
    ws_p = [0,0,0,0,0]
    ws_n = [0,0,0,0,0]
    for j in range(n_calc):
        z = np.random.normal(0,1,(x_test_p.shape[0], latent_dim))
        z_c = y_test
        results_p, results_n = generator.predict([z, z_c])

        results_p = np.exp(results_p)-1
        results_n = np.exp(results_n)-1
        try:
            ch_gen_p = np.array(results_p).reshape(-1,56,30)
            ch_gen_p = pd.DataFrame(sum_channels_parallel(ch_gen_p)).values
            ch_gen_n = np.array(results_n).reshape(-1,44,44)
            ch_gen_n = pd.DataFrame(sum_channels_parallel(ch_gen_n)).values
            for i in range(5):
                ws_p[i] = ws_p[i] + wasserstein_distance(ch_org_p[:,i], ch_gen_p[:,i])
                ws_n[i] = ws_n[i] + wasserstein_distance(ch_org_n[:,i], ch_gen_n[:,i])
            ws_p = np.array(ws_p)
            ws_n = np.array(ws_n)
            ws = ws_p+ws_n
        except ValueError as e:
            print(e)

    ws = ws/n_calc
    ws_mean = ws.sum()/5
    print("ws mean",f'{ws_mean:.2f}', end=" ")
    for n, score in enumerate(ws):
        print("ch"+str(n+1),f'{score:.2f}',end=" ")
    return ws_mean


@tf.function
def train_step(batch, step):
    # dataset proton, dataset 2 proton, dataset neutron, dataset 2 neutron, dataset_cond, dataset_std_proton, dataset_std_neutron, fake_cond
    images_p, images_p_2, images_n, images_n_2, cond, std_proton, std_neutron, noise_cond = batch
    step=step
    BATCH_SIZE = tf.shape(images_p)[0]
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    noise_2 = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # for the same conditional data generate two images from different noises
        generated_images_p, generated_images_n = generator([noise, noise_cond], training=True)
        generated_images_p_2, generated_images_n_2 = generator([noise_2, noise_cond], training=True)
        
        # produce if real image is real or fake
        real_output, real_latent = discriminator([images_p, images_n, cond], training=True)
        # real_output_2,real_latent_2  = discriminator([images_2,cond], training=True)
        
        # produce if generated images from two different latent codes are real or fake
        fake_output, fake_latent = discriminator([generated_images_p, generated_images_n, noise_cond], training=True)
        fake_output_2, fake_latent_2 = discriminator([generated_images_p_2, generated_images_n_2, noise_cond], training=True)

        gen_loss, div_loss = generator_loss(step, fake_output,
                                            fake_latent, fake_latent_2,
                                            noise, noise_2,
                                            std_proton, std_neutron)
        disc_loss = discriminator_loss(real_output, fake_output)

    #         generated_images = generator([noise,noise_cond], training=True)

    #         real_output = discriminator([images,cond], training=True)
    #         fake_output = discriminator([generated_images, noise_cond], training=True)

    #         gen_loss = generator_loss(step, fake_output)
    #         real_loss, fake_loss = discriminator_loss(real_output, fake_output)
    #         disc_loss = real_loss + fake_loss

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss, div_loss


# If model achieves WS metric less or equal to this number, its weights will be saved
WS_MEAN_SAVE_THRESHOLD = 20


if SAVE_EXPERIMENT_DATA:
    filepath_mod = f"../../{EXPERIMENT_DIR_NAME}/models/"
    create_dir(filepath_mod)

history = []
def train(dataset, epochs):
    experiment_start = time.time()
    tf_step = tf.Variable(0, dtype=float)
    step=0

    # generate first image
    generate_and_save_images(generator,
                             epochs,
                             [seed, seed_cond])

    for epoch in range(epochs):
        start = time.time()

        gen_loss_epoch = []
        div_loss_epoch = []
        disc_loss_epoch = []
        for batch in dataset:
            gen_loss, disc_loss, div_loss = train_step(batch,tf_step)

            history.append([gen_loss,disc_loss,
                100*d_acc_r.result().numpy(),
                100*d_acc_f.result().numpy(),
                100*g_acc.result().numpy(),
                ])
            tf_step.assign_add(1)
            step = step+1

            gen_loss_epoch.append(gen_loss)
            disc_loss_epoch.append(disc_loss)
            div_loss_epoch.append(div_loss)
            if step % 100 == 0:
                print("%d [D real acc: %.2f%%] [D fake acc: %.2f%%] [G acc: %.2f%%] "% (
                    step,
                    100*d_acc_r.result().numpy(),
                    100*d_acc_f.result().numpy(),
                    100*g_acc.result().numpy()))

        plot = generate_and_save_images(generator,
                                 epoch,
                                 [seed, seed_cond])

        ws_mean = calculate_ws_ch(min(epoch//5+1,5))

        if SAVE_EXPERIMENT_DATA:
            if ws_mean <= WS_MEAN_SAVE_THRESHOLD:
                # Save the model every epoch
                generator.compile()
                # discriminator.compile()
                generator.save((os.path.join(filepath_mod, "gen_"+NAME + "_"+ str(epoch) +".h5")))
                # discriminator.save((os.path.join(filepath_mod, "disc_"+NAME + "_"+ str(epoch) +".h5")))
                # np.savez(os.path.join(filepath_mod, "history_"+NAME+".npz"),np.array(history))

        wandb.log({
            'ws_mean': ws_mean,
            'gen_loss': np.mean(gen_loss_epoch),
            'div_loss': np.mean(div_loss_epoch),
            'disc_loss': np.mean(disc_loss_epoch),
            'epoch': epoch,
            'plot': wandb.Image(plot),
            'experiment_time': time.time()-experiment_start
        })

        plt.close('all')

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    return history

if SAVE_EXPERIMENT_DATA:
    filepath_img = f"../../{EXPERIMENT_DIR_NAME}/images/"
    create_dir(filepath_img)

    
def generate_and_save_images(model, epoch, test_input):
    START_INDEX = 6
    SUPTITLE_TXT = f"\nModel: GAN proton data" \
               f"\nPhotonsum interval: [{photon_sum_proton_min}, {photon_sum_proton_max}]" \
               f"\nEPOCH: {epoch}"

    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions_p, predictions_n = model(test_input, training=False)
    """
    predictions has shape (n_samples, 56, 44, 2). First channel has proton data, second has neutrons
    """
    fig = plt.figure(figsize=(15,4))

    plt.title(f"EPOCH {epoch}")

    subfigs = fig.subfigures(1, 4)

    for particle_num, subfig in enumerate(subfigs.flat):  # iterate over 4 particles
        subfig.suptitle(f'Particle {particle_num} response')
        axs = subfig.subplots(2, 2)
        
        for i, ax in enumerate(axs.flat):  # iterate over 4 images of single particle
            m_2 = i % 2  # 0 if proton, 1 if neutron
            if i < 2:
                # Real response
                if m_2 == 0:  # proton
                    x = x_test_p[START_INDEX+particle_num].reshape(56, 30)
                else:  # neutron
                    x = x_test_n[START_INDEX+particle_num].reshape(44, 44)
                axs[i//2, m_2].set_title("neutron" if m_2 else "proton")
            else:
                # Generated response
                if m_2 == 0:  # proton
                    x = predictions_p[START_INDEX+particle_num].numpy().reshape(56, 30)
                else:  # neutron
                    x = predictions_n[START_INDEX+particle_num].numpy().reshape(44, 44)
            axs[i//2, m_2].set_axis_off()
            im = axs[i//2, m_2].imshow(x, interpolation='none', cmap='gnuplot')
            fig.colorbar(im, ax=axs[i//2, m_2])

    if SAVE_EXPERIMENT_DATA:
        plt.savefig(os.path.join(filepath_img, 'image_at_epoch_{:04d}.png'.format(epoch)))
    
    return fig

history = train(dataset_with_cond, EPOCHS)