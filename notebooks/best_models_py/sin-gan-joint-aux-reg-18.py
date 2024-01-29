import wandb
import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))

from sklearn.model_selection import train_test_split

import tensorflow as tf
print(tf.__version__)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import os
import time

import sklearn
from sklearn.preprocessing import StandardScaler
from datetime import datetime

SAVE_EXPERIMENT_DATA = True

for _ in range(0, 3):

    data = pd.read_pickle('/net/tscratch/people/plgpbedkowski/data/data_proton_neutron_photonsum_proton_18_1970_neutron_18_3249_padding.pkl')
    print('Loaded: ', data.shape, "max:", data.max())

    # Data containing particle conditional data from particle having responses with proton photon sum in interval [70, 2312] without taking into consideration photon sums of neutron responses.
    data_cond = pd.read_pickle('/net/tscratch/people/plgpbedkowski/data/data_cond_stddev_photonsum_p_18_n_18.pkl')
    print('Loaded cond: ', data_cond.shape, "max:", data_cond.values.max(), "min:", data_cond.values.min())

    # data of coordinates of maximum value of pixel on the images
    data_posi = pd.read_pickle('/net/tscratch/people/plgpbedkowski/data/data_coord_proton_neutron_photonsum_18.pkl')
    print('Loaded cond: ', data_posi.shape)

    # calculate min max proton sum
    photon_sum_proton_min, photon_sum_proton_max = data_cond.proton_photon_sum.min(), data_cond.proton_photon_sum.max()
    photon_sum_neutron_min, photon_sum_neutron_max = data_cond.neutron_photon_sum.min(), data_cond.neutron_photon_sum.max()

    print(data_cond.columns)
    data_cond.columns, len(data_cond.columns)

    WS_MEAN_SAVE_THRESHOLD = 6.5

    IN_STRENGTH = 1e-10

    DI_STRENGTH = 0.1

    AUX_STRENGTH = 1e-3

    DATE_STR = datetime.now().strftime("%d_%m_%Y_%H_%M")

    NAME = "sin-gan-join-aux-reg-arch-2"

    wandb_run_name = f"{int(photon_sum_proton_min)}_{int(photon_sum_proton_max)}_{int(photon_sum_neutron_min)}_{int(photon_sum_neutron_max)}_{DATE_STR}"

    EXPERIMENT_DIR_NAME = f"experiments/{NAME}_{int(photon_sum_proton_min)}_{int(photon_sum_proton_max)}_{int(photon_sum_neutron_min)}_{int(photon_sum_neutron_max)}_{DATE_STR}"

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
            filepath = f"{EXPERIMENT_DIR_NAME}/scales/"
            create_dir(filepath)
            with open(filepath+out_fnm, mode="w") as f:
                f.write(res)


    data_cond["cond"] = data_cond["Energy"].astype(str) +"|"+ data_cond["Vx"].astype(str) +"|"+  data_cond["Vy"].astype(str) +"|"+ data_cond["Vz"].astype(str) +"|"+  data_cond["Px"].astype(str) +"|"+  data_cond["Py"].astype(str) +"|"+ data_cond["Pz"].astype(str) +"|"+  data_cond["mass"].astype(str) +"|"+  data_cond["charge"].astype(str)


    data_cond_id = data_cond[["cond"]].reset_index()


    ids = data_cond_id.merge(data_cond_id.sample(frac=1), on=["cond"], how="inner").groupby("index_x").first()
    ids = ids["index_y"]

    from sklearn.preprocessing import MinMaxScaler

    data = np.log(data+1)
    data = np.float32(data)
    print("data max", data.max(), "min", data.min())

    data_2 = data[ids]

    data_cond = data_cond.drop(columns="cond")

    scaler = MinMaxScaler()
    std_proton = data_cond["std_proton"].values.reshape(-1, 1)
    std_proton = np.float32(std_proton)
    std_proton = scaler.fit_transform(std_proton)
    print("std max", std_proton.max(), "min", std_proton.min())

    scaler = MinMaxScaler()
    std_neutron = data_cond["std_neutron"].values.reshape(-1, 1)
    std_neutron = np.float32(std_neutron)
    std_neutron = scaler.fit_transform(std_neutron)
    print("std max", std_neutron.max(), "min", std_neutron.min())

    proton_photon_sum = np.float32(data_cond["proton_photon_sum"])
    neutron_photon_sum = np.float32(data_cond["neutron_photon_sum"])

    scaler = StandardScaler()
    data_cond = np.float32(data_cond.drop(columns=["std_proton", "std_neutron", "proton_photon_sum", "neutron_photon_sum"]))
    data_cond = scaler.fit_transform(data_cond)
    print("cond max", data_cond.max(), "min", data_cond.min())

    data_posi_proton = data_posi.copy()[["max_x_proton", "max_y_proton"]]
    data_posi_neutron = data_posi.copy()[["max_x_neutron", "max_y_neutron"]]
    # # AUX REG
    # scaler_poz = StandardScaler()
    # data_xy_proton = np.float32(data_posi.copy()[["max_x_proton", "max_y_proton"]])
    # data_xy_proton = scaler_poz.fit_transform(data_xy_proton)
    # print('Load', data_xy_proton.shape, "cond max", data_xy_proton.max(), "min", data_xy_proton.min())
    #
    # # AUX REG
    # scaler_poz = StandardScaler()
    # data_xy_neutron = np.float32(data_posi.copy()[["max_x_neutron", "max_y_neutron"]])
    # data_xy_neutron = scaler_poz.fit_transform(data_xy_neutron)
    # print('Load', data_xy_neutron.shape, "cond max", data_xy_neutron.max(), "min", data_xy_neutron.min())

    x_train, x_test, x_train_2, x_test_2,\
    y_train, y_test, \
    std_p_train, std_p_test, \
    std_n_train, std_n_test, \
    intensity_p_train, intensity_p_test, \
    intensity_n_train, intensity_n_test, \
    positions_p_train, positions_p_test,\
    positions_n_train, positions_n_test = train_test_split(data, data_2,
                                                           data_cond,
                                                           std_proton,
                                                           std_neutron,
                                                           proton_photon_sum,
                                                           neutron_photon_sum,
                                                           data_posi_proton,
                                                           data_posi_neutron,
                                                           test_size=0.2, shuffle=False)

    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    #save scales
    if SAVE_EXPERIMENT_DATA:
        save_scales("Proton", scaler.mean_, scaler.scale_)

    BATCH_SIZE = 128
    dataset = tf.data.Dataset.from_tensor_slices(x_train).batch(batch_size=BATCH_SIZE)
    dataset_2 = tf.data.Dataset.from_tensor_slices(x_train_2).batch(batch_size=BATCH_SIZE)
    dataset_cond = tf.data.Dataset.from_tensor_slices(y_train).batch(batch_size=BATCH_SIZE)
    dataset_std_p = tf.data.Dataset.from_tensor_slices(std_p_train).batch(batch_size=BATCH_SIZE)
    dataset_std_n = tf.data.Dataset.from_tensor_slices(std_n_train).batch(batch_size=BATCH_SIZE)
    dataset_intensity_p = tf.data.Dataset.from_tensor_slices(intensity_p_train).batch(batch_size=BATCH_SIZE)
    dataset_intensity_n = tf.data.Dataset.from_tensor_slices(intensity_n_train).batch(batch_size=BATCH_SIZE)
    dataset_positions_p = tf.data.Dataset.from_tensor_slices(positions_p_train).batch(batch_size=BATCH_SIZE)
    dataset_positions_n = tf.data.Dataset.from_tensor_slices(positions_n_train).batch(batch_size=BATCH_SIZE)
    fake_cond = tf.data.Dataset.from_tensor_slices(y_train).shuffle(12800).batch(batch_size=BATCH_SIZE)
    dataset_with_cond = tf.data.Dataset.zip((dataset, dataset_2, dataset_cond,
                                             dataset_std_p, dataset_std_n,
                                             dataset_intensity_p, dataset_intensity_n,
                                             dataset_positions_p, dataset_positions_n, fake_cond)).shuffle(12800)

    val_dataset = tf.data.Dataset.from_tensor_slices(x_test).batch(batch_size=BATCH_SIZE)
    val_dataset_2 = tf.data.Dataset.from_tensor_slices(x_test_2).batch(batch_size=BATCH_SIZE)
    val_dataset_cond = tf.data.Dataset.from_tensor_slices(y_test).batch(batch_size=BATCH_SIZE)
    val_dataset_std_p = tf.data.Dataset.from_tensor_slices(std_p_test).batch(batch_size=BATCH_SIZE)
    val_dataset_std_n = tf.data.Dataset.from_tensor_slices(std_n_test).batch(batch_size=BATCH_SIZE)
    val_dataset_intensity_p = tf.data.Dataset.from_tensor_slices(intensity_p_test).batch(batch_size=BATCH_SIZE)
    val_dataset_intensity_n = tf.data.Dataset.from_tensor_slices(intensity_n_test).batch(batch_size=BATCH_SIZE)
    val_dataset_positions_p = tf.data.Dataset.from_tensor_slices(positions_p_test).batch(batch_size=BATCH_SIZE)
    val_dataset_positions_n = tf.data.Dataset.from_tensor_slices(positions_n_test).batch(batch_size=BATCH_SIZE)
    val_fake_cond =  tf.data.Dataset.from_tensor_slices(y_test).shuffle(12800).batch(batch_size=BATCH_SIZE)
    val_dataset_with_cond = tf.data.Dataset.zip((val_dataset, val_dataset_2, val_dataset_cond,
                                                 val_dataset_std_p, val_dataset_std_n,
                                                 val_dataset_intensity_p, val_dataset_intensity_n,
                                                 val_dataset_positions_p, val_dataset_positions_n, val_fake_cond)).shuffle(12800)

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

    latent_dim = 10
    cond_dim = 9

    ############################ generator ############################

    x = Input(shape=(latent_dim,))
    cond = Input(shape=(cond_dim,))
    inputs = Concatenate(axis=1)([x, cond])

    layer_1 = Dense(128*2)(inputs)
    layer_1_bd = Dropout(0.2)(BatchNormalization()(layer_1))
    layer_1_a = LeakyReLU(alpha=0.1)(layer_1_bd)

    layer_2 = Dense(128*20*12)(layer_1_a)
    layer_2_bd = Dropout(0.2)(BatchNormalization()(layer_2))
    layer_2_a = LeakyReLU(alpha=0.1)(layer_2_bd)

    reshaped = Reshape((20,12,128))(layer_2_a)
    reshaped_s = UpSampling2D(size=(3,2))(reshaped)

    conv1 = Conv2D(256, kernel_size=(2, 2))(reshaped_s)
    conv1_bd = Dropout(0.2)(BatchNormalization()(conv1))
    conv1_a = LeakyReLU(alpha=0.1)(conv1_bd)
    conv1_a_s = UpSampling2D(size=(1, 2))(conv1_a)

    conv2 = Conv2D(128, kernel_size=2)(conv1_a_s)
    conv2_bd = Dropout(0.2)(BatchNormalization()(conv2))
    conv2_a = LeakyReLU(alpha=0.1)(conv2_bd)

    conv3 = Conv2D(64, kernel_size=2)(conv2_a)
    conv3_bd = Dropout(0.2)(BatchNormalization()(conv3))
    conv3_a = LeakyReLU(alpha=0.1)(conv3_bd)

    outputs = Conv2D(2, kernel_size=(2, 1), activation='relu')(conv3_a)

    generator = Model([x, cond], outputs, name='generator')
    generator.summary()

    ############################ discriminator ############################

    input_img = Input(shape=[56,44,2],name='input_img')
    conv1 = Conv2D(32, kernel_size=3)(input_img)
    conv1_bd = Dropout(0.2)(BatchNormalization()(conv1))
    conv1_a = LeakyReLU(alpha=0.1)(conv1_bd)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1_a)

    conv2 = Conv2D(16, kernel_size=3)(pool1)
    conv2_bd = Dropout(0.2)(BatchNormalization()(conv2))
    conv2_a = LeakyReLU(alpha=0.1)(conv2_bd)
    pool2 = MaxPooling2D(pool_size=(2, 1))(conv2_a)

    flat = Flatten()(pool2)
    cond = Input(shape=(cond_dim,))
    inputs2 = Concatenate(axis=1)([flat, cond])
    layer_1 = Dense(128)(inputs2)
    layer_1_bd = Dropout(0.2)(BatchNormalization()(layer_1))
    layer_1_a = LeakyReLU(alpha=0.1)(layer_1_bd)

    layer_2 = Dense(64)(layer_1_a)
    layer_2_bd = Dropout(0.2)(BatchNormalization()(layer_2))
    layer_2_a = LeakyReLU(alpha=0.1)(layer_2_bd)

    outputs = Dense(1, activation='sigmoid')(layer_2_a)

    discriminator = Model([input_img, cond], [outputs, layer_2_a], name='discriminator')

    def discriminator_loss(real_output, fake_output):
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        d_acc_r.update_state(tf.ones_like(real_output), real_output)
        d_acc_f.update_state(tf.zeros_like(fake_output), fake_output)
        return total_loss

    # AUX REG
    input_img_2 = Input(shape=[56, 44, 2], name='input_img_aux')

    conv3 = Conv2D(32, kernel_size=3)(input_img_2)
    conv3_bd = Dropout(0.2)(BatchNormalization()(conv3))
    conv3_a = LeakyReLU(alpha=0.1)(conv3_bd)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3_a)

    conv4 = Conv2D(64, kernel_size=3)(pool3)
    conv4_bd = Dropout(0.2)(BatchNormalization()(conv4))
    conv4_a = LeakyReLU(alpha=0.1)(conv4_bd)
    pool4 = MaxPooling2D(pool_size=(2, 1))(conv4_a)

    conv5 = Conv2D(128, kernel_size=3)(pool4)
    conv5_bd = Dropout(0.2)(BatchNormalization()(conv5))
    conv5_a = LeakyReLU(alpha=0.1)(conv5_bd)
    pool5 = MaxPooling2D(pool_size=(2, 1))(conv5_a)

    conv6 = Conv2D(256, kernel_size=3)(pool5)
    conv6_bd = Dropout(0.2)(BatchNormalization()(conv6))
    conv6_a = LeakyReLU(alpha=0.1)(conv6_bd)

    flat_2 = Flatten()(conv6_a)

    outputs_reg = Dense(4)(flat_2)

    aux_reg = Model(input_img_2, outputs_reg, name='aux_reg')

    def regressor_loss(real_coords_p, real_coords_n,
                       fake_coords_p, fake_coords_n):
        print('----------')
        print(tf.shape(real_coords_p))
        print(tf.shape(real_coords_n))
        print(tf.shape(fake_coords_p))
        print(tf.shape(fake_coords_n))
        print('----------')

        return tf.reduce_mean(tf.keras.losses.MSE(real_coords_p, fake_coords_p)) + \
               tf.reduce_mean(tf.keras.losses.MSE(real_coords_n, fake_coords_n))

    LR_D = 1e-5
    LR_G = 1e-4
    LR_A = 1e-4
    generator_optimizer = tf.keras.optimizers.Adam(LR_G)
    discriminator_optimizer = tf.keras.optimizers.Adam(LR_D)
    aux_reg_optimizer = tf.keras.optimizers.Adam(LR_A)

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    d_acc_r = keras.metrics.BinaryAccuracy(name="d_acc_r", threshold=0.5)
    d_acc_f = keras.metrics.BinaryAccuracy(name="d_acc_r", threshold=0.5)
    g_acc = keras.metrics.BinaryAccuracy(name="g_acc_g", threshold=0.5)

    def generator_loss(step, fake_output,
                       fake_latent, fake_latent_2, noise, noise_2,
                       std_proton, std_neutron): # SDI GAN PARAMETERS

        g_acc.update_state(tf.ones_like(fake_output), fake_output)
        crossentropy_loss = cross_entropy(tf.ones_like(fake_output), fake_output)

        # SDI-GAN regularization
        div = tf.math.divide(tf.reduce_mean(tf.abs(fake_latent - fake_latent_2),(1)), tf.reduce_mean(tf.abs(noise-noise_2),(1)))

        div_loss_proton = std_proton * DI_STRENGTH / (div + 1e-5)
        div_loss_neutron = std_neutron * DI_STRENGTH / (div + 1e-5)

        div_loss_proton = tf.reduce_mean(tf.math.multiply(tf.reduce_mean(std_proton, (1)), div_loss_proton))
        div_loss_neutron = tf.reduce_mean(tf.math.multiply(tf.reduce_mean(std_neutron, (1)), div_loss_neutron))

        div_loss = div_loss_proton + div_loss_neutron

        return crossentropy_loss + div_loss, div_loss

    EPOCHS = 350
    noise_dim = latent_dim
    # Seed to reuse for generating samples for comparison during training
    SAMPLES_TO_PLOT = [2055, 3015, 1670, 128, 228, 106, 110]

    num_examples_to_generate = 16

    START_GENERATING_IMG_FROM_IDX = 20
    # Seed to reuse for generating samples for comparison during training
    seed = tf.random.normal([num_examples_to_generate, noise_dim])
    seed_cond = y_test[START_GENERATING_IMG_FROM_IDX:START_GENERATING_IMG_FROM_IDX + num_examples_to_generate]

    wandb.finish()
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="Generative-Models-for-CERN-Fast-Simulations-mine",
        entity="bedkowski-patrick",
        name=wandb_run_name,
        # track hyperparameters and run metadata
        config={
        "Model": NAME,
        "dataset": "proton_neutron_data",
        "epochs": EPOCHS,
        "Date": DATE_STR,
        "Learning rate_generator": LR_G,
        "Learning rate_discriminator": LR_D,
        "Proton_min": photon_sum_proton_min,
        "Proton_max": photon_sum_proton_max,
        "noise_dim": noise_dim,
        "Experiment_dir_name": EXPERIMENT_DIR_NAME,
        "Intensity strength": f"intensity_strength_{IN_STRENGTH}",
        "batch_size": BATCH_SIZE
        },
        tags=[f"proton_min_{photon_sum_proton_min}",
              f"proton_max_{photon_sum_proton_max}",
              f"diversity_strength_{DI_STRENGTH}", "sin-gan",
              f"intensity_strength_{IN_STRENGTH}"]
    )

    from scipy.stats import wasserstein_distance
    import pandas as pd
    from utils import sum_channels_parallel, get_max_value_image_coordinates
    from sklearn.metrics import mean_absolute_error

    org=np.exp(x_test)-1
    ch_org = np.array(org).reshape(-1,56,44)
    ch_org = pd.DataFrame(sum_channels_parallel(ch_org)).values
    del org


    def calculate_ws_ch(n_calc):
        ws = [0, 0, 0, 0, 0]
        for j in range(n_calc):
            z = np.random.normal(0, 1, (x_test.shape[0], latent_dim))
            z_c = y_test
            results = generator.predict([z, z_c])
            results = np.exp(results) - 1
            try:
                ch_gen = np.array(results).reshape(-1, 56, 44)
                ch_gen = pd.DataFrame(sum_channels_parallel(ch_gen)).values
                for i in range(5):
                    ws[i] = ws[i] + wasserstein_distance(ch_org[:, i], ch_gen[:, i])
                ws = np.array(ws)
            except ValueError as e:
                print(e)

        ws = ws / n_calc
        ws_mean = ws.sum() / 5
        print("ws mean", f'{ws_mean:.2f}', end=" ")
        for n, score in enumerate(ws):
            print("ch" + str(n + 1), f'{score:.2f}', end=" ")
        return ws_mean


    def calculate_intensity_loss(gen_im_proton, gen_im_neutron,
                                 intensity_proton, intensity_neutron):
        # intensity loss
        sum_all_axes_p = tf.reduce_sum(gen_im_proton)
        mse_value_p = tf.keras.losses.mean_squared_error(intensity_proton, sum_all_axes_p)

        sum_all_axes_n = tf.reduce_sum(gen_im_neutron)
        mse_value_n = tf.keras.losses.mean_squared_error(intensity_neutron, sum_all_axes_n)

        return IN_STRENGTH * (mse_value_p + mse_value_n)


    @tf.function
    def train_step(batch,step):
        images, images_2, cond,\
        std_p, std_n,\
        intensity_p, intensity_n,\
        true_positions_p, true_positions_n,\
        noise_cond = batch

        step=step
        BATCH_SIZE = tf.shape(images)[0]
        noise = tf.random.normal([BATCH_SIZE, noise_dim])
        noise_2 = tf.random.normal([BATCH_SIZE, noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape, tf.GradientTape() as aux_tape:
            generated_images = generator([noise,noise_cond], training=True)
            generated_images_2 = generator([noise_2,noise_cond], training=True)

            real_output, real_latent = discriminator([images, cond], training=True)

            fake_output, fake_latent = discriminator([generated_images, noise_cond], training=True)
            fake_output_2, fake_latent_2 = discriminator([generated_images_2, noise_cond], training=True)

            gen_loss, div_loss = generator_loss(step, fake_output,
                                                fake_latent, fake_latent_2,
                                                noise, noise_2,
                                                std_p, std_n)

            generated_positions = aux_reg(generated_images)
            # split positions

            generated_positions_p, generated_positions_n = generated_positions[:, :2], generated_positions[:, 2:]

            print(generated_positions_p)
            print(generated_positions_n)

            aux_reg_loss = regressor_loss(true_positions_p, true_positions_n,
                                          generated_positions_p, generated_positions_n)

            disc_loss = discriminator_loss(real_output, fake_output)
            generated_images_p, generated_images_n = generated_images[:,:,:,0], generated_images[:,:,:,1]

            intensity_loss = calculate_intensity_loss(generated_images_p, generated_images_n, intensity_p, intensity_n)
            gen_loss += intensity_loss + AUX_STRENGTH*aux_reg_loss

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        gradients_of_aux_reg = aux_tape.gradient(aux_reg_loss, aux_reg.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
        aux_reg_optimizer.apply_gradients(zip(gradients_of_aux_reg, aux_reg.trainable_variables))

        return gen_loss, disc_loss, div_loss, intensity_loss, aux_reg_loss


    if SAVE_EXPERIMENT_DATA:
        filepath_mod = f"{EXPERIMENT_DIR_NAME}/models/"
        create_dir(filepath_mod)

    def train(dataset, epochs):
        experiment_start = time.time()
        tf_step = tf.Variable(0, dtype=float)
        step=0


        for epoch in range(epochs):
            start = time.time()

            gen_loss_epoch = []
            div_loss_epoch = []
            intensity_loss_epoch = []
            disc_loss_epoch = []
            aux_reg_loss_epoch = []
            for batch in dataset:
                gen_loss, disc_loss, div_loss, intensity_loss, aux_reg_loss=train_step(batch,tf_step)

                tf_step.assign_add(1)
                step = step+1

                gen_loss_epoch.append(gen_loss)
                disc_loss_epoch.append(disc_loss)
                div_loss_epoch.append(div_loss)
                intensity_loss_epoch.append(intensity_loss)
                aux_reg_loss_epoch.append(aux_reg_loss)

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
                'intensity_loss': np.mean(intensity_loss_epoch),
                'aux_reg_loss': np.mean(aux_reg_loss_epoch),
                'disc_loss': np.mean(disc_loss_epoch),
                'epoch': epoch,
                'plot': wandb.Image(plot),
                'experiment_time': time.time()-experiment_start
            })

            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

        return


    def generate_and_save_images(model, epoch, test_input):
        START_INDEX = 6
        SUPTITLE_TXT = f"\nModel: SINGAN reg aux" \
                       f"\nPhotonsum interval: [{photon_sum_proton_min}, {photon_sum_proton_max}]" \
                       f"\nEPOCH: {epoch}"

        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        predictions = model(test_input, training=False)
        """
        predictions has shape (n_samples, 56, 44, 2). First channel has proton data, second has neutrons
        """
        fig = plt.figure(figsize=(15, 4))

        plt.title(f"EPOCH {epoch}")

        subfigs = fig.subfigures(1, 4)

        for particle_num, subfig in enumerate(subfigs.flat):  # iterate over 4 particles
            subfig.suptitle(f'Particle {particle_num} response')
            axs = subfig.subplots(2, 2)
            for i, ax in enumerate(axs.flat):  # iterate over 4 images of single particle
                m_2 = i % 2  # 0 if proton, 1 if neutron
                if i < 2:
                    # Real response
                    x = x_test[START_INDEX + particle_num][:, :, m_2].reshape(56, 44)
                    axs[i // 2, m_2].set_title("neutron" if m_2 else "proton")
                else:
                    # Generated response
                    x = predictions[START_INDEX + particle_num].numpy()[:, :, m_2].reshape(56, 44)
                axs[i // 2, m_2].set_axis_off()
                im = axs[i // 2, m_2].imshow(x, interpolation='none', cmap='gnuplot')
                fig.colorbar(im, ax=axs[i // 2, m_2])

        return fig

    history = train(dataset_with_cond, EPOCHS)
