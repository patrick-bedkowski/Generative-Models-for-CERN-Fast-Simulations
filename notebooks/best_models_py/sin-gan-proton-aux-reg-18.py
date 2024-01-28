import time
import os
import wandb

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from utils import sum_channels_parallel, calculate_ws_ch_proton_model, create_dir, save_scales

print(tf.config.list_physical_devices('GPU'))
print(tf.__version__)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

# SETTINGS & PARAMETERS
SAVE_EXPERIMENT_DATA = True
WS_MEAN_SAVE_THRESHOLD = 6
BATCH_SIZE = 128
NOISE_DIM = 10
EPOCHS = 350
LR_G = 1e-4
LR_D = 1e-5
LR_A = 1e-4
DI_STRENGTH = 0.1
IN_STRENGTH = 1e-10
AUX_STRENGTH = 1e-3

# Execute the training 3 times
for _ in range(0, 3):

    data = pd.read_pickle('/net/tscratch/people/plgpbedkowski/data/data_photonsum_proton_18_2312.pkl')
    print('Loaded: ', data.shape, "max:", data.max())

    # Data containing particle conditional data from particle having responses with proton photon sum in interval [70, 2312] without taking into consideration photon sums of neutron responses.
    data_cond = pd.read_pickle('/net/tscratch/people/plgpbedkowski/data/data_cond_photonsum_proton_18_2312.pkl')
    print('Loaded cond: ', data_cond.shape, "max:", data_cond.values.max(), "min:", data_cond.values.min())

    # data of coordinates of maximum value of pixel on the images
    data_posi = pd.read_pickle('/net/tscratch/people/plgpbedkowski/data/data_coord_proton_photonsum_proton_18_2312.pkl')
    print('Loaded cond: ', data_posi.shape, "max:", data_posi.values.max(), "min:", data_posi.values.min())

    # calculate min max proton sum
    photon_sum_proton_min, photon_sum_proton_max = data_cond.proton_photon_sum.min(), data_cond.proton_photon_sum.max()

    DATE_STR = datetime.now().strftime("%d_%m_%Y_%H_%M")
    NAME = "sin-gan-aux-reg-arch-2"  # Modification of SDI-GAN, selective intensity gan
    wandb_run_name = f"{int(photon_sum_proton_min)}_{int(photon_sum_proton_max)}_{DATE_STR}"
    EXPERIMENT_DIR_NAME = f"experiments/{NAME}_{int(photon_sum_proton_min)}_{int(photon_sum_proton_max)}_{DATE_STR}"
    print("Experiment DIR: ", EXPERIMENT_DIR_NAME)

    # group conditional data
    data_cond["cond"] = data_cond["Energy"].astype(str) + "|" + data_cond["Vx"].astype(str) + "|" + data_cond[
        "Vy"].astype(str) + "|" + data_cond["Vz"].astype(str) + "|" + data_cond["Px"].astype(str) + "|" + data_cond[
                            "Py"].astype(str) + "|" + data_cond["Pz"].astype(str) + "|" + data_cond["mass"].astype(
        str) + "|" + data_cond["charge"].astype(str)
    data_cond_id = data_cond[["cond"]].reset_index()
    ids = data_cond_id.merge(data_cond_id.sample(frac=1), on=["cond"], how="inner").groupby("index_x").first()
    ids = ids["index_y"]

    data = np.log(data + 1)
    data = np.float32(data)
    print("data max", data.max(), "min", data.min())

    data_2 = data[ids]
    data_cond = data_cond.drop(columns="cond")

    scaler = MinMaxScaler()
    std = data_cond["std"].values.reshape(-1, 1)
    std = np.float32(std)
    std = scaler.fit_transform(std)
    print("std max", std.max(), "min", std.min())

    proton_photon_sum = np.float32(data_cond["proton_photon_sum"])

    scaler = StandardScaler()
    data_cond = np.float32(data_cond.drop(columns=["std", "proton_photon_sum"]))
    data_cond = scaler.fit_transform(data_cond)
    print("cond max", data_cond.max(), "min", data_cond.min())

    x_train, x_test, x_train_2, x_test_2, \
    y_train, y_test, \
    std_train, std_test, \
    intensity_train, intensity_test, \
    positions_train, positions_test = train_test_split(data, data_2,
                                                       data_cond,
                                                       std,
                                                       proton_photon_sum,
                                                       data_posi,
                                                       test_size=0.2, shuffle=False)

    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    # Save scales
    if SAVE_EXPERIMENT_DATA:
        filepath = f"{EXPERIMENT_DIR_NAME}/scales/"
        create_dir(filepath, SAVE_EXPERIMENT_DATA)
        save_scales("Proton", scaler.mean_, scaler.scale_, filepath)

    dataset = tf.data.Dataset.from_tensor_slices(x_train).batch(batch_size=BATCH_SIZE)
    dataset_2 = tf.data.Dataset.from_tensor_slices(x_train_2).batch(batch_size=BATCH_SIZE)
    dataset_cond = tf.data.Dataset.from_tensor_slices(y_train).batch(batch_size=BATCH_SIZE)
    dataset_std = tf.data.Dataset.from_tensor_slices(std_train).batch(batch_size=BATCH_SIZE)
    dataset_intensity = tf.data.Dataset.from_tensor_slices(intensity_train).batch(batch_size=BATCH_SIZE)
    dataset_positions = tf.data.Dataset.from_tensor_slices(positions_train).batch(batch_size=BATCH_SIZE)
    fake_cond = tf.data.Dataset.from_tensor_slices(y_train).shuffle(12800).batch(batch_size=BATCH_SIZE)
    dataset_with_cond = tf.data.Dataset.zip((dataset, dataset_2, dataset_cond, dataset_std, dataset_intensity, dataset_positions, fake_cond)).shuffle(12800)

    val_dataset = tf.data.Dataset.from_tensor_slices(x_test).batch(batch_size=BATCH_SIZE)
    val_dataset_2 = tf.data.Dataset.from_tensor_slices(x_test_2).batch(batch_size=BATCH_SIZE)
    val_dataset_cond = tf.data.Dataset.from_tensor_slices(y_test).batch(batch_size=BATCH_SIZE)
    val_dataset_std = tf.data.Dataset.from_tensor_slices(std_test).batch(batch_size=BATCH_SIZE)
    val_dataset_intensity = tf.data.Dataset.from_tensor_slices(intensity_test).batch(batch_size=BATCH_SIZE)
    val_dataset_positions = tf.data.Dataset.from_tensor_slices(positions_test).batch(batch_size=128)
    val_fake_cond = tf.data.Dataset.from_tensor_slices(y_test).shuffle(12800).batch(batch_size=BATCH_SIZE)
    val_dataset_with_cond = tf.data.Dataset.zip((val_dataset, val_dataset_2, val_dataset_cond, val_dataset_std, val_dataset_intensity, val_dataset_positions, val_fake_cond)).shuffle(12800)

    from tensorflow.compat.v1.keras.layers import Input, Dense, LeakyReLU, Conv2D, MaxPooling2D, UpSampling2D, \
        Concatenate
    from tensorflow.compat.v1.keras.models import Model
    from tensorflow.compat.v1.keras.layers import Dense, Reshape, Flatten
    from tensorflow.compat.v1.keras.layers import Dropout, BatchNormalization
    from tensorflow import keras

    # Constant for the input dataset with conditional data
    cond_dim = 9

    ############################ generator ############################

    x = Input(shape=(NOISE_DIM,))
    cond = Input(shape=(cond_dim,))
    inputs = Concatenate(axis=1)([x, cond])

    layer_1 = Dense(128 * 2)(inputs)
    layer_1_bd = Dropout(0.2)(BatchNormalization()(layer_1))
    layer_1_a = LeakyReLU(alpha=0.1)(layer_1_bd)

    layer_2 = Dense(128 * 20 * 10)(layer_1_a)
    layer_2_bd = Dropout(0.2)(BatchNormalization()(layer_2))
    layer_2_a = LeakyReLU(alpha=0.1)(layer_2_bd)

    reshaped = Reshape((20, 10, 128))(layer_2_a)
    reshaped_s = UpSampling2D(size=(3, 2))(reshaped)

    conv1 = Conv2D(256, kernel_size=(2, 2))(reshaped_s)
    conv1_bd = Dropout(0.2)(BatchNormalization()(conv1))
    conv1_a = LeakyReLU(alpha=0.1)(conv1_bd)
    conv1_a_s = UpSampling2D(size=(1, 2))(conv1_a)

    conv2 = Conv2D(128, kernel_size=(2, 2))(conv1_a_s)
    conv2_bd = Dropout(0.2)(BatchNormalization()(conv2))
    conv2_a = LeakyReLU(alpha=0.1)(conv2_bd)

    conv3 = Conv2D(64, kernel_size=(2, 2))(conv2_a)
    conv3_bd = Dropout(0.2)(BatchNormalization()(conv3))
    conv3_a = LeakyReLU(alpha=0.1)(conv3_bd)

    outputs = Conv2D(1, kernel_size=(2, 7), activation='relu')(conv3_a)

    generator = Model([x, cond], outputs, name='generator')
    generator.summary()

    ############################ discriminator ############################

    input_img = Input(shape=[56, 30, 1], name='input_img')
    conv1 = Conv2D(32, kernel_size=(3, 3))(input_img)
    conv1_bd = Dropout(0.2)(BatchNormalization()(conv1))
    conv1_a = LeakyReLU(alpha=0.1)(conv1_bd)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1_a)

    conv2 = Conv2D(16, kernel_size=(3, 3))(pool1)
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
    discriminator.summary()


    def discriminator_loss(real_output, fake_output):
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        d_acc_r.update_state(tf.ones_like(real_output), real_output)
        d_acc_f.update_state(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    # AUX REG
    input_img_2 = Input(shape=[56, 30, 1], name='input_img_aux')

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

    outputs_reg = Dense(2)(flat_2)

    aux_reg = Model(input_img_2, outputs_reg, name='aux_reg')

    def regressor_loss(real_coords, fake_coords):
        return tf.reduce_mean(tf.keras.losses.MSE(real_coords, fake_coords))


    generator_optimizer = tf.keras.optimizers.Adam(LR_G)
    discriminator_optimizer = tf.keras.optimizers.Adam(LR_D)
    aux_reg_optimizer = tf.keras.optimizers.Adam(LR_A)

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    d_acc_r = keras.metrics.BinaryAccuracy(name="d_acc_r", threshold=0.5)
    d_acc_f = keras.metrics.BinaryAccuracy(name="d_acc_r", threshold=0.5)
    g_acc = keras.metrics.BinaryAccuracy(name="g_acc_g", threshold=0.5)


    def generator_loss(fake_output, fake_latent, fake_latent_2,
                       noise, noise_2, std):

        g_acc.update_state(tf.ones_like(fake_output), fake_output)
        crossentropy_loss = cross_entropy(tf.ones_like(fake_output), fake_output)

        # SDI-GAN regularization
        div = tf.math.divide(tf.reduce_mean(tf.abs(fake_latent - fake_latent_2), (1)),
                             tf.reduce_mean(tf.abs(noise - noise_2), (1)))
        div_loss = std * DI_STRENGTH / (div + 1e-5)

        div_loss = tf.reduce_mean(tf.math.multiply(tf.reduce_mean(std, (1)), div_loss))

        return crossentropy_loss + div_loss, div_loss


    # Settings for plotting
    SAMPLES_TO_PLOT = [2055, 3015, 670, 428, 628, 206, 2010]
    seed = tf.random.normal([len(SAMPLES_TO_PLOT), NOISE_DIM])
    seed_cond = y_test[SAMPLES_TO_PLOT]

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
        "dataset": "proton_data",
        "epochs": EPOCHS,
        "Date": DATE_STR,
        "Learning rate_generator": LR_G,
        "Learning rate_discriminator": LR_D,
        "Proton_min": photon_sum_proton_min,
        "Proton_max": photon_sum_proton_max,
        "Experiment_dir_name": EXPERIMENT_DIR_NAME,
        "Intensity strength": f"intensity_strength_{IN_STRENGTH}"
        },
        tags=[f"proton_min_{photon_sum_proton_min}",
              f"proton_max_{photon_sum_proton_max}",
              f"diversity_strength_{DI_STRENGTH}", "sin-gan",
              f"intensity_strength_{IN_STRENGTH}"]
    )

    # CALCULATE DISTRIBUTION OF CHANNELS IN ORIGINAL TEST DATA #
    org = np.exp(x_test) - 1
    ch_org = np.array(org).reshape(-1, 56, 30)
    ch_org = pd.DataFrame(sum_channels_parallel(ch_org)).values
    del org


    def calculate_intensity_loss(gen_im_proton, intensity_proton):
        # intensity loss
        sum_all_axes_p = tf.reduce_sum(gen_im_proton)
        mse_value_p = tf.keras.losses.mean_squared_error(intensity_proton, sum_all_axes_p)

        return IN_STRENGTH * mse_value_p


    @tf.function
    def train_step(batch, step):
        images, images_2, cond, std, intensity, true_positions, noise_cond = batch
        step = step
        BATCH_SIZE = tf.shape(images)[0]
        noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
        noise_2 = tf.random.normal([BATCH_SIZE, NOISE_DIM])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape, tf.GradientTape() as aux_tape:
            generated_images = generator([noise, noise_cond], training=True)
            generated_images_2 = generator([noise_2, noise_cond], training=True)

            real_output, real_latent = discriminator([images, cond], training=True)

            fake_output, fake_latent = discriminator([generated_images, noise_cond], training=True)
            fake_output_2, fake_latent_2 = discriminator([generated_images_2, noise_cond], training=True)

            gen_loss, div_loss = generator_loss(fake_output,
                                                fake_latent, fake_latent_2,
                                                noise, noise_2,
                                                std)

            generated_positions = aux_reg(generated_images)

            aux_reg_loss = regressor_loss(true_positions, generated_positions)

            disc_loss = discriminator_loss(real_output, fake_output)
            intensity_loss = calculate_intensity_loss(generated_images, intensity)
            gen_loss += intensity_loss + AUX_STRENGTH * aux_reg_loss

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        gradients_of_aux_reg = aux_tape.gradient(aux_reg_loss, aux_reg.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
        aux_reg_optimizer.apply_gradients(zip(gradients_of_aux_reg, aux_reg.trainable_variables))

        return gen_loss, disc_loss, div_loss, intensity_loss, aux_reg_loss


    if SAVE_EXPERIMENT_DATA:
        filepath_mod = f"{EXPERIMENT_DIR_NAME}/models/"
        create_dir(filepath_mod, SAVE_EXPERIMENT_DATA)

    history = []


    def train(dataset, epochs):
        experiment_start = time.time()
        tf_step = tf.Variable(0, dtype=float)
        step = 0

        for epoch in range(epochs):
            start = time.time()

            gen_loss_epoch = []
            div_loss_epoch = []
            intensity_loss_epoch = []
            disc_loss_epoch = []
            aux_reg_loss_epoch = []
            for batch in dataset:
                gen_loss, disc_loss, div_loss, intensity_loss, aux_reg_loss = train_step(batch, tf_step)

                tf_step.assign_add(1)
                step = step + 1

                gen_loss_epoch.append(gen_loss)
                disc_loss_epoch.append(disc_loss)
                div_loss_epoch.append(div_loss)
                intensity_loss_epoch.append(intensity_loss)
                aux_reg_loss_epoch.append(aux_reg_loss)

            plot = generate_and_save_images(generator,
                                            epoch,
                                            [seed, seed_cond])

            ws_mean = calculate_ws_ch_proton_model(min(epoch//5+1, 5),
                                                   x_test, y_test,
                                                   generator, ch_org,
                                                   NOISE_DIM)

            if SAVE_EXPERIMENT_DATA:
                if ws_mean <= WS_MEAN_SAVE_THRESHOLD:
                    # Save the model every epoch
                    generator.compile()
                    generator.save((os.path.join(filepath_mod, "gen_" + NAME + "_" + str(epoch) + ".h5")))

            wandb.log({
                'ws_mean': ws_mean,
                'gen_loss': np.mean(gen_loss_epoch),
                'div_loss': np.mean(div_loss_epoch),
                'intensity_loss': np.mean(intensity_loss_epoch),
                'aux_reg_loss': np.mean(aux_reg_loss_epoch),
                'disc_loss': np.mean(disc_loss_epoch),
                'epoch': epoch,
                'plot': wandb.Image(plot),
                'experiment_time': time.time() - experiment_start
            })

            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

        return history


    def generate_and_save_images(model, epoch, test_input):
        SUPTITLE_TXT = f"\nModel: SDI-GAN + reg. + aux. reg. proton data" \
                   f"\nPhotonsum interval: [{photon_sum_proton_min}, {photon_sum_proton_max}]" \
                   f"\nEPOCH: {epoch}"

        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        predictions = model(test_input, training=False)  # returns 5 responses

        fig, axs = plt.subplots(2, 7, figsize=(15, 5))
        fig.suptitle(SUPTITLE_TXT, x=0.1, horizontalalignment='left')

        for i, sample_x_test in zip(list(range(0, 7)), SAMPLES_TO_PLOT):
            x_1 = x_test[sample_x_test].reshape(56, 30)
            im_1 = axs[0, i % 7].imshow(x_1, cmap='gnuplot')

            x_2 = predictions[i - 7].numpy().reshape(56, 30)
            im_2 = axs[1, i % 7].imshow(x_2, cmap='gnuplot')

            axs[0, i % 7].axis('off')
            axs[1, i % 7].axis('off')
            fig.colorbar(im_1, ax=axs[0, i % 7])
            fig.colorbar(im_2, ax=axs[1, i % 7])

        fig.tight_layout(rect=[0, 0, 1, 0.975])
        return fig

    history = train(dataset_with_cond, EPOCHS)
