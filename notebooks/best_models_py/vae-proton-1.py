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
WS_MEAN_SAVE_THRESHOLD = 10
BATCH_SIZE = 128
LATENT_DIM = 10
EPOCHS = 300
LR = 1e-4

for _ in range(0, 3):
    data = pd.read_pickle('/net/tscratch/people/plgpbedkowski/data/data_proton_photonsum_proton_1_2312.pkl')
    print('Loaded: ', data.shape, "max:", data.max())

    # Data containing particle conditional data from particle having responses with proton photon sum in interval [1, 2312]
    data_cond = pd.read_pickle('/net/tscratch/people/plgpbedkowski/data/data_cond_photonsum_proton_1_2312.pkl')
    print('Loaded cond: ', data_cond.shape, "max:", data_cond.values.max(), "min:", data_cond.values.min())

    # calculate min max proton sum
    photon_sum_proton_min, photon_sum_proton_max = data_cond.proton_photon_sum.min(), data_cond.proton_photon_sum.max()
    DATE_STR = datetime.now().strftime("%d_%m_%Y_%H_%M")
    NAME = "vae"
    wandb_run_name = f"{int(photon_sum_proton_min)}_{int(photon_sum_proton_max)}_{DATE_STR}"
    EXPERIMENT_DIR_NAME = f"experiments/{NAME}_{int(photon_sum_proton_min)}_{int(photon_sum_proton_max)}_{DATE_STR}"
    print("Experiment DIR: ", EXPERIMENT_DIR_NAME)

    data = np.log(data+1)
    data = np.float32(data)
    print("data max", data.max(), "min", data.min())

    scaler = StandardScaler()
    data_cond = np.float32(data_cond.drop(columns=["std_proton", "proton_photon_sum"]))
    data_cond = scaler.fit_transform(data_cond)
    print("cond max", data_cond.max(), "min", data_cond.min())

    x_train, x_test, y_train, y_test, = train_test_split(data, data_cond, test_size=0.2, shuffle=False, random_state=42)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    # Save scales
    if SAVE_EXPERIMENT_DATA:
        filepath = f"{EXPERIMENT_DIR_NAME}/scales/"
        create_dir(filepath, SAVE_EXPERIMENT_DATA)
        save_scales("Proton", scaler.mean_, scaler.scale_, filepath)

    dataset = tf.data.Dataset.from_tensor_slices(x_train).batch(batch_size=BATCH_SIZE)
    dataset_cond = tf.data.Dataset.from_tensor_slices(y_train).batch(batch_size=BATCH_SIZE)
    dataset_with_cond = tf.data.Dataset.zip((dataset, dataset_cond)).shuffle(12800)

    val_dataset = tf.data.Dataset.from_tensor_slices(x_test).batch(batch_size=BATCH_SIZE)
    val_dataset_cond = tf.data.Dataset.from_tensor_slices(y_test).batch(batch_size=BATCH_SIZE)
    val_dataset_with_cond = tf.data.Dataset.zip((val_dataset, val_dataset_cond)).shuffle(12800)

    from tensorflow.compat.v1.keras.layers import Input, Dense, LeakyReLU, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Dense, Reshape, Flatten, Dropout,BatchNormalization
    from tensorflow.compat.v1.keras import layers
    from tensorflow.compat.v1.keras.models import Model
    from tensorflow import keras


    class Sampling(layers.Layer):
        """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

        def call(self, inputs):
            z_mean, z_log_var = inputs
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    cond_dim = 9

    ############################ encoder ############################
    input_img = Input(shape=[56, 30, 1], name='input_img')
    input_cond = Input(shape=cond_dim, name='input_cond')
    x = Conv2D(32, kernel_size=4, strides=2, padding='same')(input_img)
    x = Conv2D(64, kernel_size=4, strides=2, padding='same')(x)
    x = Conv2D(128, kernel_size=4, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Flatten()(x)
    x = layers.concatenate([input_cond, x])
    x = layers.Dense(LATENT_DIM * 2, activation="relu")(x)
    z_mean = layers.Dense(LATENT_DIM, name="z_mean")(x)
    z_log_var = layers.Dense(LATENT_DIM, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])

    encoder = keras.Model([input_img, input_cond], [z_mean, z_log_var, z], name="encoder")
    encoder.summary()

    ############################ decoder ############################

    x = Input(shape=(LATENT_DIM,))
    cond = Input(shape=(cond_dim,))
    inputs = Concatenate(axis=1)([x, cond])

    g = Dense(7 * 4 * 128)(inputs)
    g = Reshape((7, 4, 128))(g)

    g = UpSampling2D()(g)
    g = Conv2D(128, kernel_size=4, padding='same')(g)
    g = BatchNormalization()(g)
    g = LeakyReLU(alpha=0)(g)

    g = UpSampling2D()(g)
    g = Conv2D(64, kernel_size=4, padding='same')(g)
    g = BatchNormalization()(g)
    g = LeakyReLU(alpha=0)(g)

    g = UpSampling2D()(g)
    g = Conv2D(32, kernel_size=4, padding='same')(g)
    g = BatchNormalization()(g)
    g = LeakyReLU(alpha=0)(g)

    outputs = Conv2D(1, kernel_size=(1, 3), activation='relu')(g)

    generator = Model([x, cond], outputs, name='generator')
    generator.summary()

    # define optimizer
    vae_optimizer = tf.keras.optimizers.RMSprop(LR)

    num_examples_to_generate = 16
    START_GENERATING_IMG_FROM_IDX = 20
    # Seed to reuse for generating samples for comparison during training
    seed = tf.random.normal([num_examples_to_generate, LATENT_DIM])
    seed_cond = y_test[START_GENERATING_IMG_FROM_IDX:START_GENERATING_IMG_FROM_IDX+num_examples_to_generate]

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
        "LR":LR,
        "Proton_min": photon_sum_proton_min,
        "Proton_max": photon_sum_proton_max,
        "Experiment_dir_name": EXPERIMENT_DIR_NAME,
        },
        tags=[f"proton_min_{photon_sum_proton_min}",
              f"proton_max_{photon_sum_proton_max}",
              "vae"]
    )

    from scipy.stats import wasserstein_distance
    import pandas as pd
    from utils import sum_channels_parallel
    from sklearn.metrics import mean_absolute_error

    org=np.exp(x_test)-1
    ch_org = np.array(org).reshape(-1,56,30)
    ch_org = pd.DataFrame(sum_channels_parallel(ch_org)).values
    del org

    def calculate_ws_ch(n_calc):
        ws=[0,0,0,0,0]
        for j in range(n_calc):
            z = np.random.normal(0,1,(x_test.shape[0],10))
            z_c = y_test
            results = generator.predict([z,z_c])
            results = np.exp(results)-1
            try:
                ch_gen = np.array(results).reshape(-1,56,30)
                ch_gen = pd.DataFrame(sum_channels_parallel(ch_gen)).values
                for i in range(5):
                    ws[i] = ws[i] + wasserstein_distance(ch_org[:,i], ch_gen[:,i])
                ws = np.array(ws)
            except ValueError as e:
                print(e)

        ws = ws/n_calc
        ws_mean = ws.sum()/5
        print("ws mean",f'{ws_mean:.2f}', end=" ")
        for n, score in enumerate(ws):
            print("ch"+str(n+1), f'{score:.2f}', end=" ")
        return ws_mean


    @tf.function
    def train_step(batch, step):

        images, cond = batch
        step = step

        # train vae
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = encoder([images, cond])
            reconstruction = generator([z, cond])
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mean_squared_error(tf.reshape(images, (-1, 56, 30, 1)), reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = 0.7 * kl_loss + reconstruction_loss
        grads = tape.gradient(total_loss, generator.trainable_weights + encoder.trainable_weights)
        vae_optimizer.apply_gradients(zip(grads, generator.trainable_weights + encoder.trainable_weights))

        return total_loss, reconstruction_loss, kl_loss

    if SAVE_EXPERIMENT_DATA:
        filepath_mod = f"{EXPERIMENT_DIR_NAME}/models/"
        create_dir(filepath_mod, SAVE_EXPERIMENT_DATA)

    history = []
    def train(dataset, epochs):
        experiment_start = time.time()
        tf_step = tf.Variable(0, dtype=float)
        step=0

        for epoch in range(epochs):
            start = time.time()

            total_loss_a = []
            reconstruction_loss_a = []
            kl_loss_a = []
            for batch in dataset:

                total_loss, reconstruction_loss, kl_loss = train_step(batch, tf_step)
                history.append([total_loss, reconstruction_loss, kl_loss])
                tf_step.assign_add(1)
                step = step + 1
                total_loss_a.append(total_loss.numpy())
                reconstruction_loss_a.append(reconstruction_loss.numpy())
                kl_loss_a.append(kl_loss.numpy())

            plot = generate_and_save_images(generator,
                                     epoch,
                                     [seed, seed_cond])

            ws_mean = calculate_ws_ch(min(epoch//5+1,5))

            if SAVE_EXPERIMENT_DATA:
                if ws_mean <= WS_MEAN_SAVE_THRESHOLD:
                    # Save the model every epoch
                    generator.compile()
                    generator.save((os.path.join(filepath_mod, "gen_"+NAME + "_"+ str(epoch) +".h5")))

            print('Done')

            wandb.log({
                'ws_mean': ws_mean,
                'total_loss': np.mean(total_loss_a),
                'reconstruction_loss': np.mean(reconstruction_loss_a),
                'kl_loss': np.mean(kl_loss_a),
                'epoch': epoch,
                'plot': wandb.Image(plot),
                'experiment_time': time.time()-experiment_start
            })
            print('done log')
            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

        return history

    def generate_and_save_images(model, epoch, test_input):

        SUPTITLE_TXT = f"\nModel: VAE proton data" \
                   f"\nPhotonsum interval: [{photon_sum_proton_min}, {photon_sum_proton_max}]" \
                   f"\nEPOCH: {epoch}"

        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        predictions = model(test_input, training=False)  # returns 16 responses

        fig, axs = plt.subplots(2, 7, figsize=(15, 5))
        fig.suptitle(SUPTITLE_TXT, x=0.1, horizontalalignment='left')

        for i in range(0, 14):
            if i < 7:
                x = x_test[20 + i].reshape(56, 30)
            else:
                x = predictions[i - 7].numpy().reshape(56, 30)
            im = axs[i // 7, i % 7].imshow(x, cmap='gnuplot')
            axs[i // 7, i % 7].axis('off')
            fig.colorbar(im, ax=axs[i // 7, i % 7])

        fig.tight_layout(rect=[0, 0, 1, 0.975])
        return fig

    history = train(dataset_with_cond, EPOCHS)
