import time
import os
import wandb

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from utils import sum_channels_parallel, calculate_ws_ch_proton_model, create_dir, save_scales

print(tf.config.list_physical_devices('GPU'))
print(tf.__version__)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

# SETTINGS & PARAMETERS
SAVE_EXPERIMENT_DATA = True
WS_MEAN_SAVE_THRESHOLD = 5
BATCH_SIZE = 128
NOISE_DIM = 10
EPOCHS = 300
LR_G = 1e-4
LR_D = 1e-5

# Execute the training 3 times
for _ in range(0, 3):

    data = pd.read_pickle('/net/tscratch/people/plgpbedkowski/data/data_photonsum_proton_5_2312.pkl')
    print('Loaded: ',  data.shape, "max:", data.max())

    # Data containing particle conditional data from particle having responses with proton photon sum in interval [70, 2312] without taking into consideration photon sums of neutron responses.
    data_cond = pd.read_pickle('/net/tscratch/people/plgpbedkowski/data/data_cond_photonsum_proton_5_2312.pkl')
    print('Loaded cond: ',  data_cond.shape, "max:", data_cond.values.max(), "min:", data_cond.values.min())

    # calculate min max proton sum
    photon_sum_proton_min, photon_sum_proton_max = data_cond.proton_photon_sum.min(), data_cond.proton_photon_sum.max()
    
    DATE_STR = datetime.now().strftime("%d_%m_%Y_%H_%M")
    NAME = "gan"
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
    fake_cond = tf.data.Dataset.from_tensor_slices(y_train).shuffle(12800).batch(batch_size=BATCH_SIZE)
    dataset_with_cond = tf.data.Dataset.zip((dataset, dataset_cond, fake_cond)).shuffle(12800)

    val_dataset = tf.data.Dataset.from_tensor_slices(x_test).batch(batch_size=BATCH_SIZE)
    val_dataset_cond = tf.data.Dataset.from_tensor_slices(y_test).batch(batch_size=BATCH_SIZE)
    val_fake_cond = tf.data.Dataset.from_tensor_slices(y_test).shuffle(12800).batch(batch_size=BATCH_SIZE)
    val_dataset_with_cond = tf.data.Dataset.zip((val_dataset, val_dataset_cond, val_fake_cond)).shuffle(12800)

    from tensorflow.compat.v1.keras.layers import Input, Dense, LeakyReLU, Conv2D, MaxPooling2D, UpSampling2D,  Concatenate
    from tensorflow.compat.v1.keras.models import Model
    from tensorflow.compat.v1.keras.layers import Dense, Reshape, Flatten
    from tensorflow.compat.v1.keras.layers import Dropout,BatchNormalization
    from tensorflow import keras

    # Constant for the input dataset with conditional data
    cond_dim = 9

    ############################ generator ############################

    x = Input(shape=(NOISE_DIM,))
    cond = Input(shape=(cond_dim,))
    inputs = Concatenate(axis=1)([x, cond])

    layer_1 = Dense(128*2)(inputs)
    layer_1_bd = Dropout(0.2)(BatchNormalization()(layer_1))
    layer_1_a = LeakyReLU(alpha=0.1)(layer_1_bd)

    layer_2 = Dense(128*20*10)(layer_1_a)
    layer_2_bd = Dropout(0.2)(BatchNormalization()(layer_2))
    layer_2_a = LeakyReLU(alpha=0.1)(layer_2_bd)

    reshaped = Reshape((20,10,128))(layer_2_a)
    reshaped_s = UpSampling2D(size=(3,2))(reshaped)

    conv1 = Conv2D(256, kernel_size=(2, 2))(reshaped_s)
    conv1_bd = Dropout(0.2)(BatchNormalization()(conv1))
    conv1_a = LeakyReLU(alpha=0.1)(conv1_bd)
    conv1_a_s = UpSampling2D(size=(1,2))(conv1_a)

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

    input_img = Input(shape=[56, 30, 1],name='input_img')
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


    generator_optimizer = tf.keras.optimizers.Adam(LR_G)
    discriminator_optimizer = tf.keras.optimizers.Adam(LR_D)

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    d_acc_r = keras.metrics.BinaryAccuracy(name="d_acc_r", threshold=0.5)
    d_acc_f = keras.metrics.BinaryAccuracy(name="d_acc_r", threshold=0.5)
    g_acc = keras.metrics.BinaryAccuracy(name="g_acc_g", threshold=0.5)


    def generator_loss(fake_output):
        g_acc.update_state(tf.ones_like(fake_output), fake_output)
        return cross_entropy(tf.ones_like(fake_output), fake_output)

    # Settings for plotting
    num_examples_to_generate = 16
    START_GENERATING_IMG_FROM_IDX = 20
    # Seed to reuse for generating samples for comparison during training
    seed = tf.random.normal([num_examples_to_generate, NOISE_DIM])
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
        "Proton_min": photon_sum_proton_min,
        "Proton_max": photon_sum_proton_max,
        "Learning rate_generator": LR_G,
        "Learning rate_discriminator": LR_D,
        "Experiment_dir_name": EXPERIMENT_DIR_NAME,
        },
        tags=[f"proton_min_{photon_sum_proton_min}",
              f"proton_max_{photon_sum_proton_max}",
              "gan"]
    )


    # CALCULATE DISTRIBUTION OF CHANNELS IN ORIGINAL TEST DATA #
    org=np.exp(x_test)-1
    ch_org = np.array(org).reshape(-1, 56, 30)
    ch_org = pd.DataFrame(sum_channels_parallel(ch_org)).values
    del org


    @tf.function
    def train_step(batch, step):
        images, cond, noise_cond = batch
        step = step
        BATCH_SIZE = tf.shape(images)[0]
        noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator([noise, noise_cond], training=True)

            real_output, _ = discriminator([images, cond], training=True)
            fake_output, _ = discriminator([generated_images, noise_cond], training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        return gen_loss, disc_loss

    if SAVE_EXPERIMENT_DATA:
        filepath_mod = f"{EXPERIMENT_DIR_NAME}/models/"
        create_dir(filepath_mod, SAVE_EXPERIMENT_DATA)

    history = []
    def train(dataset, epochs):
        experiment_start = time.time()
        tf_step = tf.Variable(0, dtype=float)
        step = 0

        # generate first image
        generate_and_save_images(generator,
                                 epochs,
                                 [seed, seed_cond])

        for epoch in range(epochs):
            start = time.time()

            gen_loss_epoch = []
            disc_loss_epoch = []
            for batch in dataset:
                gen_loss, disc_loss = train_step(batch, tf_step)

                history.append([gen_loss,disc_loss,
                    100*d_acc_r.result().numpy(),
                    100*d_acc_f.result().numpy(),
                    100*g_acc.result().numpy(),
                    ])
                tf_step.assign_add(1)
                step = step+1

                gen_loss_epoch.append(gen_loss)
                disc_loss_epoch.append(disc_loss)

                if step % 100 == 0:
                    print("%d [D real acc: %.2f%%] [D fake acc: %.2f%%] [G acc: %.2f%%] "% (
                        step,
                        100*d_acc_r.result().numpy(),
                        100*d_acc_f.result().numpy(),
                        100*g_acc.result().numpy()))

            plot = generate_and_save_images(generator,
                                     epoch,
                                     [seed, seed_cond])

            ws_mean = calculate_ws_ch_proton_model(min(epoch//5+1, 5),
                                                   x_test, y_test,
                                                   generator, ch_org,
                                                   NOISE_DIM)

            if SAVE_EXPERIMENT_DATA:
                if ws_mean <= WS_MEAN_SAVE_THRESHOLD:
                    # Save the generator in every epoch
                    generator.compile()
                    generator.save((os.path.join(filepath_mod, "gen_"+NAME + "_"+ str(epoch) +".h5")))

            # Log to WandB tool
            wandb.log({
                'ws_mean': ws_mean,
                'gen_loss': np.mean(gen_loss_epoch),
                'disc_loss': np.mean(disc_loss_epoch),
                'epoch': epoch,
                'plot': wandb.Image(plot),
                'experiment_time': time.time()-experiment_start
            })

            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

        return history


    def generate_and_save_images(model, epoch, test_input):

        SUPTITLE_TXT = f"\nModel: GAN proton data" \
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
