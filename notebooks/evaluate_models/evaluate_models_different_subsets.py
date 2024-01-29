import wandb
from sklearn.model_selection import train_test_split
import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))
import os
import numpy as np
import pandas as pd
from utils import sum_channels_parallel
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

print(tf.__version__)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"


data = pd.read_pickle('/net/tscratch/people/plgpbedkowski/data/data_proton_photonsum_proton_1_2312.pkl')
print('Loaded: ', data.shape, "max:", data.max())

# Data containing particle conditional data from particle having responses with proton photon sum in interval [70, 2312] without taking into consideration photon sums of neutron responses.
data_cond = pd.read_pickle('/net/tscratch/people/plgpbedkowski/data/data_cond_photonsum_proton_1_2312.pkl')
print('Loaded cond: ', data_cond.shape, "max:", data_cond.values.max(), "min:", data_cond.values.min())

proton_photon_sum = data_cond['proton_photon_sum']
std_proton = data_cond['std_proton']

data_cond.drop(columns=['proton_photon_sum', 'std_proton'], inplace=True)

data = np.log(data+1)
data = np.float32(data)
print("data max", data.max(), "min", data.min())

scaler = StandardScaler()
data_cond = np.float32(data_cond)
data_cond = scaler.fit_transform(data_cond)
print("cond max", data_cond.max(), "min", data_cond.min())

_, x_test, _, y_test, _, std_test, = train_test_split(data, data_cond, std_proton, test_size=0.2, shuffle=False)
print(x_test.shape, y_test.shape, std_test.shape)

np.exp(x_test)-1
ch_org = np.array(org).reshape(-1, 56, 30)
ch_org = pd.DataFrame(sum_channels_parallel(ch_org)).values

del org

# Load best models
vae = tf.keras.models.load_model("/net/people/plgrid/plgpbedkowski/models/gen_vae_247.h5", compile=False)
gan = tf.keras.models.load_model("/net/people/plgrid/plgpbedkowski/models/gen_gan_169.h5", compile=False)
sdigan = tf.keras.models.load_model("/net/people/plgrid/plgpbedkowski/models/gen_sdi-gan_267.h5", compile=False)
sdigan_reg = tf.keras.models.load_model("/net/people/plgrid/plgpbedkowski/models/gen_sin-gan_96.h5", compile=False)
sdigan_reg_aux = tf.keras.models.load_model("/net/people/plgrid/plgpbedkowski/models/gen_sin-gan-aux-reg-arch-2_215.h5", compile=False)


def get_number_of_empty_responses(model, noise_dim = 10):
    list_n_empty_responses = []
    batch_size = 200
    n_batches = y_test.shape[0] // batch_size

    for i in range(n_batches):
        start_idx = i*batch_size
        end_idx = start_idx + batch_size
        # for _ in range(n_executions):
        seed = tf.random.normal([batch_size, noise_dim])
        input_data = [seed, y_test[start_idx:end_idx]]

        outputs = model(input_data, training=False)
        pixel_sum = tf.reduce_sum(outputs[:,:,:,0], axis=(1, 2))
        indices_of_empty_responses = tf.where(tf.equal(pixel_sum, 0))
        n_empty_responses = len(indices_of_empty_responses.numpy())
        list_n_empty_responses.append(n_empty_responses)
        # return sum(n_empty_responses)/n_executions
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

print('====== Empty Responses ======')
n_empty_responses_vae = get_number_of_empty_responses(vae)
print(f"VAE: {n_empty_responses_vae}")

n_empty_responses_gan = get_number_of_empty_responses(gan)
print(f"GAN: {n_empty_responses_gan}")

n_empty_responses_sdigan = get_number_of_empty_responses(sdigan)
print(f"SDI-GAN: {n_empty_responses_sdigan}")

n_empty_responses_sdigan_reg = get_number_of_empty_responses(sdigan_reg)
print(f"SDI-GAN + reg: {n_empty_responses_sdigan_reg}")

n_empty_responses_sdigan_reg_aux = get_number_of_empty_responses(sdigan_reg_aux)
print(f"SDI-GAN + reg + aux: {n_empty_responses_sdigan_reg_aux}")

print('====== Diversity Error ======')
n_empty_responses_vae = model_diversity_error(vae, y_test, std_test, n_calc=10)
print(f"VAE: {n_empty_responses_vae}")

n_empty_responses_gan = model_diversity_error(gan, y_test, std_test, n_calc=10)
print(f"GAN: {n_empty_responses_gan}")

n_empty_responses_sdigan = model_diversity_error(sdigan, y_test, std_test, n_calc=10)
print(f"SDI-GAN: {n_empty_responses_sdigan}")

n_empty_responses_sdigan_reg = model_diversity_error(sdigan_reg, y_test, std_test, n_calc=10)
print(f"SDI-GAN + reg: {n_empty_responses_sdigan_reg}")

n_empty_responses_sdigan_reg_aux = model_diversity_error(sdigan_reg_aux, y_test, std_test, n_calc=10)
print(f"SDI-GAN + reg + aux: {n_empty_responses_sdigan_reg_aux}")
