import os
import numpy as np
from scipy.stats import wasserstein_distance
import pandas as pd


def get_channel_masks(input_array: np.ndarray):
    """
    Returns masks of 5 for input array.

    input_array: Array of shape(N, M)
    """

    # Create a copy of the input array to use as the mask
    mask = np.ones_like(input_array)
    n, m = input_array.shape

    # Define the pattern of checks
    pattern = np.array([[0, 1], [1, 0]])

    # Fill the input array with the pattern
    for i in range(n):
        for j in range(m):
            mask[i, j] = pattern[i % 2, j % 2]

    mask5 = np.ones_like(input_array) - mask

    # Divide the mask into four equal rectangles
    rows, cols = mask.shape
    mid_row, mid_col = rows // 2, cols // 2

    mask1 = mask.copy()
    mask2 = mask.copy()
    mask3 = mask.copy()
    mask4 = mask.copy()

    mask4[mid_row:, :] = 0
    mask4[:, :mid_col] = 0

    mask2[:, :mid_col] = 0
    mask2[:mid_row, :] = 0

    mask3[mid_row:, :] = 0
    mask3[:, mid_col:] = 0

    mask1[:, mid_col:] = 0
    mask1[:mid_row, :] = 0

    return mask1, mask2, mask3, mask4, mask5


def sum_channels_parallel(data: np.ndarray):
    """
    Calculates the sum of 5 channels of input images. Each Input image is divided into 5 sections.

    data: Array of shape(x, N, M)
        Array of x images of the same size.
    """
    mask1, mask2, mask3, mask4, mask5 = get_channel_masks(data[0])

    ch1 = (data * mask1).sum(axis=1).sum(axis=1)
    ch2 = (data * mask2).sum(axis=1).sum(axis=1)
    ch3 = (data * mask3).sum(axis=1).sum(axis=1)
    ch4 = (data * mask4).sum(axis=1).sum(axis=1)
    ch5 = (data * mask5).sum(axis=1).sum(axis=1)

    return zip(ch1, ch2, ch3, ch4, ch5)


def get_max_value_image_coordinates(img):
    """

    :param img: Input iamge of any shape
    :return: Tuple with (X, Y) coordinates
    """
    return np.unravel_index(np.argmax(img), img.shape)


def create_dir(path, SAVE_EXPERIMENT_DATA):
    if SAVE_EXPERIMENT_DATA:
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)


def save_scales(model_name, scaler_means, scaler_scales, filepath):
    out_fnm = f"{model_name}_scales.txt"
    res = "#means"
    for mean_ in scaler_means:
        res += "\n" + str(mean_)
    res += "\n\n#scales"
    for scale_ in scaler_scales:
        res += "\n" + str(scale_)

    with open(filepath+out_fnm, mode="w") as f:
        f.write(res)


def calculate_ws_ch_proton_model(n_calc, x_test, y_test, generator, ch_org, noie_dim):
    ws = [0, 0, 0, 0, 0]
    for j in range(n_calc):  # perform few calculations of the ws distance
        z = np.random.normal(0, 1,
                             (x_test.shape[0], noie_dim))
        z_c = y_test
        results = generator.predict([z,z_c])
        results = np.exp(results)-1
        try:
            ch_gen = np.array(results).reshape(-1, 56, 30)
            ch_gen = pd.DataFrame(sum_channels_parallel(ch_gen)).values
            for i in range(5):
                ws[i] = ws[i] + wasserstein_distance(ch_org[:,i], ch_gen[:,i])
            ws = np.array(ws)
        except ValueError as e:
            print(e)

    ws = ws/n_calc
    ws_mean = ws.sum()/5
    print("ws mean", f'{ws_mean:.2f}', end=" ")
    for n, score in enumerate(ws):
        print("ch"+str(n+1), f'{score:.2f}',end=" ")
    return ws_mean
