import numpy as np


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
