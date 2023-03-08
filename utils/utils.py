import numpy as np


def sum_channels_parallel_old2(data):
    # for zdc proton images: shape(56, 30)
    coords = np.ogrid[0:data.shape[1], 0:28]  # grid 56 x 30
    half_x = data.shape[1] // 2  # 28
    half_y = data.shape[2] // 2  # 15

    checkerboard = (coords[0] + coords[1]) % 2 != 0
    print(checkerboard)
    checkerboard.reshape(-1, checkerboard.shape[0], checkerboard.shape[1])
    print(checkerboard)

    ch5 = (data * checkerboard).sum(axis=1).sum(axis=1)

    checkerboard = (coords[0] + coords[1]) % 2 == 0
    checkerboard = checkerboard.reshape(-1, checkerboard.shape[0], checkerboard.shape[1])

    mask = np.zeros((1, data.shape[1], data.shape[2]))
    mask[:, :half_x, :half_y] = checkerboard[:, :half_x, :half_y]
    ch1 = (data * mask).sum(axis=1).sum(axis=1)

    mask = np.zeros((1, data.shape[1], data.shape[2]))
    mask[:, :half_x, half_y:] = checkerboard[:, :half_x, half_y:]
    ch2 = (data * mask).sum(axis=1).sum(axis=1)

    mask = np.zeros((1, data.shape[1], data.shape[2]))
    mask[:, half_x:, :half_y] = checkerboard[:, half_x:, :half_y]
    ch3 = (data * mask).sum(axis=1).sum(axis=1)

    mask = np.zeros((1, data.shape[1], data.shape[2]))
    mask[:, half_x:, half_y:] = checkerboard[:, half_x:, half_y:]
    ch4 = (data * mask).sum(axis=1).sum(axis=1)

    # assert all(ch1+ch2+ch3+ch4+ch5 == data.sum(axis=1).sum(axis=1))==True

    return zip(ch1, ch2, ch3, ch4, ch5)


def sum_channels_parallel_(data):
    # Create a copy of the input array to use as the mask
    mask = np.zeros_like(data)
    mask5 = mask.copy()

    # Define the pattern of checks
    pattern = np.array([[1, 0], [0, 1]])
    pattern2 = np.array([[0, 1], [1, 0]])

    # Fill the input array with the pattern
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            mask[i, j] = pattern[i % 2, j % 2]
            mask5[i, j] = pattern2[i % 2, j % 2]

    # Divide the mask into four equal rectangles
    print(mask.shape)
    _, rows, cols = mask.shape
    mid_row, mid_col = rows // 2, cols // 2
    mask1 = mask.copy()
    mask2 = mask.copy()
    mask3 = mask.copy()
    mask4 = mask.copy()

    mask1[mid_row:, :] = 0
    mask1[:, :mid_col] = 0

    mask2[:, :mid_col] = 0
    mask2[:mid_row, :] = 0

    mask3[mid_row:, :] = 0
    mask3[:, mid_col:] = 0

    mask4[:, mid_col:] = 0
    mask4[:mid_row, :] = 0

    ch1 = (data * mask1).sum(axis=1).sum(axis=1)
    ch2 = (data * mask2).sum(axis=1).sum(axis=1)
    ch3 = (data * mask3).sum(axis=1).sum(axis=1)
    ch4 = (data * mask4).sum(axis=1).sum(axis=1)
    ch5 = (data * mask5).sum(axis=1).sum(axis=1)

    return zip(ch1, ch2, ch3, ch4, ch5)
