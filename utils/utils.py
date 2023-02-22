import numpy as np


def sum_channels_parallel_(data):
    coords = np.ogrid[0:data.shape[1], 0:data.shape[2]]
    half_x = data.shape[1] // 2
    half_y = data.shape[2] // 2

    checkerboard = (coords[0] + coords[1]) % 2 != 0
    checkerboard.reshape(-1, checkerboard.shape[0], checkerboard.shape[1])

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
