import numpy as np

from src.constants import EPSILON


def backward_bilinear_interpolate(coordinates, image, row, col):
    lu_coor = np.floor(coordinates).astype(int)
    ld_coor = lu_coor + np.array([1, 0])
    ru_coor = lu_coor + np.array([0, 1])
    rd_coor = lu_coor + np.array([1, 1])

    r, c = coordinates[:, 0], coordinates[:, 1]
    r1, c1 = lu_coor[:, 0], lu_coor[:, 1]
    r2, c1 = ld_coor[:, 0], ld_coor[:, 1]
    r1, c2 = ru_coor[:, 0], ru_coor[:, 1]
    r2, c2 = rd_coor[:, 0], rd_coor[:, 1]

    denominator = (r2-r1) * (c1-c2) + EPSILON
    w11 = np.expand_dims(((r2-r) * (c-c2) + EPSILON) / denominator, axis=1)
    w12 = np.expand_dims(((r2-r) * (c1-c) + EPSILON) / denominator, axis=1)
    w21 = np.expand_dims(((r-r1) * (c-c2) + EPSILON) / denominator, axis=1)
    w22 = np.expand_dims(((r-r1) * (c1-c) + EPSILON) / denominator, axis=1)

    lu_pixel = image[r1, c1, :]
    ld_pixel = image[r2, c1, :]
    ru_pixel = image[r1, c2, :]
    rd_pixel = image[r2, c2, :]

    target_image = w11 * lu_pixel + w21 * ld_pixel +  w12 * ru_pixel + w22 * rd_pixel

    return target_image.reshape(row, col, 3).astype(np.uint8)
