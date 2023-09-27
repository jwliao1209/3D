import numpy as np


def to_3d(points_2d):
    num, _ = points_2d.shape
    return np.hstack([points_2d, np.ones((num, 1))])


def to_2d(points_3d):
    points_3d = points_3d / np.expand_dims(points_3d[:, 2], axis=1)
    return points_3d[:, :2]
