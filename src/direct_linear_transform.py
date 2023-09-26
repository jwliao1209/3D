import numpy as np
from src.coordinate_transform import CoordinateTransform


def compute_one_point_sub_matrix(x, y, u, v):
    return np.array(
        [
            [x, y, 1, 0, 0, 0, -x*u, -y*u, -u],
            [0, 0, 0, x, y, 1, -x*v, -y*v, -v],

        ]
    )


class DLT:
    def __init__(self):
        self.H = np.eye(3, 3)

    def estimate_homography(self, point1, point2):
        matrix = np.vstack(
            [
                compute_one_point_sub_matrix(x, y, u, v)
                for (x, y), (u, v) in zip(point1, point2)
            ]
        )
        _, _, Vh = np.linalg.svd(matrix)
        self.H = Vh[-1, :].reshape(3, 3)
        self.H_inv = np.linalg.inv(self.H)

    def transform(self, points):
        ct = CoordinateTransform()
        points = ct.to_3d(points)
        return ct.to_2d((self.H @ points.T).T)

    def inverse_transform(self, points):
        ct = CoordinateTransform()
        points = ct.to_3d(points)
        return ct.to_2d((self.H_inv @ points.T).T)


class NormalizedDLT(DLT):
    def __init__(self):
        self.T = np.eye(3, 3)
        self.T_inv = np.eye(3, 3)

    def estimate_norm_matrix(self, row, col):
        self.T = np.array(
            [
                [2 / row, 0,       -1],
                [0,       2 / col, -1],
                [0,       0,        1],
            ]
        )
        self.T_inv = np.linalg.inv(self.T)

    def estimate_homography(self, point1, point2):
        ct = CoordinateTransform()
        point1 = ct.to_3d(point1)
        point2 = ct.to_3d(point2)
        point1_norm = ct.to_2d((self.T @ point1.T).T)
        point2_norm = ct.to_2d((self.T @ point2.T).T)
        super().estimate_homography(point1_norm, point2_norm)
        self.H = self.T_inv @ self.H @ self.T
