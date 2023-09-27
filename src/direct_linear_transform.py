import numpy as np

from src.coordinate_transform import to_3d, to_2d
from src.homography import Homography
from src.error import compute_error


class DLT(Homography):
    def __init__(self):
        self.H = np.eye(3, 3)

    def estimate_homography(self, point1, point2):
        super().estimate(point1, point2)
        return

    def transform(self, points):
        points = to_3d(points)
        return to_2d((self.H @ points.T).T)

    def inverse_transform(self, points):
        points = to_3d(points)
        return to_2d((self.H_inv @ points.T).T)
    
    def compute_error(self, soruce_points, target_points):
        return compute_error(self.transform(soruce_points), target_points)


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
        return

    def estimate_homography(self, point1, point2):
        point1 = to_3d(point1)
        point2 = to_3d(point2)
        point1_norm = to_2d((self.T @ point1.T).T)
        point2_norm = to_2d((self.T @ point2.T).T)
        super().estimate_homography(point1_norm, point2_norm)
        self.H = self.T_inv @ self.H @ self.T
        return


# class RANSAC(DLT):
#     def __init__(self):
#         pass

#     def estimate_homography(self, point1, point2, iters):
#         num, _ = point1.shape
#         for i in range(iters):
#             selected_index = np.random.randint(num)
#             super().estimate_homography(self, point1, point2, cond=False)
