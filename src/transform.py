import numpy as np

from copy import copy
from src.error import compute_error
from src.utils import fix_random_seed
from src.constants import EPSILON


def to_3d(points_2d):
    num, _ = points_2d.shape
    return np.hstack([points_2d, np.ones((num, 1))])


def to_2d(points_3d):
    points_3d = points_3d / (np.expand_dims(points_3d[:, 2], axis=1) + EPSILON)
    return points_3d[:, :2]


class Base3DTransform:
    def __init__(self):
        self.matrix = np.zeros((3, 3))
        self.inverse_matrix = np.zeros((3, 3))
    
    def update(self, matrix):
        self.matrix = matrix
        self.inverse_matrix = np.linalg.inv(matrix)
        return

    def transform(self, points):
        points = to_3d(points)
        return to_2d((self.matrix @ points.T).T)

    def inverse_transform(self, points):
        points = to_3d(points)
        return to_2d((self.inverse_matrix @ points.T).T)

    def get_error(self, soruce_points, target_points, mean=True):
        return compute_error(self.transform(soruce_points), target_points, mean=mean)

    def get_condition_number(self):
        _, S, _ = np.linalg.svd(self.matrix)
        return S[0] / S[-1]


class Homography(Base3DTransform):
    def compute_one_point_sub_matrix(self, x, y, u, v):
        return np.array(
            [
                [x, y, 1, 0, 0, 0, -x*u, -y*u, -u],
                [0, 0, 0, x, y, 1, -x*v, -y*v, -v],

            ]
        )

    def estimate(self, points1, points2):
        A = np.vstack(
            [
                self.compute_one_point_sub_matrix(x, y, u, v)
                for (x, y), (u, v) in zip(points1, points2)
            ]
        )
        _, S, Vh = np.linalg.svd(A)
        H = Vh[-1, :].reshape(3, 3)

        if np.linalg.matrix_rank(H) < 3 or S[-1] < EPSILON:
            raise Exception("H is singular matrix.")
        
        self.update(H)
        return


class Normalize(Base3DTransform):
    def estimate(self, points1, points2):
        mean = np.mean(points1, axis=0)
        distances = np.sqrt(np.sum((points1 - mean)**2, axis=1))
        s = 2**0.5 / np.mean(distances, axis=0)
        tx, ty = -s * mean[:2]
        T = np.array(
            [
                [s, 0, tx],
                [0, s, ty],
                [0, 0,  1],
            ]
        )

        if np.linalg.matrix_rank(T) < 3:
            raise Exception("T is singular matrix.")

        self.update(T)
        return


class DLT(Base3DTransform):
    def __init__(self):
        self.homography = Homography()

    def estimate(self, points1, points2):
        self.homography.estimate(points1, points2)
        self.update(self.homography.matrix)
        # print(self.homography.get_condition_number())
        return


class NormalizedDLT(Base3DTransform):
    def __init__(self):
        self.homography = Homography()
        self.normalize = Normalize()

    def estimate(self, points1, points2):
        points1 = to_3d(points1)
        points2 = to_3d(points2)
        self.normalize.estimate(points1, points2)

        points1_norm = to_2d((self.normalize.matrix @ points1.T).T)
        points2_norm = to_2d((self.normalize.matrix @ points2.T).T)
        self.homography.estimate(points1_norm, points2_norm)
        matrix = self.normalize.inverse_matrix @ self.homography.matrix @ self.normalize.matrix
        self.update(matrix)
        # print(self.homography.get_condition_number())
        return


class RANSAC(Base3DTransform):
    def __init__(self, point_num, iters, threshold, norm=False):
        self.point_num = point_num
        self.iters = iters
        self.threshold = threshold
        self.norm = norm

    @fix_random_seed()
    def estimate(self, points1, points2):
        best_inlier_error = np.Inf
        for _ in range(self.iters):
            dlt = NormalizedDLT() if self.norm else DLT()
            try:
                index = np.random.randint(points1.shape[0], size=(self.point_num,))
                dlt.estimate(points1[index], points2[index])
            except:  # If H is singular matrix, skip this case
                continue

            errors = dlt.get_error(points1, points2, mean=False)
            inlier_index = np.where(errors < self.threshold)[0]
            inlier_num = inlier_index.shape[0]
            inlier_errors = np.mean(errors[inlier_index]) if inlier_num > 0 else np.Inf

            if (inlier_errors < best_inlier_error) and (inlier_num > 20):
                best_inlier_error = inlier_errors
                best_index = index
                best_dlt = copy(dlt)

        self.update(best_dlt.matrix)
        self.best_index = best_index
        # print(best_dlt.homography.get_condition_number())
        return
    
    def get_good_matches(self, good_matches):
        return [good_matches[i] for i in self.best_index]
