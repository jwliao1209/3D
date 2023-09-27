import numpy as np


class Homography:
    def compute_one_point_sub_matrix(self, x, y, u, v):
        return np.array(
            [
                [x, y, 1, 0, 0, 0, -x*u, -y*u, -u],
                [0, 0, 0, x, y, 1, -x*v, -y*v, -v],

            ]
        )

    def estimate(self, points1, points2):
        matrix = np.vstack(
            [
                self.compute_one_point_sub_matrix(x, y, u, v)
                for (x, y), (u, v) in zip(points1, points2)
            ]
        )
        _, S, Vh = np.linalg.svd(matrix)
        self.H = Vh[-1, :].reshape(3, 3)
        self.H_inv = np.linalg.inv(self.H)
        self.H_cond = S[0] / S[-1]
        return
