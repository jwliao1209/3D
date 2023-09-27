import os
import sys
import cv2 as cv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from src.sift import get_sift_correspondences
from src.direct_linear_transform import NormalizedDLT
from src.utils import remove_outlier_feature_points, plot_figure
from src.error import compute_error


if __name__ == '__main__':
    os.makedirs("outputs", exist_ok=True)
    img1 = cv.imread(sys.argv[1])
    img2 = cv.imread(sys.argv[2])
    gt_correspondences = np.load(sys.argv[3])
    points1, points2, kp1, kp2, good_matches = get_sift_correspondences(img1, img2)
    remove_outlier_index = remove_outlier_feature_points(points1, points2, threshold=0.5, kdeplot=False)

    for k in [4, 8, 20, 25]:
        for normalize in [False, True]:
            print(f"---------- k={k} ----------")
            selected_index = remove_outlier_index[:k]
            selected_good_matches = [good_matches[i] for i in selected_index]

            r, c, _ = img1.shape
            dlt = NormalizedDLT()

            if normalize:
                dlt.estimate_norm_matrix(c, r)

            dlt.estimate_homography(points1[selected_index, :], points2[selected_index, :])
            error = dlt.compute_error(gt_correspondences[0], gt_correspondences[1])

            print("Error:", error)
            plot_figure(
                img1, kp1, img2, kp2, selected_good_matches,
                os.path.join("outputs", f"k={k}_norm={normalize}.png")
            )
