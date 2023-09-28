import os
import sys
import cv2 as cv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from src.sift import get_sift_correspondences
from src.transform import DLT, NormalizedDLT, RANSAC
from src.utils import remove_outlier_feature_points, plot_figure


if __name__ == '__main__':
    os.makedirs("outputs", exist_ok=True)
    img1 = cv.imread(sys.argv[1])
    img2 = cv.imread(sys.argv[2])
    gt_correspondences = np.load(sys.argv[3])
    points1, points2, kp1, kp2, good_matches = get_sift_correspondences(img1, img2)
    remove_outlier_index = remove_outlier_feature_points(points1, points2, threshold=0.5, kdeplot=False)

    for k in [4, 8, 20]:
        for use_normalize in [False, True]:
            for use_ransac in [False, True]:
                
                selected_index = remove_outlier_index[:k]
                selected_good_matches = [good_matches[i] for i in selected_index]

                if use_ransac:
                    print(f"==========  {k} points DLT + RANSEC ==========")
                    ransac = RANSAC(k, 5000, 5, "dlt")
                    ransac.estimate(points1, points2)
                    error = ransac.get_error(gt_correspondences[0], gt_correspondences[1])

                else:
                    if use_normalize:
                        print(f"========== {k} points Normalized DLT ==========")
                        dlt = NormalizedDLT()
                    else:
                        print(f"========== {k} points DLT ==========")
                        dlt = DLT()

                    dlt.estimate(points1[selected_index, :], points2[selected_index, :])
                    error = dlt.get_error(gt_correspondences[0], gt_correspondences[1])

                print(f"Error of ground truth: {error}\n")
                    # plot_figure(
                    #     img1, kp1, img2, kp2, selected_good_matches,
                    #     os.path.join("outputs", f"k={k}_norm={normalize}.png")
                    # )
