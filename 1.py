import os
import sys
import argparse
import cv2 as cv
import numpy as np

from itertools import product

from src.constants import SAVE_DIR
from src.sift import get_sift_correspondences
from src.transform import DLT, NormalizedDLT, RANSAC
from src.utils import remove_outlier, plot_matching_figure


PARAMS = {
    "k": [4, 8, 20],
    "normalize": [False, True],
    "outlier_removal": [False, True],
    "ransac": [False, True],
}

REMOVE_OUTLIER_THRES = 0.1

def get_params_grids(params):
    keys = params.keys()
    params_grids = product(*params.values())
    return [dict(zip(keys, items)) for items in params_grids]


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, default="1.1",
                        help="Problem correspond to assignment")
    parser.add_argument("--image1", type=str, default="images/1-0.png",
                        help="Path of input image1")
    parser.add_argument("--image2", type=str, default="images/1-1.png",
                        help="Path of input image2")
    parser.add_argument("--ransac_iteration", type=int, default=5000,
                        help="Interation of RANSAC algorithm")
    parser.add_argument("--ransac_threshold", type=float, default=10,
                        help="Threshold of RANSAC inlier error")
    parser.add_argument("--gt_correspondences", type=str,
                        default="groundtruth_correspondences/correspondence_01.npy",
                        help="Path of ground truth correspondences")
    parser.add_argument("--display", action="store_true",
                        help="Display figure")
    return parser.parse_args()


if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    args = parse_arguments()
    img1 = cv.imread(args.image1)
    img2 = cv.imread(args.image2)
    gt_correspondences = np.load(args.gt_correspondences)
    params_grids = get_params_grids(PARAMS)

    for p in params_grids:
        k = p.get("k", 4)
        use_normalize = p.get("normalize", False)
        use_outlier_removal = p.get("outlier_removal", False)
        use_ransac = p.get("ransac", False)

        print(f"===== k={k}, use_normalize={use_normalize}, use_outlier_removal={use_outlier_removal}, use_ransac={use_ransac} =====")
        points1, points2, kp1, kp2, good_matches = get_sift_correspondences(img1, img2)

        if use_outlier_removal:
            points1, points2, good_matches = remove_outlier(
                points1, points2, good_matches,
                threshold=REMOVE_OUTLIER_THRES,
                kdeplot=False,
            )

        if use_ransac:
            ransac = RANSAC(k, iters=args.ransac_iteration, threshold=args.ransac_threshold, norm=use_normalize)
            ransac.estimate(points1, points2)
            error = ransac.get_error(gt_correspondences[0], gt_correspondences[1])

        else:
            dlt = NormalizedDLT() if use_normalize else DLT()
            dlt.estimate(points1[:k, :], points2[:k, :])
            error = dlt.get_error(gt_correspondences[0], gt_correspondences[1])

        plot_matching_figure(
            img1, kp1, img2, kp2, good_matches,
            save_path=os.path.join(
                SAVE_DIR,
                f"{args.problem}_k={k}_norm={use_normalize}_remove_outlier={use_outlier_removal}_ransac={use_ransac}_error={error:.4f}.png"),
            display=args.display,
        )
        print(f"Error of ground truth: {error}\n")
