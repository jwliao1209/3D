import cv2 as cv
import numpy as np

from itertools import product
from src.constants import RANDOM_SEED, EPSILON


def fix_random_seed(seed=RANDOM_SEED):
    def decorator(func):
        def wrap(*args, **kargs):
            np.random.seed(seed)
            func(*args, **kargs)
        return wrap
    return decorator


def get_params_grids(params):
    keys = params.keys()
    params_grids = product(*params.values())
    return [dict(zip(keys, items)) for items in params_grids]


def remove_outlier(points1, points2, good_matches, threshold=1, kdeplot=False, save_path="output.png"):
    vec12 = points1 - points2
    slope = (vec12[:, 1] + EPSILON) / (vec12[:, 0] + EPSILON)
    score = np.abs((slope-np.median(slope)) / (np.std(slope) + EPSILON))

    selected_indices = np.where(score < threshold)[0]
    selected_points1 = points1[selected_indices]
    selected_points2 = points2[selected_indices]
    selected_good_matches = [good_matches[i] for i in selected_indices]

    if kdeplot:
        import seaborn as sns
        import matplotlib.pyplot as plt

        sns.histplot(x=slope, label="original slope")
        sns.histplot(x=slope[selected_indices], label="removing outlier slope")
        plt.legend()
        plt.xlim(-10, 10)
        plt.savefig(save_path)
        plt.close()

    return selected_points1, selected_points2, selected_good_matches


def plot_matching_figure(img1, kp1, img2, kp2, good_matches, save_path=None, display=False):
    img_draw_match = cv.drawMatches(
        img1, kp1, img2, kp2, good_matches, None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    if save_path is not None:
        cv.imwrite(save_path, img_draw_match)
    if display:
        cv.imshow('match', img_draw_match)
        cv.waitKey(0)
    return


def generate_grid_points(height, width):
    r, c = np.meshgrid(range(height), range(width))
    return np.array([r.T.flatten(), c.T.flatten()]).T
