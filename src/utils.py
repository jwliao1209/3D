import cv2 as cv
import numpy as np
import seaborn as sns

from src.constants import RANDOM_SEED, EPSILON


def fix_random_seed(seed=RANDOM_SEED):
    def decorator(func):
        def wrap(*args, **kargs):
            np.random.seed(seed)
            func(*args, **kargs)
        return wrap
    return decorator


def remove_outlier_feature_points(
        points1, points2, threshold,
        kdeplot=False, save_name="output.png"
    ):
    vec12 = points1 - points2
    slope = (vec12[:, 1] + EPSILON) / (vec12[:, 0] + EPSILON)
    score = np.abs((slope-np.median(slope)) / (np.std(slope) + EPSILON))
    selected_index = np.where(score < threshold)[0]

    if kdeplot:
        sns_plot = sns.kdeplot(x=slope)
        sns_plot = sns.kdeplot(x=slope[selected_index])
        fig = sns_plot.get_figure()
        fig.savefig(save_name)

    return selected_index


def plot_figure(img1, kp1, img2, kp2, good_matches, save_name):
    img_draw_match = cv.drawMatches(
        img1, kp1, img2, kp2, good_matches, None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    # cv.imshow('match', img_draw_match)
    # cv.waitKey(0)
    cv.imwrite(save_name, img_draw_match)
    return
