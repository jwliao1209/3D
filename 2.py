import os
import cv2 as cv
import numpy as np

from itertools import product

from src.transform import DLT
from src.interpolation import backward_bilinear_interpolate


if __name__ == '__main__':
    image = cv.imread(os.path.join("images", "citi.png"))
    source_points = np.array([[17, 3], [579, 2], [92, 813], [483, 821]])
    height, width = source_points[3] - source_points[0]
    target_points = np.array([[0, 0], [height, 0], [0, width], [height, width]])

    dlt = DLT()
    dlt.estimate(source_points, target_points)
    target_points = dlt.transform(source_points)

    points = np.array(list(product(range(height), range(width))))
    coor = dlt.inverse_transform(points)
    target_image = backward_bilinear_interpolate(coor, image, height, width)

    cv.imshow('match', target_image)
    cv.waitKey(0)
    cv.imwrite("target_image.png", target_image)

    
