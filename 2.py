import os
import cv2 as cv
import numpy as np

from itertools import product

from src.direct_linear_transform import DLT
from src.interpolation import backward_bilinear_interpolate


image = cv.imread(os.path.join("images", "1-0.png"))
source_points = np.array([[403, 163], [266, 554], [1126, 175], [1237, 573]])
target_points = np.array([[0, 0], [0, 1000], [1000, 0], [1000, 1000]])

dlt = DLT()
dlt.estimate_homography(source_points, target_points)
target_points = dlt.transform(source_points)

points = np.array(list(product(range(1000), range(1000))))
coor = dlt.inverse_transform(points)
target_image = backward_bilinear_interpolate(coor, image, 1000, 1000)

# cv.imshow('match', target_image)
# cv.waitKey(0)
