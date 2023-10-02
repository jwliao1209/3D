import os
import time
import argparse
import cv2 as cv
import numpy as np

from src.constants import SAVE_DIR
from src.transform import DLT
from src.interpolation import backward_bilinear_interpolate
from src.utils import generate_grid_points


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="images/citi.png",
                        help="Path of input image")
    parser.add_argument("--output_name", type=str, default="2_output.png",
                        help="Output image name")
    parser.add_argument("--display", action="store_true",
                        help="Display figure")
    return parser.parse_args()


if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    args = parse_arguments()

    image = cv.imread(args.image)
    source_points = np.array([[17, 3], [579, 2], [92, 813], [483, 821]])
    height, width = source_points[3] - source_points[0]
    target_points = np.array([[0, 0], [height, 0], [0, width], [height, width]])

    start_time = time.time()
    dlt = DLT()
    dlt.estimate(source_points, target_points)
    target_points = dlt.transform(source_points)
    target_coordinates = generate_grid_points(height, width)
    source_coordinates = dlt.inverse_transform(target_coordinates)
    target_image = backward_bilinear_interpolate(source_coordinates, image, height, width)
    end_time = time.time()
    print(f"interpolation spending time: {end_time - start_time}")
    cv.imwrite(os.path.join(SAVE_DIR, args.output_name), target_image)

    if args.display:
        cv.imshow("backward_interpolation", target_image)
        cv.waitKey(0)
