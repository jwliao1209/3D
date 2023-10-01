#!/bin/bash

python 1.py --image1 "images/1-0.png" \
            --image2 "images/1-1.png" \
            --gt_correspondences "groundtruth_correspondences/correspondence_01.npy"

python 1.py --image1 "images/1-0.png" \
            --image2 "images/1-2.png" \
            --gt_correspondences "groundtruth_correspondences/correspondence_02.npy"

python 2.py --image "images/citi.png" \
            --output_name "2_output.png"
