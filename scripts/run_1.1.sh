#!/bin/bash

python 1.py --problem "1.1" \
            --image1 "images/1-0.png" \
            --image2 "images/1-1.png" \
            --gt_correspondences "groundtruth_correspondences/correspondence_01.npy" \
            --ransac_threshold 0.05 \
            --display
