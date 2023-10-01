#!/bin/bash

python 1.py --problem "1.2" \
            --image1 "images/1-0.png" \
            --image2 "images/1-2.png" \
            --gt_correspondences "groundtruth_correspondences/correspondence_02.npy" \
            --ransac_threshold 10 \
            # --display
