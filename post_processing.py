import numpy as np
import cv2


def morphological_erosion(mask):
    mask[mask >= 125] = 255         # check that mask is binarised
    mask[mask < 125] = 0            # check that mask is binarised
    mask = mask.astype(np.uint8)    # check mask is uint8
    kernel = np.ones((5, 5), np.uint8)
    mask_eroded = cv2.erode(mask, kernel, iterations=1)
    return mask_eroded


def watershed_transform(mask):
    mask[mask >= 125] = 255         # check that mask is binarised
    mask[mask < 125] = 0            # check that mask is binarised
    mask = mask.astype(np.uint8)    # check mask is uint8
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)
    return sure_fg