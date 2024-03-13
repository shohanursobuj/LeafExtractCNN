import os
import numpy as np
from skimage.feature import hog, local_binary_pattern
from skimage import exposure, util
import cv2

def extract_hog_features(image, save_path=None):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate HOG features
    features, hog_image = hog(gray_image, orientations=9, pixels_per_cell=(14, 14),
                              cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')

    # Rescale histogram for better visualization
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    # Convert HOG image to unsigned byte format
    hog_image_uint8 = util.img_as_ubyte(hog_image_rescaled)

    # Save the HOG image if save_path is provided
    if save_path is not None:
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, hog_image_uint8)

    return features, hog_image_rescaled

def extract_lbp_features(image, radius=3, points=24, save_path=None):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate LBP features
    lbp_image = local_binary_pattern(gray_image, points, radius, method='uniform')

    # Rescale LBP image to range [0, 255]
    lbp_image_rescaled = exposure.rescale_intensity(lbp_image, in_range=(0, np.max(lbp_image)), out_range=(0, 255))

    # Convert LBP image to unsigned byte format
    lbp_image_uint8 = lbp_image_rescaled.astype(np.uint8)

    # Save the LBP image if save_path is provided
    if save_path is not None:
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, lbp_image_uint8)

    return lbp_image_uint8