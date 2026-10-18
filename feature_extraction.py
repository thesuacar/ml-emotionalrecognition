import numpy as np
from skimage.feature import hog

# Feature hyperparameters (keep these in the report)
HOG_PARAMS     = dict(orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))

def extract_hog(images):
    return np.stack([hog(im / 255.0, block_norm="L2-Hys", **HOG_PARAMS) for im in images]).astype(np.float32)

def extract_hog_kdef(images):
    return np.stack([hog(im, block_norm="L2-Hys", **HOG_PARAMS) for im in images]).astype(np.float32)

