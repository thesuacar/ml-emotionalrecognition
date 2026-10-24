"""Shared FER feature extraction.

Imported by BOTH the training notebooks and the real-time demo so that training and inference
compute byte-identical features. If you change HOG_PARAMS here, retrain and delete the feature cache.

Input convention for every function: array of uint8 grayscale 48x48 face images, shape (N, 48, 48).
"""
import os
import numpy as np
from skimage.feature import hog

IMG_SIZE = 48
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]   # FER labels 0..6

HOG_PARAMS = dict(orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))

# Repository root (this file lives in scripts/), so the model path works from any working folder.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LANDMARK_MODEL = os.path.join(_ROOT, "models", "faceAndLandmarkModels",
                              "shape_predictor_68_face_landmarks.dat")


def extract_hog(images):
    return np.stack([hog(im / 255.0, block_norm="L2-Hys", **HOG_PARAMS) for im in images]).astype(np.float32)


def extract_pixels(images):
    """Raw pixels in [0,1]; PCA (if used) lives inside the sklearn Pipeline."""
    return np.asarray(images).reshape(len(images), -1).astype(np.float32) / 255.0


_landm_det = None
def extract_landmarks(images):
    """dlib 68 landmarks, face box = whole 48x48 image (0,0,47,47), centred and scaled to [-1,1]."""
    global _landm_det
    import dlib
    if _landm_det is None:
        if not os.path.exists(LANDMARK_MODEL):
            raise FileNotFoundError(f"dlib landmark model not found: {LANDMARK_MODEL}")
        _landm_det = dlib.shape_predictor(LANDMARK_MODEL)
    rect = dlib.rectangle(left=0, top=0, right=47, bottom=47)
    x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
    out = []
    for img in images:
        shape = _landm_det(np.asarray(img).astype(np.uint8), rect)
        pts = np.array([[(p.x - 0.5 * (x2 - x1)) / (x2 - x1),
                         (p.y - 0.5 * (y2 - y1)) / (y2 - y1)] for p in shape.parts()], dtype=np.float32)
        pts -= pts.mean(axis=0, keepdims=True)
        pts /= np.max(np.abs(pts), axis=0, keepdims=True)
        out.append(pts.flatten())
    return np.stack(out)


# name used in the notebooks -> (cache file prefix, function)
FEATURE_FNS = {"HOG": ("hog", extract_hog),
               "Landmarks": ("landmarks", extract_landmarks),
               "PCA": ("pixels", extract_pixels)}