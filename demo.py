#!/usr/bin/env python3
"""Task 6 — Fuzzy webcam demo.

In the repository root, run: python demo.py
Use the emotion_fuzzy Conda environment (scikit-learn 1.7.2).
Press q or ESC in the camera window to exit.
"""
from pathlib import Path
from time import perf_counter
import sys
import cv2
import dlib
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DETECT_WIDTH = 640  # Reduce detector work; extract landmarks at full camera resolution.
MODEL_PATH = ROOT / "models/fuzzy/fuzzy_classifier.pkl"
PREDICTOR_PATH = ROOT / "models/faceAndLandmarkModels/shape_predictor_68_face_landmarks.dat"

# Exact Task 4 feature names and class order — do not change without retraining.
FEATURES = ["mouth_width", "mouth_opening", "left_eye_opening",
            "right_eye_opening", "eyebrow_height", "smile_ratio",
            "mouth_corner_asymmetry"]
CLASSES = ["afraid", "angry", "disgust", "happy", "neutral", "sad", "surprise"]


def geometric_features(points):
    """Return the seven Task 4 geometric measurements from 68 dlib landmarks."""
    def distance(a, b):
        return float(np.linalg.norm(points[a] - points[b]))

    eps = 1e-6
    face_width = max(distance(0, 16), eps)
    mouth_width = distance(48, 54)
    mouth_height = distance(51, 57)
    return {
        "mouth_width": mouth_width / face_width,
        "mouth_opening": distance(62, 66) / face_width,
        "left_eye_opening": (distance(37, 41) + distance(38, 40)) /
                            (2 * max(distance(36, 39), eps)),
        "right_eye_opening": (distance(43, 47) + distance(44, 46)) /
                             (2 * max(distance(42, 45), eps)),
        "eyebrow_height": (distance(19, 37) + distance(24, 44)) / (2 * face_width),
        "smile_ratio": mouth_width / max(mouth_height, eps),
        "mouth_corner_asymmetry": abs(float(points[48, 1] - points[54, 1])) / face_width,
    }


# Team-generated files only: loading untrusted pickle files is unsafe.
model = joblib.load(MODEL_PATH)
if list(model["features"]) != FEATURES or list(model["classes"]) != CLASSES:
    raise ValueError("Saved fuzzy model is not compatible with Task 4 features/classes")
predictor = dlib.shape_predictor(str(PREDICTOR_PATH))
detector = dlib.get_frontal_face_detector()

def detect_face(gray):
    """Detect on a smaller image; return the largest face in original-frame coordinates."""
    height, width = gray.shape
    small_width = min(width, DETECT_WIDTH)
    small_height = round(height * small_width / width)
    small = cv2.resize(gray, (small_width, small_height), interpolation=cv2.INTER_AREA)
    faces = detector(small, 0)  # No upsampling: this is the key FPS improvement.
    if not faces:
        return None

    face = max(faces, key=lambda r: r.width() * r.height())
    sx, sy = width / small_width, height / small_height
    return dlib.rectangle(
        max(0, round(face.left() * sx)),
        max(0, round(face.top() * sy)),
        min(width - 1, round((face.right() + 1) * sx) - 1),
        min(height - 1, round((face.bottom() + 1) * sy) - 1),
    )


def predict_expression(gray, face):
    """Extract full-resolution landmarks and classify with the saved fuzzy model."""
    shape = predictor(gray, face)
    points = np.array([(p.x, p.y) for p in shape.parts()], dtype=float)
    measurements = geometric_features(points)
    feature_table = pd.DataFrame([measurements], columns=FEATURES)
    scaled = model["scaler"].transform(feature_table)
    scores = [float(np.asarray(model["models"][name].predict_label(scaled)).ravel()[0])
              for name in CLASSES]
    if not np.isfinite(scores).all():
        raise ValueError("The fuzzy classifier returned a non-finite score")
    label = CLASSES[int(np.argmax(scores))]
    return "Fear" if label == "afraid" else label.capitalize(), points


def annotate_frame(frame):
    """Return an annotated frame, whether a face was found, and timing details."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    start = perf_counter()
    face = detect_face(gray)
    detect_ms = (perf_counter() - start) * 1000
    predict_ms = None
    label = None
    if face is not None:
        start = perf_counter()
        label, points = predict_expression(gray, face)
        predict_ms = (perf_counter() - start) * 1000
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()),
                      (0, 220, 220), 2)
        cv2.putText(frame, label, (face.left(), max(25, face.top() - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 220), 2)
    return frame, label, detect_ms, predict_ms

def open_camera():
    # AVFoundation works on macOS; CAP_ANY is a fallback for other OpenCV builds.
    backends = (cv2.CAP_AVFOUNDATION, cv2.CAP_ANY) if sys.platform == "darwin" else (cv2.CAP_ANY,)
    for backend in backends:
        cap = cv2.VideoCapture(0, backend)
        if cap.isOpened():
            return cap
        cap.release()
    raise RuntimeError("Cannot open camera 0; check macOS camera permissions")



def run_webcam():
    """Live inference with basic FPS measurements; q/ESC exits without killing Jupyter."""
    cap = open_camera()
    window = "Task 6 - Fuzzy webcam (q/ESC to stop)"
    frames = found = 0
    detector_times = []
    predict_times = []
    started = perf_counter()
    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        try:
            cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 1)
        except cv2.error:
            pass  # This property is not available in every macOS OpenCV build.
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames += 1
            frame, label, detect_ms, predict_ms = annotate_frame(frame)
            detector_times.append(detect_ms)
            if label is not None:
                found += 1
                predict_times.append(predict_ms)
            fps = frames / max(perf_counter() - started, 1e-9)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
            cv2.imshow(window, frame)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), ord("Q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        for _ in range(20):
            cv2.waitKey(1)  # macOS: finish processing window-close events.
    seconds = max(perf_counter() - started, 1e-9)
    result = {"frames": frames, "frames_with_faces": found,
              "face_detection_rate": round(found / frames, 3) if frames else 0,
              "processed_fps": round(frames / seconds, 2),
              "mean_detection_ms": round(float(np.mean(detector_times)), 2) if detector_times else None,
              "mean_prediction_ms": round(float(np.mean(predict_times)), 2) if predict_times else None}
    print(pd.Series(result).to_string())
    return result

if __name__ == "__main__":
    print("Model:", MODEL_PATH)
    print("Detector width:", DETECT_WIDTH)
    print("Press q or ESC in the webcam window to exit")
    run_webcam()
