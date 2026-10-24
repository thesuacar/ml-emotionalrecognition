"""Webcam emotion recognition demo.

Run from the command line:   python demo_fer.py      (press q to quit)
"""
import os
import cv2
import numpy as np
import joblib
from scripts.fer_features import extract_hog, EMOTIONS      # the SAME feature code that was used for training

# ---------- settings: change these two lines if you want to use another model ----------
MODEL_FILE = "models/fer_hog_svm.pkl"
FEATURE_FUNCTION = extract_hog        # must be the same features the model was trained on
#   other options: extract_landmarks, extract_hog_landmarks (import them from fer_features)
# ----------------------------------------------------------------------------------------


def find_file(relative_path):
    """Look for a file next to this script, in the folder above it (project root), and in the current folder."""
    here = os.path.dirname(os.path.abspath(__file__))
    for folder in [here, os.path.dirname(here), os.getcwd()]:
        path = os.path.join(folder, relative_path)
        if os.path.exists(path):
            return path
    raise FileNotFoundError("Could not find " + relative_path)


# Load the trained model (a scikit-learn pipeline: scaler + classifier)
model = joblib.load(find_file(MODEL_FILE))

# Haar cascade = OpenCV's face detector. Use our own copy if we have it, otherwise the one that comes with OpenCV.
try:
    cascade_path = find_file("models/faceAndLandmarkModels/haarcascade_frontalface_default.xml")
except FileNotFoundError:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

# Start the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open the webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale (FER images are grayscale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Cut out the face and resize it to 48x48, like the FER images
        face = gray[y:y + h, x:x + w]
        face_resized = cv2.resize(face, (48, 48), interpolation=cv2.INTER_AREA)

        # Compute the features. The function expects a stack of images, so we add one extra dimension.
        features = FEATURE_FUNCTION(face_resized[np.newaxis])

        # Predict the emotion
        prediction = model.predict(features)[0]
        label = EMOTIONS[int(prediction)]

        # Draw the result
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, label, (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()