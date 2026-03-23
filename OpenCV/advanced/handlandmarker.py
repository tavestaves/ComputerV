import cv2
import mediapipe as mp
import time
import numpy as np

# MediaPipe Tasks API
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------- Aliases ----------------
BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

# Hand landmark connections for drawing (no mp.solutions in Tasks API)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

# ---------------- Load Model ----------------
model_path = "hand_landmarker.task"

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

landmarker = HandLandmarker.create_from_options(options)

# ---------------- Video Input ----------------
cap = cv2.VideoCapture(0)

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30

delay = int(1000 / fps)
timestamp = 0

# ---------------- Main Loop ----------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect_for_video(mp_image, timestamp)

    h, w = frame.shape[:2]

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            for lm in hand_landmarks:
                x = min(int(lm.x * w), w - 1)
                y = min(int(lm.y * h), h - 1)
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

            for conn in HAND_CONNECTIONS:
                start_idx, end_idx = conn
                start = hand_landmarks[start_idx]
                end = hand_landmarks[end_idx]

                x1 = min(int(start.x * w), w - 1)
                y1 = min(int(start.y * h), h - 1)
                x2 = min(int(end.x * w), w - 1)
                y2 = min(int(end.y * h), h - 1)

                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

    cv2.imshow("Hand Landmark Detection (MediaPipe Tasks)", frame)

    timestamp += delay

    if cv2.waitKey(delay) & 0xFF == 27:
        break

cap.release()
landmarker.close()
cv2.destroyAllWindows()