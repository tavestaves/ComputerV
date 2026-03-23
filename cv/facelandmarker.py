import cv2
import mediapipe as mp
import time
import numpy as np

# MediaPipe Tasks API
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------- Aliases ----------------
BaseOptions = python.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

# ---------------- Load Model ----------------
model_path = "face_landmarker.task"  # Make sure this file exists

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1
)

landmarker = FaceLandmarker.create_from_options(options)

# ---------------- Video Input ----------------
# Use 0 for webcam OR '01.mp4' for video file
cap = cv2.VideoCapture(0)

# Get FPS safely
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30  # fallback for webcam or unknown video

delay = int(1000 / fps)
timestamp = 0

# ---------------- Main Loop ----------------
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Resize for performance
    frame = cv2.resize(frame, (640, 480))

    # Convert to RGB (MediaPipe requirement)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to MediaPipe Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    # Run detection
    result = landmarker.detect_for_video(mp_image, timestamp)

    h, w = frame.shape[:2]

    # Draw landmarks and blur face
    if result.face_landmarks:
        for face_landmarks in result.face_landmarks:
            # Get bounding box from landmarks
            x_coords = [int(lm.x * w) for lm in face_landmarks]
            y_coords = [int(lm.y * h) for lm in face_landmarks]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            # Add some padding
            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)
            
            # Draw rectangle
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Blur the face region
            face_roi = frame[y_min:y_max, x_min:x_max]
            blurred_face = cv2.GaussianBlur(face_roi, (23, 23), 30)
            frame[y_min:y_max, x_min:x_max] = blurred_face

    # Show output
    cv2.imshow("Face Landmark Detection (MediaPipe Tasks)", frame)

    # Update timestamp
    timestamp += delay

    # Exit on ESC
    if cv2.waitKey(delay) & 0xFF == 27:
        break

# ---------------- Cleanup ----------------
cap.release()
landmarker.close()
cv2.destroyAllWindows()