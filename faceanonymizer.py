import cv2
import os 
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Initialize the face detector
base_options = python.BaseOptions()
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

#read image 
img = cv2.imread(os.path.join('.', 'assets', 'face.png'))

#detect faces
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
detection_result = detector.detect(mp_image)

print(detection_result.detections)

#blur faces 
if detection_result.detections:
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
        # Blur the face region
        face = img[y:y+h, x:x+w]
        face = cv2.GaussianBlur(face, (99, 99), 30)
        img[y:y+h, x:x+w] = face

#save image 
cv2.imwrite(os.path.join('.', 'assets', 'face_blurred.png'), img)

cv2.imshow('img', img)
cv2.waitKey(0)