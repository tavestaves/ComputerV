# Color Detection Script
# This script captures video from the webcam, detects yellow objects,
# and draws a bounding box around them in real-time.

from cmath import pi
import cv2
from utils import get_limits
from PIL import Image



# Define the color to detect in BGR format
yellow = [255, 255, 0] # BGR Format

# Initialize video capture from the default webcam (index 0)
cap = cv2.VideoCapture(0)

# Main loop to process video frames
while True:
    # Capture a frame from the webcam
    ret, frames = cap.read()


    # Convert the frame from BGR to HSV color space for better color detection
    hsv_img = cv2.cvtColor(frames, cv2.COLOR_BGR2HSV)

    # Get the lower and upper HSV limits for the yellow color
    lower_limit, upper_limit = get_limits(yellow)

    # Create a binary mask where pixels within the color range are white, others black
    mask = cv2.inRange(hsv_img, lower_limit, upper_limit)
    
    # Convert mask to PIL Image to find bounding box
    mask_ = Image.fromarray(mask)
    bbox = mask_.getbbox()

    print(bbox)

    # If a bounding box is found, draw a green rectangle around the detected object
    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(frames, (x, y), (w, h), (0, 255, 0), 2)

    # Display the original frame and the mask in separate windows
    cv2.imshow('Webcam', frames)
    cv2.imshow('Mask', mask)


    # Exit the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()