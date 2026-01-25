import cv2 
from utils import get_limits
from PIL import Image



pink = [255, 0, 255] #pink in BGR

cap = cv2.VideoCapture(0)

while True:
    ret, frames = cap.read()


    hsv_img = cv2.cvtColor(frames, cv2.COLOR_BGR2HSV)

    lower_limit, upper_limit = get_limits(pink)

    mask = cv2.inRange(hsv_img, lower_limit, upper_limit)
    
    mask_ = Image.fromarray(mask)
    bbox = mask_.getbbox()

    print(bbox)

    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(frames, (x, y), (w, h), (0, 255, 0), 2)

    cv2.imshow('Webcam', frames)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()