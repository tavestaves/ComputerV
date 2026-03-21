
import os

import cv2

face_cascade = cv2.CascadeClassifier('models/facess.xml')
if face_cascade.empty():
    print("Error: Cascade file not loaded")
    exit()

image_path= os.path.join('.', 'assets/me.jpg')
img = cv2.imread(image_path)
if img is None:
    print("Error: Image not found")
    exit()

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)

for (x, y, w, h) in faces:
    face = img[y:y+h, x:x+w]
    blurred_face = cv2.GaussianBlur(face, (99, 99), 30)
    rectangleed_img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    img[y:y+h, x:x+w] = blurred_face

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()



#detect faces

#blur faces

#save image