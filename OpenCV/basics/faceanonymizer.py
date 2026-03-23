
import cv2

face_cascade = cv2.CascadeClassifier('models/facess.xml')

img = cv2.imread('assets/me.jpg')

cv2.imshow('img', img)

#detect faces

#blur faces

#save image