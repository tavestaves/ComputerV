import os
import cv2

#all image sare in BGR color space by default


img = cv2.imread(os.path.join('.', 'assets', 'bird.png'))

image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
image_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

cv2.imshow('Img', img)
cv2.imshow('Image RGB', image_rgb)
cv2.imshow('Image Gray', image_gray)
cv2.imshow('Image HSV', image_hsv)
cv2.waitKey(0)
