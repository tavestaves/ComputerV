import cv2
import os
import numpy as np

img = cv2.imread(os.path.join('.', 'assets', 'basketballplayer.png'))

img_edge =cv2.Canny(img, 100, 200)

img_gilated = cv2.dilate(img_edge, np.ones((3,3), dtype=np.uint8))
img_eroded = cv2.erode(img_gilated, np.ones((3,3), dtype=np.uint8))


cv2.imshow('Img', img)
cv2.imshow('img_edge', img_edge)
cv2.imshow('img_gilated', img_gilated)
cv2.imshow('img_eroded', img_eroded)
cv2.waitKey(0)