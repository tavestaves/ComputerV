import cv2
import os

img = cv2.imread(os.path.join('.', 'assets', 'bbirds.png'))

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret , thresh = cv2.threshold(img_gray, 120, 255, cv2.THRESH_BINARY_INV)

contours , hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    print(cv2.contourArea(cnt))
    if cv2.contourArea(cnt) > 2000:
        #cv2.drawContours(img, cnt, -1, (0, 255, 0), 3) 

        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)

cv2.imshow('img', img)
#cv2.imshow('img_gray', img_gray)
cv2.imshow('thresh', thresh)

cv2.waitKey(0)