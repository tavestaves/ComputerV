import os
import cv2

img = cv2.imread(os.path.join('.', 'assets', 'whiteboard.png'))
print(img.shape)

#line
cv2.line(img, (50, 100), (400, 100), (255, 0, 0), 5)

#rectangle
cv2.rectangle(img, (50, 150), (400, 300), (0, 255, 0), -1)

#circle
cv2.circle(img, (300, 400), 75, (0, 0, 255), 5)

#text
cv2.putText(img, 'Taves', (70, 500), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 150, 0), 5)

cv2.imshow('img', img)
cv2.waitKey(0)
