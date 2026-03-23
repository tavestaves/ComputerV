#cropping

import os
import cv2

img = cv2.imread(os.path.join('.', 'assets', 'me.jpg'))

#not a meaningful crop but it worked

cropped_img = img[50:250, 100:300]

cv2.imshow('img', img)
cv2.imshow('cropped_img', cropped_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
