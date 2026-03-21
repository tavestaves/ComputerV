import os

import cv2 

#read image 
image_path= os.path.join('.', 'assets', 'me.JPG')

img = cv2.imread(image_path)

if img is None:
    print("Error: Could not load image from", image_path)
else:
    print("Image loaded successfully, shape:", img.shape)

#write image
cv2.imwrite(os.path.join('.', 'assets', 'me_out.JPG'), img)


#visualize image 
cv2.imshow('image', img)
cv2.waitKey(0)