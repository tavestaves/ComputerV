import os
import cv2



img = cv2.imread(os.path.join('.', 'assets', 'freelancer.png'))
noisy_img = cv2.imread(os.path.join('.', 'assets', 'noisy.png'))

k_size = 9
k_size_denoise = 13

img_blur = cv2.blur(img , (k_size, k_size))
img_gaussian = cv2.GaussianBlur(img, (k_size, k_size), 5)
img_median = cv2.medianBlur(img, k_size)


denoisy_img = cv2.medianBlur(noisy_img, k_size_denoise)


cv2.imshow('Img', img)
cv2.imshow('Img Blur', img_blur)
cv2.imshow('Img Gaussian', img_gaussian)
cv2.imshow('Img Median', img_median)

cv2.imshow('Noisy Img', noisy_img)
cv2.imshow('Denoisy Img', denoisy_img)

cv2.waitKey(0)
