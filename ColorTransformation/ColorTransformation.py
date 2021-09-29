import cv2
import numpy as np
import matplotlib.pyplot as plt

# Color Mappings

fox = 'fox.jpg'
wall = 'wall.jpg'

read_img = cv2.imread(fox)
gray_img = cv2.cvtColor(read_img, cv2.COLOR_RGB2GRAY)
cv2.imwrite('gray_img.jpg', gray_img)
rgb_img = cv2.cvtColor(read_img, cv2.COLOR_BGR2RGB)
cv2.imwrite('rgb_img.jpg', rgb_img)
rgb_hsv_img = cv2.cvtColor(read_img, cv2.COLOR_RGB2HSV)
cv2.imwrite('rgb_hsv_img.jpg', rgb_hsv_img)
bgr_hsv_img = cv2.cvtColor(read_img, cv2.COLOR_BGR2HSV)
cv2.imwrite('bgr_hsv_img.jpg', bgr_hsv_img)
hls_img = cv2.cvtColor(read_img, cv2.COLOR_RGB2HLS)
cv2.imwrite('hls_img.jpg', hls_img)
