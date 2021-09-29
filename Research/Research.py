import cv2
import numpy as np
import colorsys
import random as rd

img1 = './img1.jpg'
read_img = cv2.imread(img1)

# Convert Image to Data
img_data = np.array(read_img)
# print(img_data)
print('\n')

# Convert Image from HSV to BGR (RGB)
img_hsv2bgr = cv2.cvtColor(read_img, cv2.COLOR_HSV2BGR)

# Get and Split B, G, R
b, g, r = cv2.split(read_img)
print(b, g, r)
print('\n')

# Convert Image from HSV to BGR (RGB)
img_bgr2hsv = cv2.cvtColor(read_img, cv2.COLOR_HSV2BGR)

# Get and Split H, S, V
h, s, v = cv2.split(img_bgr2hsv)
print(h, s, v)
print('\n')

# Get Image Height, Width, and Channels
height, width, channels = read_img.shape
print(height, width, channels)
print('\n')

# Get Image Height, Width
print(read_img.shape[:2])
print('\n')

# test_color = colorsys.hsv_to_rgb(359, 100, 100)
# print(test_color)