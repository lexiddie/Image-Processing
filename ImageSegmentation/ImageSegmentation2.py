# Image Segmentation with Distance Transform and Watershed Algorithm

from random import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import colorsys
import argparse
import random as rng

img1 = './cards1.jpg'
img2 = './cards2.jpg'

read_img = cv2.imread(img1)
img_data = np.array(read_img)

# Change the background from white to black, since that will help later to extract
# better results during the use of Distance Transform
read_img[np.all(read_img == 255, axis=2)] = 0

# Create a kernel that we will use to sharpen our image
# an approximation of second derivative, a quite strong kernel
kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
# do the laplacian filtering as it is
# well, we need to convert everything in something more deeper then CV_8U
# because the kernel has some negative values,
# and we can expect in general to have a Laplacian image with negative values
# BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
# so the possible negative number will be truncated
img_laplacian = cv2.filter2D(read_img, cv2.CV_32F, kernel)
sharp = np.float32(read_img)
img_result = sharp - img_laplacian

# convert back to 8bits gray scale
img_result = np.clip(img_result, 0, 255)
img_result = img_result.astype('uint8')
img_laplacian = np.clip(img_laplacian, 0, 255)
img_laplacian = np.uint8(img_laplacian)

# Create binary image from source image
bw = cv2.cvtColor(img_result, cv2.COLOR_BGR2GRAY)
_, bw = cv2.threshold(bw, 40, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imwrite('binary_img.jpg', bw)

# Perform the distance transform algorithm
dist = cv2.distanceTransform(bw, cv2.DIST_L2, 3)
# Normalize the distance image for range = {0.0, 1.0}
# so we can visualize and threshold it
cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
cv2.imwrite('distance_transform_img.jpg', dist)

# Threshold to obtain the peaks
# This will be the markers for the foreground objects
_, dist = cv2.threshold(dist, 0.4, 1.0, cv2.THRESH_BINARY)
# Dilate a bit the dist image
kernel1 = np.ones((3, 3), dtype=np.uint8)
dist = cv2.dilate(dist, kernel1)
cv2.imwrite('peaks_img.jpg', dist)

# Create the CV_8U version of the distance image
# It is needed for findContours()
dist_8u = dist.astype('uint8')
# Find total markers
contours, _ = cv2.findContours(dist_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Create the marker image for the watershed algorithm
markers = np.zeros(dist.shape, dtype=np.int32)
# Draw the foreground markers
for i in range(len(contours)):
    cv2.drawContours(markers, contours, i, (i + 1), -1)
# Draw the background marker
cv2.circle(markers, (5, 5), 3, (255, 255, 255), -1)
cv2.imwrite('makers_img.jpg', markers * 10000)

# Perform the watershed algorithm
cv2.watershed(img_result, markers)
# mark = np.zeros(markers.shape, dtype=np.uint8)
mark = markers.astype('uint8')
mark = cv2.bitwise_not(mark)
# uncomment this if you want to see how the mark
# image looks like at that point
# cv.imshow('Markers_v2', mark)
# Generate random colors
colors = []
for contour in contours:
    colors.append((rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256)))
# Create the result image
dst = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)
# Fill labeled objects with random colors
for i in range(markers.shape[0]):
    for j in range(markers.shape[1]):
        index = markers[i, j]
        if 0 < index <= len(contours):
            dst[i, j, :] = colors[index - 1]
# Visualize the final image
cv2.imwrite('visualize_img.jpg', dst)