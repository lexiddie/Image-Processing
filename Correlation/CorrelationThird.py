import cv2
from scipy.ndimage import correlate, convolve
import numpy as np
from scipy import ndimage
from scipy import signal
import matplotlib.pyplot as plt
from skimage import io, feature, registration
from scipy.ndimage.filters import correlate, convolve

# Divide each value by 45
# kernel = np.true_divide(kernel, 45)

# Convert all value to Int
# img_data = np.array(kernel, dtype=np.uint8)

kernel = np.array((
    [6, 0, 1, 8, 0, 0, 2],
    [0, 1, 8, 0, 0, 2, 6],
    [1, 8, 0, 0, 2, 6, 0],
    [8, 0, 0, 2, 6, 0, 1],
    [0, 0, 2, 6, 0, 1, 8],
    [0, 2, 6, 0, 1, 8, 0],
    [2, 6, 0, 1, 8, 0, 0]), dtype=np.float32)


kernelMain = np.array((
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]), dtype=np.float32)

img = cv2.imread('./Lena.png')
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
kernel = np.true_divide(kernel, 49)
img_filtered = cv2.filter2D(grayImg, -1, kernel)

correlation = correlate(grayImg, kernel)

result = np.concatenate((grayImg, img_filtered, correlation), axis=1)

cv2.imshow('Original & Kernel Filter2D & Correlation', result)
print('Finish Compute')
cv2.waitKey(0)
cv2.destroyAllWindows()
