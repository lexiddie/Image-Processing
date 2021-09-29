import cv2
from scipy.ndimage import convolve
import numpy as np


kernel = np.array((
    [6, 0, 1, 8, 0, 0, 2],
    [0, 1, 8, 0, 0, 2, 6],
    [1, 8, 0, 0, 2, 6, 0],
    [8, 0, 0, 2, 6, 0, 1],
    [0, 0, 2, 6, 0, 1, 8],
    [0, 2, 6, 0, 1, 8, 0],
    [2, 6, 0, 1, 8, 0, 0]), dtype=np.float32)


img = cv2.imread('./IMG.png')
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
kernel = np.true_divide(kernel, 49)
img_filtered = cv2.filter2D(grayImg, -1, kernel)
convolution = convolve(grayImg, kernel, mode='constant', cval=0.0)

result = np.concatenate((grayImg, img_filtered, convolution), axis=1)
cv2.imshow('Original & Kernel Filter2D & Convolution', result)
cv2.imwrite('convolution.bmp', convolution)
print('Finish Compute')
cv2.waitKey(0)
cv2.destroyAllWindows()
