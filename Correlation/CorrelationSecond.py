import cv2
from scipy.ndimage import correlate
import numpy as np

img = cv2.imread('./Lena.png')
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_src = grayImg
w, h = img_src.shape
img_data = np.array(img_src)
print(img_data.reshape(w, h))

kernel = np.array((
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]), dtype="int")

img_data = correlate(img_src, kernel)

correlation = np.concatenate((img_src, img_data), axis=1)

cv2.imshow('Correlation Original & Kernel Filter2D', correlation)
print('Finish Compute')
cv2.waitKey(0)
cv2.destroyAllWindows()
