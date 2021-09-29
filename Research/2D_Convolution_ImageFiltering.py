import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img1 = './img1.jpg'
read_img = cv.imread(img1)

kernel = np.ones((5, 5), np.float32) / 25
dst = cv.filter2D(read_img, -1, kernel)
plt.subplot(121), plt.imshow(read_img), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(dst), plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()
