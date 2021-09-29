import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# img1 = cv2.imread('./Lena.jpg', cv2.IMREAD_GRAYSCALE)
# 
# plt.imshow(img1)
# plt.show()
# hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
# plt.hist(img1.ravel(), 256, [0, 256])
# plt.title('My Image Histogram')
# plt.show()


img = cv2.imread('./Lena.png')
# img = cv2.imread('./forest.jpg')
# img = cv2.imread('./IMG_4970.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow('Original Image', img)
cv2.imshow('Gray Image', gray)


# plt.imshow(img, cmap="gray", vmin=0, vmax=255)
# plt.imshow(img)
# plt.imshow(gray)
# plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()

