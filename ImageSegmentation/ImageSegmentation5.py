import cv2
import numpy as np
import matplotlib.pyplot as plt

# Interactive Foreground Extraction using GrabCut Algorithm

img1 = './GrabCut.jpg'
read_img = cv2.imread(img1)

mask = np.zeros(read_img.shape[:2], np.uint8)

bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

rect = (161, 79, 150, 150)

cv2.grabCut(read_img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
img = read_img * mask2[:, :, np.newaxis]

plt.imshow(img)
plt.colorbar()
plt.show()
