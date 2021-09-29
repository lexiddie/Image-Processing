import cv2 as cv
import numpy as np

img1 = './img1.jpg'
read_img = cv.imread(img1)
img_data = np.array(read_img)
gray = cv.cvtColor(read_img, cv.COLOR_BGR2GRAY)


stack = np.hstack([read_img, gray])
cv.imshow('Orginal', stack)
print('Finish Compute')
cv.waitKey(0)
cv.destroyAllWindows()
