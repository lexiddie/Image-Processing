import cv2 as cv
import numpy as np

img = cv.imread('./dog.jpg')
b, g, r = cv.split(img)

zeros = np.zeros(img.shape[:2], np.uint8)

red_merge = cv.merge([zeros, zeros, r])
green_merge = cv.merge([zeros, g, zeros])
blue_merge = cv.merge([b, zeros, zeros])

stack = np.hstack([img, red_merge, green_merge, blue_merge])
cv.imshow('Orginal', stack)
print('Finish Compute')
cv.waitKey(0)
cv.destroyAllWindows()
