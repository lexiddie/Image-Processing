import cv2 as cv
import numpy as np

img = cv.imread('./dog_noise.jpg').astype(np.float32) / 255
median = cv.medianBlur(img, 5)

result = np.concatenate((img, median), axis=1)
cv.imshow('Orginial', result)
print('Finish Compute')
cv.waitKey(0)
cv.destroyAllWindows()
