import cv2 as cv
import numpy as np

# Median Blur
# Grayscale
# Binary threshold
# Find contours

image = cv.imread('./coin_connect.jpg')
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
seperate_blur = cv.medianBlur(image, 25)
gray_img = cv.cvtColor(seperate_blur, cv.COLOR_BGR2GRAY)

# Use invert when the objects are connected
ret, thresh = cv.threshold(gray_img, 160, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
# ret, thresh = cv.threshold(gray_img, 160, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

contours, hierarchy = cv.findContours(thresh.copy(), cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    if hierarchy[0][i][3] == -1:
        cv.drawContours(image, contours, i, (0, 0, 255), 10)

stack = np.hstack([gray_img, thresh])
cv.imshow('Stack', stack)
cv.imshow('Orginal', image)
print('Finish Compute')
cv.waitKey(0)
cv.destroyAllWindows()
