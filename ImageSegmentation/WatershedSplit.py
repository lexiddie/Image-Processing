import cv2 as cv
import numpy as np

# Median Blur
# Grayscale
# Binary threshold
# Noise Removal (Optional)
# Find contours

image = cv.imread('./coin_connect.jpg')
# image = cv.imread('./virus.jpg')
seperate_blur = cv.medianBlur(image, 15)
gray_img = cv.cvtColor(seperate_blur, cv.COLOR_BGR2GRAY)

# Use invert when the objects are connected
ret, thresh = cv.threshold(gray_img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
# ret, thresh = cv.threshold(gray_img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

# noise removal
kernel = np.ones((3, 3), np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

# sure background area
sure_bg = cv.dilate(opening, kernel, iterations=3)

# Finding sure foreground area
dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg, sure_fg)

# Marker labelling
ret, markers = cv.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers + 1

# Now, mark the region of unknown with zero
markers[unknown == 255] = 0
markers = cv.watershed(image, markers)
cv.imwrite('coin_connect_makers.jpg', markers)
image[markers == -1] = [255, 0, 0]

contours, hierarchy = cv.findContours(markers.copy(), cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    if hierarchy[0][i][3] == -1:
        cv.drawContours(image, contours, i, (0, 0, 255), 5)

stack = np.hstack([thresh, sure_fg, unknown])
cv.imshow('Stack', stack)
cv.imshow('Makers', image)
print('Finish Compute')
cv.waitKey(0)
cv.destroyAllWindows()
