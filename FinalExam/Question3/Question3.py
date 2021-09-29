import cv2 as cv
import numpy as np

image = cv.imread('./COVID-19.jpg')
seperate_blur = cv.medianBlur(image, 49)
gray_img = cv.cvtColor(seperate_blur, cv.COLOR_BGR2GRAY)

# Use invert when the objects are connected
ret, thresh = cv.threshold(gray_img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
# ret, thresh = cv.threshold(gray_img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
cv.imwrite('thresh_otsu.jpg', thresh)

# noise removal
kernel = np.ones((3, 3), np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
cv.imwrite('opening.jpg', opening)

# sure background area
sure_bg = cv.dilate(opening, kernel, iterations=3)
cv.imwrite('sure_background.jpg', sure_bg)

# Finding sure foreground area
dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
cv.imwrite('distance_transford.jpg', dist_transform)
ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
cv.imwrite('sure_foreground.jpg', sure_fg)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg, sure_fg)
cv.imwrite('unknown_region.jpg', unknown)

# Marker labelling
ret, markers = cv.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers + 1

# Now, mark the region of unknown with zero
markers[unknown == 255] = 0
markers = cv.watershed(image, markers)
cv.imwrite('connect_makers.jpg', markers)
image[markers == -1] = [255, 0, 0]

contours, hierarchy = cv.findContours(markers.copy(), cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    if hierarchy[0][i][3] == -1:
        cv.drawContours(image, contours, i, (0, 0, 255), 5)

stack = np.hstack([thresh, sure_fg, unknown])
cv.imwrite('visualize_original.bmp', image)
cv.imshow('Stack', stack)
cv.imshow('Makers', image)
print('Finish Compute')
cv.waitKey(0)
cv.destroyAllWindows()
