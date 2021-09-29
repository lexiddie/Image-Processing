import cv2
import numpy as np

img1 = './img1.jpg'
read_img = cv2.imread(img1)
img_data = np.array(read_img)
gray = cv2.cvtColor(read_img, cv2.COLOR_BGR2GRAY)
# Apply GaussianBlur to reduce image noise if it is required
gray = cv2.GaussianBlur(gray, (5, 5), 0)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cv2.imwrite('thresh.jpg', thresh)

# Applying Otsu's method setting the flag value into cv.THRESH_OTSU.
# Use a bimodal image as an input.
# Optimal threshold value is determined automatically.
otsu_threshold, image_result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,)
cv2.imwrite('otsu_method.jpg', image_result)
print("Otsu's Binarization; Obtained threshold: ", otsu_threshold)

# noise removal
kernel = np.ones((3, 3), np.uint8)
# opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
opening = cv2.morphologyEx(image_result, cv2.MORPH_OPEN, kernel, iterations=2)
cv2.imwrite('open.jpg', opening)
# sure background area
sure_bg = cv2.dilate(opening, kernel, iterations=3)
cv2.imwrite('sure_background.jpg', sure_bg)
# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
cv2.imwrite('sure_foreground.jpg', sure_fg)
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)
# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)
# ret, markers = cv2.connectedComponents(sure_bg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers + 1
# Now, mark the region of unknown with zero
markers[unknown == 255] = 0
markers = cv2.watershed(read_img, markers)
cv2.imwrite('maker.jpg', markers)
read_img[markers == -1] = [255, 0, 0]

cv2.imshow('Result Images', read_img)
cv2.imwrite('final_result.bmp', read_img)
print('Finish Compute')
cv2.waitKey(0)
cv2.destroyAllWindows()
