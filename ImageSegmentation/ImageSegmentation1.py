import cv2
import numpy as np

image = cv2.imread('./1.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(gray)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
kernel = np.ones((5, 32), np.uint8)
img_dilation = cv2.dilate(thresh, kernel, iterations=1)
contours, hierarchy = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

for i, ctr in enumerate(sorted_contours):
    x, y, w, h = cv2.boundingRect(ctr)
    roi = image[y:y + h, x:x + w]
    cv2.imwrite('%s/%s.png' % ('images', i), roi)
    cv2.rectangle(image, (x, y), (x + w, y + h), (90, 0, 255), 2)
    cv2.waitKey(0)

cv2.imshow('Marked Area', image)
cv2.waitKey(0)


