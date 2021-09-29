import cv2

img = cv2.imread('./IMG.png')
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, binary = cv2.threshold(grayImg, 80, 255, cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
image = cv2.drawContours(img, contours, -1, (0, 255, 0), 5)

cv2.imshow('Binary', binary)
cv2.imshow('Draw Contours', image)
cv2.imwrite('extract_object.bmp', image)
print('Finish Compute')
cv2.waitKey(0)
cv2.destroyAllWindows()
