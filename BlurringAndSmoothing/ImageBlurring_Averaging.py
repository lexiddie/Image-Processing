import cv2 as cv
import matplotlib.pyplot as plt


# Image Blurring (Image Smoothing)

img1 = './img1.jpg'
read_img = cv.imread(img1)

blur = cv.blur(read_img, (5, 5))
cv.imwrite('averaging.jpg', blur)
plt.subplot(121), plt.imshow(read_img), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(blur), plt.title('Averaged Blurring')
plt.xticks([]), plt.yticks([])
plt.show()
