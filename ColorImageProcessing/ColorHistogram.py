import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('./dog.jpg')
rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
b, g, r = cv.split(img)

red_histogram = cv.calcHist([img], [2], None, [256], [0, 255])
green_histogram = cv.calcHist([img], [1], None, [256], [0, 255])
blue_histogram = cv.calcHist([img], [0], None, [256], [0, 255])


np.savetxt('red_histogram.dat', red_histogram)
np.savetxt('green_histogram.dat', green_histogram)
np.savetxt('blue_histogram.dat', blue_histogram)

plt.subplot(3, 1, 1)
plt.xlim([0, 255])
plt.plot(red_histogram, color='r')
plt.title('Red Histogram')

plt.subplot(3, 1, 2)
plt.xlim([0, 255])
plt.plot(green_histogram, color='g')
plt.title('Green Histogram')

plt.subplot(3, 1, 3)
plt.xlim([0, 255])
plt.plot(blue_histogram, color='b')
plt.title('Blue Histogram')

plt.show()
