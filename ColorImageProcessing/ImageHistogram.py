import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('./dog.jpg')

colors = ['b', 'g', 'r']

for i, color in enumerate(colors):
    histogram = cv.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histogram, color=color)
    plt.xlim([0, 256])

plt.title('Histogram')
plt.show()
