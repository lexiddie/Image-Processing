import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv.LUT(image, table)


img_path = './COVID-19.jpg'
r_channel = 'r_channel.bmp'
g_channel = 'g_channel.bmp'
b_channel = 'b_channel.bmp'

img = cv.imread(img_path)
rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

b, g, r = cv.split(img)

cv.imwrite(r_channel, r)
cv.imwrite(g_channel, g)
cv.imwrite(b_channel, b)

colors = ['b', 'g', 'r']

for i, color in enumerate(colors):
    histogram = cv.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histogram, color=color)
    plt.xlim([0, 256])

plt.title('Histogram')
plt.show()


increase_intensity = adjust_gamma(img, 0.33)
stack_intensity = np.hstack([img, increase_intensity])
cv.imwrite('increase_intensity.jpg', increase_intensity)
cv.imshow('Increase intensity', stack_intensity)
print('Finish Compute')
cv.waitKey(0)
cv.destroyAllWindows()
