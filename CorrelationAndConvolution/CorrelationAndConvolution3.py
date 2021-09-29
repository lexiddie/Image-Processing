import cv2
import numpy as np
from scipy.ndimage import correlate, convolve
from skimage.exposure import rescale_intensity
# from matplotlib import pyplot as plt
import matplotlib.pyplot as plt


def convolve_custom(image, custom_kernel):
    (iH, iW) = image.shape[:2]
    (kH, kW) = custom_kernel.shape[:2]
    pad = (kW - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype="float32")
    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
            k = (roi * custom_kernel).sum()
            output[y - pad, x - pad] = k
    output = rescale_intensity(output, in_range=(0, 255))
    output = 255 * output
    output = (output * 255).astype("uint8")
    return output


kernel = np.array((
    [6, 0, 1, 8, 0, 0, 2],
    [0, 1, 8, 0, 0, 2, 6],
    [1, 8, 0, 0, 2, 6, 0],
    [8, 0, 0, 2, 6, 0, 1],
    [0, 0, 2, 6, 0, 1, 8],
    [0, 2, 6, 0, 1, 8, 0],
    [2, 6, 0, 1, 8, 0, 0]), dtype=np.float32)

img = cv2.imread('./mastercard.png')
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
kernel = np.true_divide(kernel, 49)
img_filtered = cv2.filter2D(grayImg, -1, kernel)
convolution = convolve(grayImg, kernel, mode='constant', cval=0.0)
correlation = correlate(grayImg, kernel)
# convolution = convolve_custom(grayImg, kernel)

hist1 = cv2.calcHist([convolution], [0], None, [256], [0, 256])
hist2 = cv2.calcHist([correlation], [0], None, [256], [0, 256])
plt.hist(convolution.ravel(), 256, [0, 256])
plt.title('Convolution')
plt.show()
plt.hist(correlation.ravel(), 256, [0, 256])
plt.title('Correlation')
plt.show()
# np.savetxt('histogram1.dat', hist1)
# np.savetxt('histogram2.dat', hist2)

# result = np.concatenate((grayImg, img_filtered, convolution), axis=1)
# cv2.imshow('Original & Kernel Filter2D & Convolution', result)
# cv2.imwrite('convolution.bmp', convolution)
# print('Finish Compute')
# cv2.waitKey(0)
# cv2.destroyAllWindows()
