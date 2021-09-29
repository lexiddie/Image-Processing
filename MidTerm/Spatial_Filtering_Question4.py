import cv2
import numpy as np
from scipy import ndimage
from skimage.metrics import structural_similarity as ssim
import imutils
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.ndimage import correlate, convolve
from skimage.exposure import rescale_intensity


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
    [1, 8, 0, 0, 2],
    [8, 0, 0, 2, 1],
    [0, 0, 2, 1, 8],
    [0, 2, 1, 8, 0],
    [2, 1, 8, 0, 0]), dtype=np.float32)

img = cv2.imread('./object1.png')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
kernel = np.true_divide(kernel, 25)

correlation = correlate(gray_img, kernel)
convolution = convolve_custom(gray_img, kernel)
# convolution = convolve(gray_img, kernel, mode='constant', cval=0.0)


(score, diff) = ssim(correlation, convolution, multichannel=True, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))
print("Diff: {}".format(diff))

diff_img = diff.copy()

_, thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)

for c in contours:
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(correlation, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.rectangle(convolution, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.rectangle(diff_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

result = np.concatenate((correlation, convolution), axis=1)
cv2.imshow('Correlation & Convolution', result)
cv2.imshow('The Difference', diff_img)
cv2.imwrite('correlation_convolution_difference.bmp', diff_img)
cv2.imwrite('correlation.bmp', correlation)
cv2.imwrite('convolution.bmp', convolution)
print('Finish Compute')
cv2.waitKey(0)
cv2.destroyAllWindows()

