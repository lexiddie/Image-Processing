#Sokvathara Lin 6018002
import cv2
import numpy as np
from scipy import ndimage
from skimage.exposure import rescale_intensity
from skimage.registration import phase_cross_correlation


def convolve(image, kernel):
    (iH, iW) = image.shape[:2]
    (kH, kW) = kernel.shape[:2]
    pad = (kW - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype="float32")
    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
            k = (roi * kernel).sum()
            output[y - pad, x - pad] = k
    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype("uint8")
    return output


kernelStudentID = np.array((
    [6, 0, 1],
    [8, 0, 0],
    [2, 1, 1]), dtype="int")

testKernel = np.array((
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]), dtype="int")

img = cv2.imread('./Lena.png')
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
convolution = convolve(grayImg, kernelStudentID)
# convolution = convolve(grayImg, testKernel)


def correlation_coefficient(patch1, patch2):
    product = np.mean((patch1 - patch1.mean()) * (patch2 - patch2.mean()))
    stds = patch1.std() * patch2.std()
    if stds == 0:
        return 0
    else:
        product /= stds
        return product


im1 = grayImg
sh_row, sh_col = im1.shape
im2 = convolution
translation = phase_cross_correlation(im1, im2, upsample_factor=10)[0]
im2_register = ndimage.shift(im2, translation)
d = 1
correlation = np.zeros_like(im1)

for i in range(d, sh_row - (d + 1)):
    for j in range(d, sh_col - (d + 1)):
        correlation[i, j] = correlation_coefficient(im1[i - d: i + d + 1,
                                                    j - d: j + d + 1],
                                                    im2[i - d: i + d + 1,
                                                    j - d: j + d + 1])

cv2.imwrite('Lenna_output_gray.tif', grayImg)
cv2.imwrite('Lenna_output_correlation.tif', correlation)
cv2.imwrite('Lenna_output_convolution.tif', convolution)
cv2.imshow('Original', grayImg)
cv2.imshow('Correlation', correlation)
cv2.imshow('Convolution', convolution)
print('Finish Compute')
cv2.waitKey(0)
cv2.destroyAllWindows()
