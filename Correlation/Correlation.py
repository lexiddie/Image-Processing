import cv2
import numpy as np

# kernel = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
kernel = np.array((
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]), dtype="int")

# Read in images in grayscale mode
# img = cv2.imread('./Lena.png', 0)

img = cv2.imread('./Lena.png')
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(type(grayImg))
print(grayImg.shape)
print(grayImg.dtype)

img_src = grayImg
kernel = np.true_divide(kernel, 45)
dst = cv2.filter2D(img_src, -1, kernel)
result = np.concatenate((img_src, dst), axis=1)
cv2.imwrite('Lenna_output.png', result)

w, h = img_src.shape
img_data = np.array(img_src)

for i in range(1, w - 1):
    for j in range(1, h - 1):
        temp = img_src[i - 1, j - 1] * kernel[0, 0] + img_src[i - 1, j] * kernel[0, 1] + img_src[i - 1, j + 1] * kernel[0, 2] + img_src[
            i, j - 1] * kernel[1, 0] + img_src[i, j] * kernel[1, 1] + img_src[i, j + 1] * kernel[1, 2] + img_src[i + 1, j - 1] * \
               kernel[2, 0] + img_src[i + 1, j] * kernel[2, 1] + img_src[i + 1, j + 1] * kernel[2, 2]
        img_data[i][j] = temp

img_data = np.array(img_data, dtype=np.uint8)
correlation = np.concatenate((img_src, img_data), axis=1)
print(np.array(img_data).reshape(w, h))
cv2.imwrite('Lenna_output.tif', img_data)
cv2.imshow('Original & Correlation Filter2D Result', correlation)
print('Finish Compute')
cv2.waitKey(0)
cv2.destroyAllWindows()
