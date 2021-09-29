import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread('./brick.jpg').astype(np.float32) / 255
rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# Allow you to increase or decrease what is the perceived as brightness in the actual image
gamma = 1/4
gamma_correction = np.power(img, gamma)

# Blurring low pass filter with a 2D convolution
font = cv.FONT_HERSHEY_COMPLEX
clone_img = img.copy()
cv.putText(clone_img, text='Bricks', org=(10, 500), fontFace=font, fontScale=10, color=(0, 0, 255), thickness=4)

# Setup kernel for the low pass filter
kernel = np.ones(shape=(5, 5), dtype=np.float32) / 25
# Negative -1 means, I want the input that actually assign the desired output
dst = cv.filter2D(clone_img, -1, kernel)

# cv2 built in blurr kernel
blurr_img = cv.blur(clone_img, ksize=(5, 5))


# result = np.concatenate((img, gamma_correction, clone_img), axis=1)
result = np.concatenate((img, dst, blurr_img), axis=1)
cv.imshow('Orginial, Blurr', result)
print('Finish Compute')
cv.waitKey(0)
cv.destroyAllWindows()
