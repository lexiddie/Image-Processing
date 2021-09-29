import cv2
import numpy as np
from scipy.stats import skew

# Intensity
img = cv2.imread('./object1.png')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print('Image size {}'.format(gray_img.size))
min_intensity = gray_img.min()
print('Minimum Grayscale Intensity value in this image {}'.format(min_intensity))
print('Maximum Grayscale Intensity value in this image {}'.format(gray_img.max()))
print('Average Grayscale Intensity value in this image {}'.format(gray_img.mean()))

# Variance
w, h = gray_img.shape
img_data = np.array(gray_img)
print('The variance deviation value in this image {}'.format(np.var(img_data)))
print('The standard deviation value in this image {}'.format(np.std(img_data)))

# Asymmetry (Skew)
print('Skew of this grayscale image is: {}'.format(skew(img_data)))

