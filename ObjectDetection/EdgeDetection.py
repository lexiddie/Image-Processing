import cv2 as cv
import numpy as np
import copy
import matplotlib.pyplot as plt

img = cv.imread('./items.jpg')
original = copy.deepcopy(img)
rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
gray_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
b, g, r = cv.split(img)

edges = cv.Canny(image=img, threshold1=127, threshold2=130)
median_value = np.median(img)

# Lower threshold to either 0 or 70% of the median value whenever is greater
lower = int(max(0, 0.7 * median_value))
# Upper threshold to either 130% of the median or the max 255, whenerver is smaller
upper = int(min(255, 1.3 * median_value))

# To show strong edges of the image further can increase the kernel size expansion
blur_img = cv.blur(img, ksize=(5, 5))
# Expanding the upper threshold limit to go higher and higher to generalize a little bit more with (100)
edges = cv.Canny(image=blur_img, threshold1=lower, threshold2=upper+100)

stack = np.hstack([gray_img, edges])
cv.imshow('Orginal', stack)
print('Finish Compute')
cv.waitKey(0)
cv.destroyAllWindows()
