import cv2
import numpy as np
import matplotlib.pyplot as plt


def hist_match(source, template):
    old_shape = source.shape
    source = source.ravel()
    template = template.ravel()
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
    return interp_t_values[bin_idx].reshape(old_shape)


img1 = cv2.imread('./object1.png')
img2 = cv2.imread('./object2.png')
gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

hist = cv2.calcHist([gray_img1], [0], None, [256], [0, 256])
plt.hist(gray_img1.ravel(), 256, [0, 256])
plt.title('Histogram')
plt.show()
np.savetxt('histogram.dat', hist)
hist_equalization = cv2.equalizeHist(gray_img1)
hist_eq = cv2.calcHist([hist_equalization], [0], None, [256], [0, 256])
plt.hist(hist_equalization.ravel(), 256, [0, 256])
plt.title('Histogram Equalization')
plt.show()
np.savetxt('histogram_equalization.dat', hist)

histogram_matching = hist_match(gray_img1, gray_img2)

result = np.concatenate((gray_img1, hist_equalization, histogram_matching), axis=1)
cv2.imshow('The Original & Histogram Equalization & Histogram Matching', result)
# cv2.imshow('The Difference', diff_img)
# cv2.imwrite('creditcard_difference.bmp', diff)
print('Finish Compute')
cv2.waitKey(0)
cv2.destroyAllWindows()
