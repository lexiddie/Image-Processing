import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread('./IMG_4969.jpg', cv2.IMREAD_GRAYSCALE)
hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
plt.hist(img1.ravel(), 256, [0, 256])
plt.title('My Image Histogram')
plt.show()
np.savetxt('ImageHistogram.dat', hist1)

histEqualization, bins = np.histogram(img1.flatten(), 256, [0, 256])

cdf = histEqualization.cumsum()
cdf_normalized = cdf * histEqualization.max() / cdf.max()
plt.plot(cdf_normalized, color='b')
plt.hist(img1.flatten(), 256, [0, 256], color='r')
plt.title('My Histogram Equalization')
plt.xlim([0, 256])
plt.legend(('cdf', 'histogram'), loc='upper left')
plt.show()
np.savetxt('HistEqualization.dat', histEqualization)

originalImage = cv2.imread('./IMG_4969.jpg')
secondImage = cv2.imread('./IMG_4970.jpg')


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


matched = hist_match(originalImage, secondImage)

# hist3 = cv2.calcHist([matched], [0], None, [256], [0, 256])
plt.hist(matched.ravel(), 256, [0, 256])
plt.title('My Match Histogram')
plt.show()
# np.savetxt('match.dat', value)
