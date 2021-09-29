import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import imutils
import matplotlib.pyplot as plt

creditcard = cv2.imread('./creditcard.png')
mastercard = cv2.imread('./mastercard.png')
grayImg1 = cv2.cvtColor(creditcard, cv2.COLOR_BGR2GRAY)
grayImg2 = cv2.cvtColor(mastercard, cv2.COLOR_BGR2GRAY)

(score, diff) = ssim(grayImg1, grayImg2, multichannel=True, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))
print("Diff: {}".format(diff))

diff_img = diff.copy()

_, thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)

for c in contours:
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(creditcard, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.rectangle(mastercard, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.rectangle(diff_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

# result1 = np.concatenate((diff, thresh), axis=1)
# cv2.imshow('The Thresh & The Difference', result1)

hist = cv2.calcHist([diff_img], [0], None, [256], [0, 256])
plt.hist(diff_img.ravel(), 256, [0, 256])
plt.title('Histogram')
plt.show()
np.savetxt('histogram.dat', hist)

result2 = np.concatenate((creditcard, mastercard), axis=1)
cv2.imshow('The Original & The Modified', result2)
# cv2.imshow('The Difference', diff_img)
cv2.imwrite('creditcard_difference.bmp', diff)
print('Finish Compute')
cv2.waitKey(0)
cv2.destroyAllWindows()
