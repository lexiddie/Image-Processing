import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import imutils

correlation_img = cv2.imread('./correlation.bmp')
convolution_img = cv2.imread('./convolution.bmp')
grayImg1 = cv2.cvtColor(correlation_img, cv2.COLOR_BGR2GRAY)
grayImg2 = cv2.cvtColor(convolution_img, cv2.COLOR_BGR2GRAY)

(score, diff) = ssim(grayImg1, grayImg2, multichannel=True, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))
print("Diff: {}".format(diff))

_, thresh = cv2.threshold(diff, 80, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(grayImg1, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.rectangle(grayImg2, (x, y), (x + w, y + h), (0, 0, 255), 2)

result = np.concatenate((diff, thresh), axis=1)
cv2.imshow('The Difference & The Thresh', result)

cv2.imwrite('difference_correlation_convolution.bmp', diff)
print('Finish Compute')
cv2.waitKey(0)
cv2.destroyAllWindows()
