# import cv2
# import numpy as np
# 
# gray_img = cv2.imread('./forest.jpg', cv2.IMREAD_GRAYSCALE)
# cv2.imshow('GoldenGate', gray_img)
# 
# while True:
#     k = cv2.waitKey(0) & 0xFF
#     if k == 27 : break  # ESC key to exit
# cv2.destroyAllWindows()

import cv2
import numpy as np
from matplotlib import pyplot as plt

gray_img = cv2.imread('./IMG_4969.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow('Forest', gray_img)
hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
print(hist)
plt.hist(gray_img.ravel(), 256, [0, 256])
plt.title('My Desk Histogram')
plt.show()
np.savetxt('desk.dat', hist)

while True:
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()
