import cv2
import matplotlib.pyplot as plt

img = cv2.imread('./forest.jpg')
plt.imshow(img)
plt.show()

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()

img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
plt.imshow(img)
plt.show()

img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
plt.imshow(img)
plt.show()

