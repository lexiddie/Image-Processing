import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import colorsys
import random as rd

img = cv.imread('./dog.jpg')
rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

b, g, r = cv.split(img)
# print(b, g, r)
np.savetxt('b_channel.dat', b)
np.savetxt('g_channel.dat', g)
np.savetxt('r_channel.dat', r)
cv.imwrite('b_channel.jpg', b)
cv.imwrite('g_channel.jpg', g)
cv.imwrite('r_channel.jpg', r)

hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
h, s, v = cv.split(hsv_img)
print(h, s, v)
np.savetxt('h_channel.dat', h)
np.savetxt('s_channel.dat', s)
np.savetxt('v_channel.dat', v)
cv.imwrite('h_channel.jpg', h)
cv.imwrite('s_channel.jpg', s)
cv.imwrite('v_channel.jpg', v)
hsv_split = np.concatenate((h, s, v), axis=1)
# cv.imshow('Split HSV', hsv_split)

merge_img = cv.merge([b, g, r])

# result = np.concatenate((img, rgb_img), axis=1)
# cv.imshow('Orginial', result)
plt.imshow(img)
plt.show()
stack = np.hstack([img, rgb_img, hsv_img])
cv.imshow('Orginal', stack)
print('Finish Compute')
cv.waitKey(0)
cv.destroyAllWindows()
