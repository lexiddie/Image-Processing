import cv2
import numpy as np

bmp_path = './IMG_6059.bmp'
jpeg_path = './IMG_6059.jpg'
jpeg_2000_path = './IMG_6059_2000.jpg'

img = cv2.imread(bmp_path)
cv2.imwrite(jpeg_path, img, [cv2.IMWRITE_JPEG_OPTIMIZE, 9])
cv2.imwrite(jpeg_2000_path, img, [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, 9])
read_jpeg = cv2.imread(jpeg_path)

result = np.concatenate((img, read_jpeg), axis=1)
print(read_jpeg.shape[0])
cv2.imshow('BMP & JPEG', result)
print('Finish Compute')
cv2.waitKey(0)
cv2.destroyAllWindows()
