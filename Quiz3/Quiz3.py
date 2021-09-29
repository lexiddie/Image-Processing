import cv2
import numpy as np

bmp_path = './IMG_6447.bmp'
r_channel = 'r_channel.bmp'
g_channel = 'g_channel.bmp'
b_channel = 'b_channel.bmp'
jpeg_path = 'jpeg_q100_compression.jpg'
png_path = 'png_compression.png'
jpeg2000_path = 'jpeg2000_compression.jp2'

jpeg_q25_path = 'jpeg_q25_compression.jpg'
jpeg_q50_path = 'jpeg_q50_compression.jpg'
jpeg_q95_path = 'jpeg_q95_compression.jpg'

img = cv2.imread(bmp_path)
img_data = np.array(img)
print('Image Source Shape; Width, Height, Dimension:', img_data.shape)
b, g, r = cv2.split(img_data)
print('Image RGB Value; R:', r)
print('Image RGB Check; R:', img_data[:, :, 2])
print('Image RGB Value; G:', g)
print('Image RGB Check; G:', img_data[:, :, 1])
print('Image RGB Value; B:', b)
print('Image RGB Check; B:', img_data[:, :, 0])

cv2.imwrite(r_channel, r)
cv2.imwrite(g_channel, g)
cv2.imwrite(b_channel, b)

cv2.imwrite(jpeg_path, img_data, [cv2.IMWRITE_JPEG_QUALITY, 100])
cv2.imwrite(png_path, img_data, [cv2.IMWRITE_PNG_COMPRESSION, 9])
cv2.imwrite(jpeg2000_path, img_data, [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, 272])


cv2.imwrite(jpeg_q25_path, img_data, [cv2.IMWRITE_JPEG_QUALITY, 25])
cv2.imwrite(jpeg_q50_path, img_data, [cv2.IMWRITE_JPEG_QUALITY, 50])
cv2.imwrite(jpeg_q95_path, img_data, [cv2.IMWRITE_JPEG_QUALITY, 95])


# result = np.concatenate((img, read_jpeg), axis=1)
# cv2.imshow('R', r)
# cv2.imshow('G', g)
# cv2.imshow('B', b)
# print('Finish Compute')
# cv2.waitKey(0)
# cv2.destroyAllWindows()
