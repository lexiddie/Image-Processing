import cv2
import numpy as np

img_path = './COVID-19.jpg'

jpeg_path = 'jpeg_compression.jpg'
png_path = 'png_compression.png'
jpeg2000_path = 'jpeg2000_compression.jp2'

jpeg_q25_path = 'jpeg_q25_compression.jpg'
jpeg_q50_path = 'jpeg_q50_compression.jpg'
jpeg_q95_path = 'jpeg_q95_compression.jpg'

img = cv2.imread(img_path)
img_data = np.array(img)

# For JPEG, it can be a quality from 0 to 100 (the higher is the better). Default value is 95.
cv2.imwrite(jpeg_path, img_data, [cv2.IMWRITE_JPEG_QUALITY, 100])
# For PNG, it can be the compression level from 0 to 9. A higher value means a smaller size and longer compression time
cv2.imwrite(png_path, img_data, [cv2.IMWRITE_PNG_COMPRESSION, 9])
# For JPEG2000, use to specify the target compression rate (multiplied by 1000). The value can be from 0 to 1000.
# Default is 1000.
cv2.imwrite(jpeg2000_path, img_data, [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, 700])

cv2.imwrite(jpeg_q25_path, img_data, [cv2.IMWRITE_JPEG_QUALITY, 25])
cv2.imwrite(jpeg_q50_path, img_data, [cv2.IMWRITE_JPEG_QUALITY, 50])
cv2.imwrite(jpeg_q95_path, img_data, [cv2.IMWRITE_JPEG_QUALITY, 95])

maximum_lossy_compression = cv2.imread(jpeg_q25_path)
stack = np.hstack([img, maximum_lossy_compression])
cv2.imwrite('compression_difference.jpg', stack)
cv2.imshow('Orginal Image & Maximum Lossy Compression (JPEG:25)', stack)
print('Finish Compute')
cv2.waitKey(0)
cv2.destroyAllWindows()
