import cv2
import numpy as np

# Windows bitmaps - *.bmp, *.dib (always supported)
# JPEG files - *.jpeg, *.jpg, *.jpe (see the Note section)
# JPEG 2000 files - *.jp2 (see the Note section)
# Portable Network Graphics - *.png (see the Note section)
# WebP - *.webp (see the Note section)
# Portable image format - *.pbm, *.pgm, *.ppm *.pxm, *.pnm (always supported)
# PFM files - *.pfm (see the Note section)
# Sun rasters - *.sr, *.ras (always supported)
# TIFF files - *.tiff, *.tif (see the Note section)
# OpenEXR Image files - *.exr (see the Note section)
# Radiance HDR - *.hdr, *.pic (always supported)
# Raster and Vector geospatial data supported by GDAL (see the Note section)

img_path = './dog.jpg'
r_channel = 'r_channel.bmp'
g_channel = 'g_channel.bmp'
b_channel = 'b_channel.bmp'
jpeg_path = 'jpeg_q100_compression.jpg'
png_path = 'png_compression.png'
jpeg2000_path = 'jpeg2000_compression.jp2'

jpeg_q25_path = 'jpeg_q25_compression.jpg'
jpeg_q50_path = 'jpeg_q50_compression.jpg'
jpeg_q95_path = 'jpeg_q95_compression.jpg'

img = cv2.imread(img_path)
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

