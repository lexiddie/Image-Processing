from math import atan2, sqrt, cos, sin, pi
import cv2
import numpy as np
import matplotlib.pyplot as plt


def drawAxis(src_img, p_, q_, colour, scale):
    p = list(p_)
    q = list(q_)

    angle = atan2(p[1] - q[1], p[0] - q[0])
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv2.line(src_img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv2.line(src_img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv2.line(src_img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)


def getOrientation(pts, src_img):
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for j in range(data_pts.shape[0]):
        data_pts[j, 0] = pts[j, 0, 0]
        data_pts[j, 1] = pts[j, 0, 1]
    mean = np.empty(0)
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    contour = (int(mean[0, 0]), int(mean[0, 1]))
    cv2.circle(src_img, contour, 3, (255, 0, 255), 2)
    p1 = (
        contour[0] + 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0],
        contour[1] + 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0])
    p2 = (
        contour[0] - 0.02 * eigenvectors[1, 0] * eigenvalues[1, 0],
        contour[1] - 0.02 * eigenvectors[1, 1] * eigenvalues[1, 0])
    drawAxis(src_img, contour, p1, (0, 255, 0), 1)
    drawAxis(src_img, contour, p2, (255, 255, 0), 5)
    angle = atan2(eigenvectors[0, 1], eigenvectors[0, 0])
    return angle


# img = cv2.imread('./colorful.png')
img = cv2.imread('./test.jpg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dimension = img.shape
height = dimension[0]
width = dimension[1]
update_img = cv2.resize(img, (width, height))

blur = cv2.GaussianBlur(gray_img, (25, 25), 0)
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
noise_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, noise_kernel, iterations=2)
close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
close_img = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, close_kernel, iterations=3)

# coordination = cv2.findNonZero(close_img)
# x, y, w, h = cv2.boundingRect(coordination)
# cv2.rectangle(update_img, (x, y), (x + w, y + h), (36, 255, 12), 2)
# original = img.copy()
# crop_img = original[y:y + h, x:x + w]

for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    # (x, y, w, h) = cv2.boundingRect(cnt)
    if area < 1e2 or 1e5 < area:
        continue
    cv2.drawContours(update_img, contours, i, (0, 0, 255), 2)
    cv2.drawContours(close_img, contours, i, (0, 0, 255), 2)
    getOrientation(cnt, update_img)

hist = cv2.calcHist([close_img], [0], None, [256], [0, 256])
plt.hist(close_img.ravel(), 256, [0, 256])
plt.title('Histogram')
plt.show()
np.savetxt('histogram.dat', hist)

cv2.imshow('Orientation & Visualize', update_img)
print('Finish Compute')
cv2.waitKey(0)
cv2.destroyAllWindows()
