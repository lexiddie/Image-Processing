from math import atan2, sqrt, cos, sin, pi
import cv2
import numpy as np


class MyImage:
    def __init__(self, img_name):
        self.img = cv2.imread(img_name)
        self.__name = img_name

    def __str__(self):
        return self.__name


def drawAxis(img, p_, q_, colour, scale):
    p = list(p_)
    q = list(q_)

    angle = atan2(p[1] - q[1], p[0] - q[0])
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)


def getOrientation(pts, img):
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i, 0] = pts[i, 0, 0]
        data_pts[i, 1] = pts[i, 0, 1]
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    cntr = (int(mean[0, 0]), int(mean[0, 1]))
    cv2.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (
        cntr[0] + 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0],
        cntr[1] + 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0])
    p2 = (
        cntr[0] - 0.02 * eigenvectors[1, 0] * eigenvalues[1, 0],
        cntr[1] - 0.02 * eigenvectors[1, 1] * eigenvalues[1, 0])
    drawAxis(img, cntr, p1, (0, 255, 0), 1)
    drawAxis(img, cntr, p2, (255, 255, 0), 5)
    angle = atan2(eigenvectors[0, 1], eigenvectors[0, 0])  # orientation in radians
    return angle


img = cv2.imread('./IMG.png')
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, binary = cv2.threshold(grayImg, 80, 255, cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
image = cv2.drawContours(img, contours, -1, (0, 255, 0), 5)

dimension = img.shape
height = dimension[0]
width = dimension[1]
resized = cv2.resize(img, (width - 50, height - 50))

for cnt in contours:
    area = cv2.contourArea(cnt)
    (x, y, w, h) = cv2.boundingRect(cnt)
    if w > 100 and h > 100:
        cv2.rectangle(resized, (x, y), (x + w, y + h), (0, 255, 0), 1)
        crop_img = resized[y:y + h, x:x + w]
    if area > 200:
        cv2.drawContours(resized, cnt, -1, (0, 255, 0), 5)
        angle = getOrientation(cnt, resized)

cv2.imshow('Orientation & Visualize', resized)
cv2.imwrite('orientation_visualize.bmp', resized)
print('Finish Compute')
cv2.waitKey(0)
cv2.destroyAllWindows()
