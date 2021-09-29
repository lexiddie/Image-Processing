import cv2
import numpy as np
from scipy import ndimage
from skimage.metrics import structural_similarity as ssim
import imutils
import matplotlib.pyplot as plt
from scipy.stats import skew


kernel = np.array((
    [1, 8, 0, 0, 2],
    [8, 0, 0, 2, 1],
    [0, 0, 2, 1, 8],
    [0, 2, 1, 8, 0],
    [2, 1, 8, 0, 0]), dtype=np.float32)

img = cv2.imread('./object.png')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
kernel = np.true_divide(kernel, 25)