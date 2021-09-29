import cv2
import numpy as np
import colorsys
import random as rd
import scipy
import os

# img1 = './handwritten.png'
# read_img = cv2.imread(img1)
#
# # gray scale conversion
# gray_img = cv2.cvtColor(read_img, cv2.COLOR_BGR2GRAY)
# img_data = np.array(gray_img)
#
# # We will divide the image
# # into 5000 small dimensions
# # of size 20x20
# height, width, channels = read_img.shape
# divisions = list(np.hsplit(i, height) for i in np.vsplit(gray_img, width))
#
# # Convert into Numpy array
# # of size (50,100,20,20)
# NP_array = np.array(divisions)
#
# # Preparing train_data
# # and test_data.
# # Size will be (2500,20x20)
# train_data = NP_array[:, :50].reshape(-1, 400).astype(np.float32)
#
# # Size will be (2500,20x20)
# test_data = NP_array[:, 50:100].reshape(-1, 400).astype(np.float32)
#
# # Create 10 different labels
# # for each type of digit
# k = np.arange(10)
# train_labels = np.repeat(k, 250)[:, np.newaxis]
# test_labels = np.repeat(k, 250)[:, np.newaxis]
#
# # Initiate kNN classifier
# knn = cv2.ml.KNearest_create()
#
# # perform training of data
# knn.train(train_data,
#           cv2.ml.ROW_SAMPLE,
#           train_labels)
#
# # obtain the output from the
# # classifier by specifying the
# # number of neighbors.
# ret, output, neighbours, distance = knn.findNearest(test_data, k=3)
#
# # Check the performance and
# # accuracy of the classifier.
# # Compare the output with test_labels
# # to find out how many are wrong.
# matched = output == test_labels
# correct_OP = np.count_nonzero(matched)
#
# # Calculate the accuracy.
# accuracy = (correct_OP * 100.0) / output.size
#
# # Display accuracy.
# print(accuracy)

# Load the training image
img = cv2.imread('./texts.png')
# Convert this Image in gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Now we split the image to 5000 cells, each 20x20 size
cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]
# Make it into a Numpy array. It size will be (50,100,20,20)
x = np.array(cells)

# Now we prepare train data and test data.
train = x[:, :50].reshape(-1, 400).astype(np.float32)  # Size = (2500,400)

test = x[:, 50:100].reshape(-1, 400).astype(np.float32)  # Size = (2500,400)
# Create labels for train and test data
k = np.arange(10)
train_labels = np.repeat(k, 250)[:, np.newaxis]

test_labels = train_labels.copy()
# Initiate kNN, train the data, then test it with test data for k=5
knn = cv2.ml.KNearest_create()
knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)
