import numpy as np
import cv2 as sv
from matplotlib import pyplot as plt

# img1 = './img1.png'
#
# img = cv2.imread(img1)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # Now we split the image to 5000 cells, each 20x20 size
# cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]
#
# # Make it into a Numpy array. It size will be (50,100,20,20)
# x = np.array(cells)
#
# # Now we prepare train_data and test_data.
# train = x[:, :50].reshape(-1, 400).astype(np.float32)  # Size = (2500,400)
# test = x[:, 50:100].reshape(-1, 400).astype(np.float32)  # Size = (2500,400)
#
# # Create labels for train and test data
# k = np.arange(10)
# train_labels = np.repeat(k, 250)[:, np.newaxis]
# test_labels = train_labels.copy()
#
# # Initiate kNN, train the data, then test it with test data for k=1
# knn = cv2.ml.KNearest_create()
# knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)
# ret, result, neighbours, dist = knn.find_nearest(test, k=5)
#
# # Now we check the accuracy of classification
# # For that, compare the result with test_labels and check which are wrong
# matches = result == test_labels
# correct = np.count_nonzero(matches)
# accuracy = correct * 100.0 / result.size
# print(accuracy)


import numpy as np
import cv2 as cv

img1 = './img1.png'
img2 = './img2.png'
img3 = './img3.png'
img4 = './img4.png'
img5 = './img5.png'
img6 = './img6.png'
img7 = './img7.png'

main_read = img7

img = cv.imread(main_read)
# width = 2500
# height = 400  # keep original height
# dim = (width, height)
#
# # resize image
# img = cv.resize(img, dim, interpolation=cv.INTER_AREA)

# gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# rows = np.vsplit(img, 50)
# cells = []
# for row in rows:
#     row_cells = np.hsplit(row, 50)
#     for cell in row_cells:
#         cell = cell.flatten()
#         cells.append(cell)
# cells = np.array(cells, dtype=np.float32)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Now we split the image to 5000 cells, each 20x20 size
cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]
# Make it into a Numpy array: its size will be (50,100,20,20)
x = np.array(cells)
# Now we prepare the training data and test data
train = x[:, :50].reshape(-1, 400).astype(np.float32)  # Size = (2500,400)
test = x[:, 50:100].reshape(-1, 400).astype(np.float32)  # Size = (2500,400)
# Create labels for train and test data
k = np.arange(10)
train_labels = np.repeat(k, 250)[:, np.newaxis]
test_labels = train_labels.copy()
# Initiate kNN, train it on the training data, then test it with the test data with k=1
knn = cv.ml.KNearest_create()
knn.train(train, cv.ml.ROW_SAMPLE, train_labels)
ret, result, neighbours, dist = knn.findNearest(test, k=5)
# Now we check the accuracy of classification
# For that, compare the result with test_labels and check which are wrong
matches = result == test_labels
correct = np.count_nonzero(matches)
accuracy = correct * 100.0 / result.size
print('Accuracy', accuracy)
# Save the data
np.savez('knn_data.npz', train=train, train_labels=train_labels)
with np.load('knn_data.npz') as data:
    print(data.files)
    train = data['train']
    train_labels = data['train_labels']

knn = cv.ml.KNearest_create()
knn.train(train, cv.ml.ROW_SAMPLE, train_labels)
test_img = cv.imread(main_read)
test_img = cv.cvtColor(test_img, cv.COLOR_BGR2GRAY)
test_img = cv.resize(test_img, (20, 20))
x = np.array(test_img)
test_img = x.reshape(-1, 400).astype(np.float32)
ret, result, neighbours, distance = knn.findNearest(test_img, k=1)
print('Result', result, 'Neighbours', neighbours, 'Distance', distance)
