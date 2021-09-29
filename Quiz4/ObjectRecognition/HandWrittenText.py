import cv2 as cv
import numpy as np

# Load the data and convert the letters to numbers
data = np.loadtxt('letter-recognition.data', dtype='float32', delimiter=',',
                  converters={0: lambda ch: ord(ch) - ord('A')})
# Split the dataset in two, with 10000 samples each for training and test sets
train, test = np.vsplit(data, 2)
# Split trainData and testData into features and responses
responses, trainData = np.hsplit(train, [1])
labels, testData = np.hsplit(test, [1])
# Initiate the kNN, classify, measure accuracy
knn = cv.ml.KNearest_create()
knn.train(trainData, cv.ml.ROW_SAMPLE, responses)
ret, result, neighbours, dist = knn.findNearest(testData, k=5)
correct = np.count_nonzero(result == labels)
accuracy = correct * 100.0 / 10000
print(accuracy)

# Initiate kNN classifier
knn = cv.ml.KNearest_create()

# perform training of data
knn.train(trainData,
          cv.ml.ROW_SAMPLE,
          labels)

# obtain the output from the
# classifier by specifying the
# number of neighbors.
ret, output, neighbours, distance = knn.findNearest(testData, k=3)

# Check the performance and
# accuracy of the classifier.
# Compare the output with test_labels
# to find out how many are wrong.
matched = output == labels
correct_OP = np.count_nonzero(matched)

# Calculate the accuracy.
accuracy = (correct_OP * 100.0) / output.size

# Display accuracy.
print(accuracy)

test_img = cv.imread('test_text.png')

test_img = cv.cvtColor(test_img, cv.COLOR_BGR2GRAY)
test_img = cv.resize(test_img, (160, 160))
x = np.array(test_img)
test_img = x.reshape(-1, 400).astype(np.float32)
ret, result, neighbours, distance = knn.findNearest(test_img, k=1)
# Print the predicted number
print('result', result, 'neighbours', neighbours, 'distance', distance)