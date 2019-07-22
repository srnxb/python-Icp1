# Kate Williams
# Lab 2 Problem 2
# Due 6/29/2018

from sklearn import datasets
import matplotlib.pyplot as mpl
from sklearn import svm
import random

# Choose and load a data set
irisdataset = datasets.load_iris()  # Load the data set and save it in a variable
theSet = irisdataset.data[:, :2]  # We only take the first two features.
x = []  # List for x variables (sepal length)
y = []  # List for y variable (sepal width)
counter = 0
while counter < len(theSet):  # Split the data into x and y variables
    x.append(theSet[counter])
    y.append(theSet[counter + 1])
    counter += 2

# Randomize the data we're testing on
num = 120  # This is the number of data points we have in the set
i = 0  # This is a counter variable to iterate through
nums = []  # This is a list to store the numbers we're going to use
while i < num:
    nums.append(random.randint(0, 150))  # Include a random number in the set
    i += 1

# Split the data--there are 150 points, so a 20% 80 % split = 30 and 120 points
testData = []  # Empty list to hold the test data points
trainData = []  # Empty list to hold the training data points
counter = 0  # Counter variable
while counter < len(x):
    if counter in nums:  # If the index has been marked earlier
        trainData.append(x[counter])  # Put it in training data
        trainData.append(y[counter])
    else:  # Else
        testData.append(x[counter])  # Put it in testing data
        testData.append(y[counter])
    counter += 2  # Iterate counter variable

# Apply SVC using linear kernel
x = trainData  # Set x variable
y = []  # Set y variable
i = 0
while i < len(x):  # Fill y variable
    y.append(i)
    i += 1
line = svm.SVC(kernel='linear')  # Apply SVC
line.fit(x, y)  # Apply numbers to SVC
mpl.plot(x, 'ro')  # Plot the data
mpl.plot(line.predict(x), 'yo')  # Plot the predicted data
mpl.show()  # Show the data

# Apply SVC using rbf kernel
x = trainData  # Set x variable
y = []  # Set y variable
i = 0
while i < len(x):  # Fill y variable
    y.append(i)
    i += 1
rbf = svm.SVC(kernel='rbf')  # Apply SVC
rbf.fit(x, y)  # Apply numbers to SVC
mpl.plot(x, 'ro')  # Plot the data
mpl.plot(rbf.predict(x), 'yo')  # Plot the predicted data
mpl.show()  # Show the data
