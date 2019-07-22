# Kate Williams
# Student #39
# Lab 1 Project 6
# Due 6/15/2018

import numpy as np

x = np.random.randint(0, 21, 15)  # Generate the random list with parameters given

print("The list is " + str(x))  # Print the random list

frequencies = np.bincount(x)  # Make a list of how frequently each number shows up using a NumPy function

theMax = 0  # Variable to hold the maximum value
marker = 0  # Variable to hold where the largest frequency is
counter = 0  # Variable to count out where the largest frequency is

while counter < len(frequencies):  # For every integer in the list of frequencies
    if frequencies[counter] >= theMax:  # If the current number is larger than the maximum
        theMax = frequencies[counter]  # The maximum is the current number
        marker = frequencies[counter]
    counter += 1

print("The most frequent number is " + str(x[marker]))  # Print the most frequent number
