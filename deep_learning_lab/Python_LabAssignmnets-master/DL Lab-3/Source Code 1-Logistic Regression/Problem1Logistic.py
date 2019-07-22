# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 23:31:20 2018

@author: shruthi
"""

#! /usr/bin/env python3

"""
Implementation of logistic regression using the breast cancer Wisconsin set.
Dataset contains data on breast cancer tumors along with a classification of
whether tumor is benign or malignant.  Logistic regression model is created
to predict tumor status based on prediction factors
http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29
"""

import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

data_file = 'winscdata.txt'

# load data from CSV file into dataframe using pandas.  Data has these columns:
#
# sample code number
# clump thickness
# uniformity of cell size
# uniformity of cell shape
# marginal adhesion
# single epithelial cell size
# bare nuclei
# bland chromatin
# normal nucleoli
# mitoses
# class (2 = benign, 4 = malignant)
data = pd.read_csv(data_file, header=None)

# make sure everything is an integer
data = data.astype(int)

#print(data)
# make numpy array of 10 predictors alone
factors = np.array(data.values[:,0:10])

# determine total number of rows in dataset
num_rows = data.values.shape[0]

# make one_hot vectors for each data point to record classification labels
# each one hot vector is a two element array where 'truth' position has a value
# of one.  For our vector, treat column 0 as the '2'/benign value and column 1
# as the '4'/malignant value
one_hots = np.zeros(shape=[num_rows,2], dtype=np.float32)
for i in range(num_rows):
    if data.values[i,10] == 2:
        one_hots[i] = [1,0]
    elif data.values[i,10] == 4:
        one_hots[i] = [0,1]

# use scikit to split predictors and one hot vectors/labels into training
# and test sets.  Use 75% of data for training
x_train, x_test, y_train, y_test =  train_test_split(factors, one_hots, test_size=0.25, random_state=42)

# set learning rate
learning_rate = 0.1

# make 100 iterations over training set
training_runs = 100

# create placeholder variables for 10 predictors and two classifications
x = tf.placeholder(tf.float32, [None, 10], name='x') 
y = tf.placeholder(tf.float32, [None, 2], name='y')

# create placeholder variables for weights and bias
w = tf.Variable(tf.zeros([10, 2]), name='w')
b = tf.Variable(tf.zeros([2]), name='b')

# create model, define loss function and create optimizer to reduce error
logits = tf.matmul(x, w) + b
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
loss = tf.reduce_mean(entropy)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
    # initialize variables
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./graphs/linear', sess.graph)
    

    # iterate over training set and process data, print loss at each loop
    for i in range(training_runs):
        _, l = sess.run([optimizer, loss], feed_dict={x: x_train, y: y_train})
        print("training run:", i, "loss: ", l)

    # After optimizing model, use it on test set
    total_correct_preds = 0
    _, loss_batch, logits_batch = sess.run([optimizer, loss, logits],
                                           feed_dict = {x: x_test, y : y_test})

    # make predictions
    preds = tf.nn.softmax(logits_batch)
    # calculate how many predictions were correct
    correct_preds = tf.equal(tf.argmax(preds,1), tf.argmax(y_test,1))
    accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
    total_correct_preds += sess.run(accuracy)

    # print model accuracy as a perecentage (number of correct predictions
    # divided by total number in test set
    print("Accuracy {0}".format(total_correct_preds / y_test.shape[0] ))