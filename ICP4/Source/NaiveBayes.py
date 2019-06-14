import pandas as pd
import numpy as np
import random as rnd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create SVM classification object
model = GaussianNB()
# there is other distribution for multinomial classes like Bernoulli Naive Bayes, Refer link
# Train the model using the training sets and check score
X_train = pd.read_csv('./glass.csv')
train_df, test_df = train_test_split(X_train, test_size=0.5, random_state=0)
X_train = train_df.drop("Type",axis=1)
Y_train = train_df["Type"]
X_test = test_df.drop("Type",axis=1)
Y_test = test_df["Type"]
combine = [train_df, test_df]
model.fit(X_train, Y_train)
#Predict Output
predicted= model.predict(X_test)
#print("train dataset",X_train)
#print("test dataset",X_test)
#print("prediction",predicted)
acc_nb = round(model.score(X_train, Y_train) * 100, 2)
acc_nb1 = round(model.score(X_test, Y_test) * 100, 2)
print("Naive Bayes accuracy is:",acc_nb)
print("Naive Bayes accuracy is:",acc_nb1)
# X_train.info()
# X_train.info()
# print('_'*40)
# print("y_train",Y_train)