# Support Vector Machine  Classification Using Linear and RBF Kernel
# Importing the libraries



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Importing the dataset

dataset = pd.read_csv('glass.csv')

# looking at the values of the dataset

dataset.head()

# Spliting the dataset in independent and dependent variables

X = dataset.values  # taking features for prediction
#print(X)
Y = dataset['Type'].values


# Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)


# Fitting SVC Classification to the Training set with rbf  kernel


svc_rbf = SVC(kernel='rbf', C=1, gamma=0.1).fit(X_train, y_train)


# Accuracy of SVM RBF kernel on Training set

print('Accuracy of the SVM RBF Kernel Classification on training part is: ', svc_rbf.score(X_train, y_train))


# Accuracy of SVM RBF Kernel on Test Set

print('Accuracy of the SVM  RBF Kernel Classification on testing part is: ', svc_rbf.score(X_test, y_test))