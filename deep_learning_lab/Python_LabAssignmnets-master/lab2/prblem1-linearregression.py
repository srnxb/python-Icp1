# Importing the libraries
import numpy as np
import matplotlib.pyplot as plot
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

boston = datasets.load_diabetes()

# load boston dataset - this contains neighborhood stats and median
# house prices for boston neighborhoods
X = boston.data
print(boston.feature_names)

# get output -- this is median value of homes in $1000s
y = boston.target

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Applying Linear Discriminant Analysis

lda = LinearDiscriminantAnalysis(n_components = 2)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)


#Logistic regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test

aranged_pc1 = np.arange(start=X_set[:, 0].min(), stop=X_set[:, 0].max(), step=0.01)
aranged_pc2 = np.arange(start=X_set[:, 1].min(), stop=X_set[:, 1].max(), step=0.01)

X1, X2 = np.meshgrid(aranged_pc1, aranged_pc2)
plot.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.5, cmap=ListedColormap(('pink', 'yellow', 'green')))

plot.xlim(X1.min(), X1.max())
plot.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plot.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'black', 'blue'))(i), label=j)
plot.title('LDA for Boston Dataset')
plot.xlabel('LD1')
plot.ylabel('LD2')
plot.legend()
plot.show()
