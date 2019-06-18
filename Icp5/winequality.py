import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

wine = pd.read_csv('winequality-red.csv')

# print(wine.columns)
# df['Married'] =df['Married'].astype('category').cat.codes
# print(wine.info())
# print(wine.describe())
# print(wine.corr())
df = pd.DataFrame(wine)
print('top 3 correlated columns are :\n' + str(df[df.columns[:]].corr()['quality'][:11].sort_values(ascending=False)[:3]))

# print(wine.select_dtypes(include=[np.number]).info)

# nulls = pd.DataFrame(wine.isnull().sum().sort_values(ascending=False))
# nulls.columns = ['Null Count']
# nulls.index.name  = 'Feature'
# print(nulls)

# data = wine.select_dtypes(include=[np.number]).interpolate().dropna()
# print(sum(data.isnull().sum()  != 0))

X = wine.drop('quality',axis=1)
y = wine['quality']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.2)

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)

# print(lm.intercept_)

coeff = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
# print(coeff)

predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)
# plt.show()

# print(X_test,y_test,predictions)

# print('r2 score is ',lm.score(X_test,y_test))
# print('r2 score is ',lm.score(y_test,predictions))
from sklearn.metrics import mean_squared_error
print('rmse',mean_squared_error(y_test,predictions))