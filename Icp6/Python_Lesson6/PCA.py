from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn import metrics, datasets
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.decomposition import PCA

# You can add the parameter data_home to wherever to where you want to download your data
dataset = pd.read_csv('IRR.csv')
x = dataset.iloc[:,[1,2,3,4]]
y = dataset.iloc[:,-1]
# test_size: what proportion of original data is used for test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.33, random_state=0)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fit on training set only.
scaler.fit(x_train)

# Apply transform to both the training set and the test set.
train_img = scaler.transform(x_train)
test_img = scaler.transform(x_test)

pca = PCA(2)
train_img = pca.fit_transform(train_img)
test_img = pca.fit_transform(test_img)
df2 = pd.DataFrame(data=train_img)
finaldf = pd.concat([df2,dataset[['Species']]],axis=1)
print(finaldf)
# Returns a NumPy Array
# Predict for One Observation (image)
# logisticRegr.predict(test_img[0].reshape(1,-1))
#
# # Predict for Multiple Observations (images) at Once
# logisticRegr.predict(test_img[0:10])
# score = logisticRegr.score(test_img, test_lbl)
# print(score)