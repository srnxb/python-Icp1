from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
#import seaborn as sns
#sns.set(style="white", color_codes=True)
import warnings
warnings.filterwarnings("ignore")

train = pd.read_csv('CC.csv')
#print(pd.DataFrame(train.isnull()))
data = train.select_dtypes(include=[np.number]).interpolate().fillna(train.select_dtypes(include=[np.number]).interpolate().mean(axis=0))
#print(train.select_dtypes(include=[np.number]).interpolate().mean(axis=0))
#nulls = pd.DataFrame(data.isnull().sum().sort_values(ascending=False)[:25])
#nulls.columns = ['Null Count']
#nulls.index.name = 'Feature'
#print(nulls)
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(
                                    data, random_state=42, test_size=.33)

from sklearn import preprocessing

scaler = preprocessing.StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.decomposition import PCA
#Make an instance of the Model
pca = PCA(.95)
pca.fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

from sklearn.cluster import KMeans
nclusters = 3
seed = 0
km = KMeans(n_clusters=nclusters, random_state=seed)
km.fit(X_train)
y_cluster = km.predict(X_test)
print(km.cluster_centers_)

from sklearn import metrics
score = metrics.silhouette_score(X_test, y_cluster)

scores = metrics.silhouette_samples(X_test, y_cluster)
print(score)
print(scores)

#sns.distplot(scores)
#df_scores = pd.DataFrame()
#df_scores['SilhouetteScore'] = scores
#df_scores['tenure'] = int(X_test['TENURE'])
#df_scores.hist(by='tenure', column='SilhouetteScore', range=(0,1.0), bins=20)

wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++', max_iter=300, n_init=10,random_state=0)
    kmeans.fit(data)
    cluster_an = kmeans.predict(data)
    wcss.append(kmeans.inertia_)
#print(wcss)
plt.plot(range(1, 11), wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()

plt.scatter(y_cluster, scores, alpha=.75,
            color='b')
plt.xlabel('Cluster')
plt.ylabel('Scores')
plt.show()

plt.scatter(range(1,len(y_cluster)+1), y_cluster, alpha=.75,
            color='b')
#plt.xticks(np.arange(1, len(y_cluster), 20))
plt.xlabel('Data Point')
plt.ylabel('Cluster')
plt.show()

plt.hist(y_cluster,color="blue")
plt.xlabel('Type')
plt.ylabel('Count')
plt.title('K-Means Model')
plt.show()