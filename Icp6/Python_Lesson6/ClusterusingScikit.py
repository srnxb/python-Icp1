
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", color_codes=True)
import warnings
warnings.filterwarnings("ignore")

dataset = pd.read_csv('IRR.csv')
x = dataset.iloc[:,[1,2,3,4]]
y = dataset.iloc[:,-1]
print(x.shape, y.shape)
# see how many samples we have of each species
print(dataset["Species"].value_counts())

sns.FacetGrid(dataset, hue="Species", size=4) \
   .map(plt.scatter, "SepalLengthCm", "SepalWidthCm") \
   .add_legend()
# do same for petals
sns.FacetGrid(dataset, hue="Species", size=4) \
   .map(plt.scatter, "PetalLengthCm", "PetalWidthCm") \
   .add_legend()
plt.show()
# note that the species are nearly linearly separable with petal size,but sepal sizes are more mixed.

# but a clustering algorithm might have a hard time realizing that there were
# three separate species, which we happen to know in advance -
# usually if you're doing exploratory data analysis (EDA), you don't know this,
# e.g. if you were looking for different groups of customers.

# note that the species are nearly linearly separable with petal size,
# but sepal sizes are more mixed
#the data is unbalanced (eg sepallength ~4x petalwidth), so should do feature scaling,
# otherwise the larger features will dominate the others in clustering, etc

from sklearn import preprocessing

scaler = preprocessing.StandardScaler()

scaler.fit(x)
X_scaled_array = scaler.transform(x)
X_scaled = pd.DataFrame(X_scaled_array, columns = x.columns)

from sklearn.cluster import KMeans
nclusters = 3 # this is the k in kmeans
seed = 0
km = KMeans(n_clusters=nclusters, random_state=seed)
km.fit(X_scaled)
# predict the cluster for each data point
y_cluster_kmeans = km.predict(X_scaled)
from sklearn import metrics
score = metrics.silhouette_score(X_scaled, y_cluster_kmeans)

# note that this is the mean over all the samples - there might be some clusters
# that are well separated and others that are closer together.

# so let's look at the distribution of silhouette scores...

scores = metrics.silhouette_samples(X_scaled, y_cluster_kmeans)
sns.distplot(scores)
# # can we add the species info to that plot?
# # well, can plot them separately using pandas -
df_scores = pd.DataFrame()
df_scores['SilhouetteScore'] = scores
df_scores['Species'] = dataset['Species']
df_scores.hist(by='Species', column='SilhouetteScore', range=(0,1.0), bins=20)

wcss = []
##elbow method to know the number of clusters
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()
