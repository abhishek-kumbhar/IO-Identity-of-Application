import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


pd.set_option('display.float_format', '{:.10f}'.format)



from sklearn import datasets

# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn import mixture
from sklearn.cluster import DBSCAN

import warnings
warnings.filterwarnings("ignore")

iris = datasets.load_iris()
# print('Iris Keys are: ', iris.keys())
# print('\n\n')

# print('The feature names are: ', iris['feature_names'])
# print('\n\n')

# print('The target names are: ', iris['target_names'])
# print('\n\n')

# print('The target values are: ', iris['target'])
# print('\n\n')


df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# print(df.info())

# print(df['target'].value_counts())

X = df[df.columns.difference(['target'])]

print(X.columns)

cols = list(X.columns)


val = []

print(X.head())


for i in range(X.shape[0]):
    ll = []
    for j in range(X.shape[1]):
        ll.append(X[cols[j]][i])
    val.append(ll)


X1 = np.array(val).reshape(X.shape[0], X.shape[1])

X1 = X.to_numpy()

print(X1)

print('\n\n')
print('*' * 20, '--- K-Means ---', '*' * 20)

kmeans = KMeans(n_clusters=3, random_state=42)
y_kmeans = kmeans.fit_predict(X)
df['y_kmeans'] = y_kmeans
print('kMeans value counts are:\n')
print(df['y_kmeans'].value_counts())

print('*' * 20, '--- K-Means 2 ---', '*' * 20)

kmeans = KMeans(n_clusters=3, random_state=42)
y_kmeans = kmeans.fit_predict(X1)
df['y_kmeans'] = y_kmeans
print('kMeans value counts are:\n')
print(df['y_kmeans'].value_counts())


# print('\n\n')
# print('*' * 20, '--- Agglomerative Clustering ---', '*' * 20)

# hc = AgglomerativeClustering(n_clusters=3)
# y_hc = hc.fit_predict(X)
# df['y_hc'] = y_hc
# print('Agglomerative Clustering value counts are:\n')
# print(df['y_hc'].value_counts())


# print('\n\n')
# print('*' * 20, '--- GMM Clustering ---', '*' * 20)

# gmm = mixture.GaussianMixture(n_components=3, n_init=5, random_state=42)
# y_gmm = gmm.fit_predict(X)
# df['y_gmm'] = y_gmm
# print(df['y_gmm'].value_counts())


# print('\n\n')
# print('*' * 20, '--- DBSCAN Clustering ---', '*' * 20)

# dbscan = DBSCAN(eps=0.8, min_samples=5)
# y_dbscan = dbscan.fit_predict(X)
# df['y_dbscan'] = y_dbscan
# print(df['y_dbscan'].value_counts())


# print('\n\n')
# print('*' * 20, '--- Dimensionality Reduction ---', '*' * 20)

# pca = PCA(n_components=2).fit_transform(X)
# df['PCA1'] = pca[:, 0]
# df['PCA2'] = pca[:, 1]

# tsne = TSNE(n_components=2).fit_transform(X)
# df['TSNE1'] = pca[:, 0]
# df['TSNE2'] = pca[:, 1]

# print(df.head())








# grp1 = df.groupby(['target', 'y_kmeans']).size().reset_index(name='counts')
# print(grp1)

# df['y_kmeans'] = df['y_kmeans'].map({1:0, 0:1, 2:2})
# grp1 = df.groupby(['target', 'y_kmeans']).size().reset_index(name='counts')
# print(grp1)


# grp2 = df.groupby(['target', 'y_hc']).size().reset_index(name='counts')
# print(grp2)

# df['y_hc'] = df['y_hc'].map({1:0, 0:1, 2:2})
# grp2 = df.groupby(['target', 'y_hc']).size().reset_index(name='counts')
# print(grp1)



# grp3 = df.groupby(['target', 'y_gmm']).size().reset_index(name='counts')
# print(grp3)

# df['y_gmm'] = df['y_gmm'].map({1:0, 0:1, 2:2})
# grp3 = df.groupby(['target', 'y_gmm']).size().reset_index(name='counts')
# print(grp3)


# grp4 = df.groupby(['target', 'y_dbscan']).size().reset_index(name='counts')
# print(grp3)

# df['y_dbscan'] = df['y_dbscan'].map({1:0, 0:1, 2:2})
# grp4 = df.groupby(['target', 'y_dbscan']).size().reset_index(name='counts')
# print(grp4)






# df = pd.read_parquet('argonne-full.parquet', engine='pyarrow')

# df1 = df['TOTAL_MPIIO_INDEP_READS']

# y = df1.describe()

# print(y)


# print(df['NPROCS'].skew())
# print(df['NPROCS'].describe())

# print(df['NPROCS'].quantile(0.10))
# print(df['NPROCS'].quantile(0.90))

# df["NPROCS"] = np.where(df["NPROCS"] < 48.0, 48.0,df['NPROCS'])
# df["NPROCS"] = np.where(df["NPROCS"] > 16384.0, 16384.0,df['NPROCS'])
# print(df['NPROCS'].skew())

# print(df['NPROCS'].describe())



# headers = list(df.columns.values)
# headers = list(headers)

# f = open('output.txt', 'w')

# for i in headers:
#     s = str(i) + '\n'
#     f.write(s)

# f.close()