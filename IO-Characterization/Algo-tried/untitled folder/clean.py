import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

from sklearn.cluster import AgglomerativeClustering
from sklearn import mixture
from sklearn.cluster import DBSCAN


warnings.filterwarnings("ignore")
pd.set_option('display.float_format', '{:.10f}'.format)

df = pd.read_parquet('argonne-full.parquet', engine='pyarrow')

print(df.shape)
df = df.dropna()
print(df.shape)
df.drop_duplicates(inplace=True)
print(df.shape)


df = df.iloc[:, 3:]
df = df.drop('MACHINE_NAME', axis=1)

cols = list(df.columns)

X = df.to_numpy()

print('shape', X.shape)

kmeans = KMeans(n_clusters=3, random_state=42)
y_kmeans = kmeans.fit_predict(X)
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

# gmm = mixture.GaussianMixture(random_state=42)
# y_gmm = gmm.fit_predict(X)
# df['y_gmm'] = y_gmm
# print(df['y_gmm'].value_counts())


# print('\n\n')
# print('*' * 20, '--- DBSCAN Clustering ---', '*' * 20)

# dbscan = DBSCAN(eps=0.8, min_samples=5)
# y_dbscan = dbscan.fit_predict(X)
# df['y_dbscan'] = y_dbscan
# print(df['y_dbscan'].value_counts())

# X1 = np.array(val).reshape(df.shape[0], df.shape[1])


# print(X1.shape)



distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
K = range(1, 15)

# for k in K:
# 	# Building and fitting the model
# 	kmeanModel = KMeans(n_clusters=k).fit(X)
# 	kmeanModel.fit(X)

# 	distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_,
# 										'euclidean'), axis=1)) / X.shape[0])
# 	inertias.append(kmeanModel.inertia_)

# 	mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_,
# 								'euclidean'), axis=1)) / X.shape[0]
# 	mapping2[k] = kmeanModel.inertia_

# for key, val in mapping1.items():
# 	print(f'{key} : {val}')

# plt.plot(K, distortions, 'bx-')
# plt.xlabel('Values of K')
# plt.ylabel('Distortion')
# plt.title('The Elbow Method using Distortion')
# plt.show()

# for key, val in mapping2.items():
# 	print(f'{key} : {val}')

# plt.plot(K, inertias, 'bx-')
# plt.xlabel('Values of K')
# plt.ylabel('Inertia')
# plt.title('The Elbow Method using Inertia')
# plt.show()






# Q1 = df.quantile(0.25)
# Q3 = df.quantile(0.75)
# IQR = Q3 - Q1



# for i in range(len(cols)):
#     # print(cols[i], ': {:.3f}'.format(df[cols[i]].skew()), end='  >>>>  ')
#     q1 = df[cols[i]].quantile(0.10)
#     q2 = df[cols[i]].quantile(0.90)
    
#     df[cols[i]] = np.where(df[cols[i]] < q1, q1, df[cols[i]])
#     df[cols[i]] = np.where(df[cols[i]] > q2, q2, df[cols[i]])
#     # print('{:.3f}'.format(df[cols[i]].skew()))


# kmeans = KMeans()
# y_kmeans = kmeans.fit_predict(df)
# df['y_kmeans'] = y_kmeans
# print('kMeans value counts are:\n')
# print(df['y_kmeans'].value_counts())

