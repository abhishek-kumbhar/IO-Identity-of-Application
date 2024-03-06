import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
warnings.filterwarnings("ignore")


distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
K = range(1, 5)

df = pd.read_parquet('argonne-full.parquet', engine='pyarrow')

df = df.dropna()
df.drop_duplicates(inplace=True)
df = df.iloc[:, 3:]
df = df.drop('MACHINE_NAME', axis=1)

X = df.to_numpy()

for k in K:
	print('Calculating for K = {}'.format(k))
	kmeanModel = KMeans(n_clusters=k).fit(X)
	kmeanModel.fit(X)

	distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_,
										'euclidean'), axis=1)) / X.shape[0])
	inertias.append(kmeanModel.inertia_)

	mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_,
								'euclidean'), axis=1)) / X.shape[0]
	mapping2[k] = kmeanModel.inertia_

print('\nPrinting mappings: 1\n')
for key, val in mapping1.items():
	print(f'{key} : {val}')

plt.plot(K, distortions, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Distortion')
plt.title('The Elbow Method using Distortion')
plt.show()

print('\nPrinting mappings: 2\n')
for key, val in mapping2.items():
	print(f'{key} : {val}')

plt.plot(K, inertias, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Inertia')
plt.title('The Elbow Method using Inertia')
plt.show()



