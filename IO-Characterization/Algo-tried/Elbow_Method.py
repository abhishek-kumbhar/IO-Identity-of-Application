import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings("ignore")


df = pd.read_parquet('/Users/abhi/Desktop/IO-Characterization/Dataset/argonne-full.parquet')

df = df.dropna()
df.drop_duplicates(inplace=True)

columns_to_drop = ['COBALT_JOBID', 'DARSHAN_LIB_VERSION', 'DARSHAN_LOG_VERSION', 'END_TIME', 'EXE_NAME_GENID', 'MACHINE_NAME', 'RUN_DATE_ID', 'START_TIME', 'TOTAL_MPIIO_COLL_READS',
                   'TOTAL_MPIIO_COLL_WRITES', 'TOTAL_READ_OPS', 'TOTAL_WRITE_OPS']

df = df.drop(columns_to_drop, axis=1)

print('Final Shape of Data            : ', df.shape)

cols = list(df.columns)
Q1 = df[cols].quantile(0.25)
Q3 = df[cols].quantile(0.75)
IQR = Q3 - Q1

df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

print('Final Shape of Data (Quantiled): ', df.shape)


X = df.to_numpy()


distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
K = range(2, 15)

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


