import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.spatial.distance import cdist
from functools import reduce

from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")
pd.set_option('display.float_format', '{:.10f}'.format)


df = pd.read_parquet('argonne-full.parquet', engine='pyarrow')

# print('Original Shape of Data: ', df.shape)


# df1 = df.iloc[12345:12345+5, :]
# df1 = df1.T

# df1.to_excel('transpose1.xls')


columns_to_drop = ['COBALT_JOBID', 'DARSHAN_LIB_VERSION', 'DARSHAN_LOG_VERSION', 'END_TIME', 'EXE_NAME_GENID', 'MACHINE_NAME', 'RUN_DATE_ID', 'START_TIME', 'TOTAL_MPIIO_COLL_READS',
                   'TOTAL_MPIIO_COLL_WRITES', 'TOTAL_READ_OPS', 'TOTAL_WRITE_OPS']

df = df.drop(columns_to_drop, axis=1)

cols = list(df.columns)



Q1 = df[cols].quantile(0.25)
Q3 = df[cols].quantile(0.75)
IQR = Q3 - Q1

print('shape before outliers removal: ', df.shape)

df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

print('shape after outliers removal: ', df.shape)


columns_to_consider = ['NPROCS', 'RUN_TIME', 'TOTAL_IO_TIME', 'TOTAL_POSIX_F_META_TIME', 'TOTAL_POSIX_F_READ_TIME', 'TOTAL_POSIX_F_WRITE_TIME', 'TOTAL_IO_PER_PROC', 'TOTAL_IO_OPS', 
        'TOTAL_MD_OPS', 'TOTAL_READ_TIME', 'TOTAL_WRITE_TIME', 'TOTAL_MD_TIME']


# columns_to_consider = df.columns

y = df[columns_to_consider]

print('shape of considered columns: ', y.shape)


ll = []

for i in columns_to_consider:
    x = y[(y[i] < y[i].quantile(0.25))]
    ll.append(sorted(list(x.index)))

dIdx = sorted(list(reduce(lambda i, j: i & j, (set(x) for x in ll))))

idx1 = list(y.index)
idx2 = dIdx

rIdx = sorted(list(set(idx1) - set(idx2)))

debugApps = df.loc[dIdx]
realApps = df.loc[rIdx]

df = realApps

# distortions = []
# inertias = []
# mapping1 = {}
# mapping2 = {}
# K = range(1, 10)

# X = debugApps.to_numpy()

# for k in K:
# 	print('Calculating for K = {}'.format(k))
# 	kmeanModel = KMeans(n_clusters=k).fit(X)
# 	kmeanModel.fit(X)

# 	distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_,
# 										'euclidean'), axis=1)) / X.shape[0])
# 	inertias.append(kmeanModel.inertia_)

# 	mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_,
# 								'euclidean'), axis=1)) / X.shape[0]
# 	mapping2[k] = kmeanModel.inertia_

# print('\nPrinting mappings: 1\n')
# for key, val in mapping1.items():
# 	print(f'{key} : {val}')

# plt.plot(K, distortions, 'bx-')
# plt.xlabel('Values of K')
# plt.ylabel('Distortion')
# plt.title('The Elbow Method using Distortion')
# plt.show()

# print('\nPrinting mappings: 2\n')
# for key, val in mapping2.items():
# 	print(f'{key} : {val}')

# plt.plot(K, inertias, 'bx-')
# plt.xlabel('Values of K')
# plt.ylabel('Inertia')
# plt.title('The Elbow Method using Inertia')
# plt.show()



X = df.to_numpy()



kmeans = KMeans(n_clusters=3, random_state=42)
y_kmeans = kmeans.fit_predict(X)



df['y_kmeans'] = y_kmeans
print('kMeans value counts are:\n')
print(df['y_kmeans'].value_counts())

u_labels = np.unique(y_kmeans)

# print(df[y_kmeans == 0])

df = df.values

for i in u_labels:
    plt.scatter(df[y_kmeans == i , 0] , df[y_kmeans == i , 1] , label = i)
plt.legend()
plt.show()



