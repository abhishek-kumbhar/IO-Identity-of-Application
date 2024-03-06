import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import plotly.express as px


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


scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)


kmeans = KMeans(n_clusters=3, random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

df['y_kmeans'] = y_kmeans
print('kMeans value counts are:\n')
print(df['y_kmeans'].value_counts())

