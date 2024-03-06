import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import itertools

from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score as ss
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


epsilons = np.linspace(0.01, 1, num=15)
min_samples = np.arange(2, 20, step=3)

combinations = list(itertools.product(epsilons, min_samples))
N = len(combinations)

def get_scores_and_labels(combinations, X):
  scores = []
  all_labels_list = []

  for i, (eps, num_samples) in enumerate(combinations):
    dbscan_cluster_model = DBSCAN(eps=eps, min_samples=num_samples).fit(X)
    labels = dbscan_cluster_model.labels_
    labels_set = set(labels)
    num_clusters = len(labels_set)
    if -1 in labels_set:
      num_clusters -= 1
    
    if (num_clusters < 2) or (num_clusters > 50):
      scores.append(-10)
      all_labels_list.append('bad')
      c = (eps, num_samples)
      print(f"Combination {c} on iteration {i+1} of {N} has {num_clusters} clusters. Moving on")
      continue
    
    scores.append(ss(X, labels))
    all_labels_list.append(labels)
    print(f"Index: {i}, Score: {scores[-1]}, Labels: {all_labels_list[-1]}, NumClusters: {num_clusters}")

  best_index = np.argmax(scores)
  best_parameters = combinations[best_index]
  best_labels = all_labels_list[best_index]
  best_score = scores[best_index]

  return {'best_epsilon': best_parameters[0],
          'best_min_samples': best_parameters[1], 
          'best_labels': best_labels,
          'best_score': best_score}

best_dict = get_scores_and_labels(combinations, X_scaled)


df['cluster'] = best_dict['best_labels']

df.to_csv('best_dict_op.csv')
