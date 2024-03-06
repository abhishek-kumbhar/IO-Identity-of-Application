import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler  
from sklearn.cluster import KMeans



data = '/Users/abhi/Desktop/untitled folder/Live.csv'

df = pd.read_csv(data)
print(df.head())



df.drop(['Column1', 'Column2', 'Column3', 'Column4'], axis=1, inplace=True)

print(df.describe())

df.drop(['status_id', 'status_published'], axis=1, inplace=True)


X = df
y = df['status_type']

le = LabelEncoder()
X['status_type'] = le.fit_transform(X['status_type'])
y = le.transform(y)

print(X.info())


cols = X.columns
ms = MinMaxScaler()
X = ms.fit_transform(X)
X = pd.DataFrame(X, columns=[cols])

print(X.head())


kmeans = KMeans(n_clusters=2, random_state=0, n_init='auto') 
kmeans.fit(X)

labels = kmeans.labels_
correct_labels = sum(y == labels)

print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))

print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))


cs = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    cs.append(kmeans.inertia_)

plt.plot(range(1, 11), cs)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('CS')
plt.show()
