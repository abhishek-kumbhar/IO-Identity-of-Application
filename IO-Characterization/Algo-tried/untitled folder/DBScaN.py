import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN


data = pd.read_csv("Mall_customers.csv")

# print(data.head())
# print(data.tail())
# print(data.shape)

df = data.iloc[:, [3,4]]

# print(df)

plt.scatter(df.iloc[:,0], df.iloc[:,1], s=15, c= "orange")
# plt.show()


wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init = "k-means++", max_iter= 300, n_init=10)
    kmeans.fit(df)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss)
plt.title("Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
# plt.show()

dbscan = DBSCAN(eps=5, min_samples=5)
labels = dbscan.fit_predict(df)
unq = np.unique(labels)


print([labels == -1, 0])

# plt.scatter(df[labels == -1, 0], df[labels == -1, 1], s=10, c="black")
# plt.scatter(df[labels == 0, 0], df[labels == 0, 1], s=10, c="blue")
# plt.scatter(df[labels == 1, 0], df[labels == 1, 1], s=10, c="red")
# plt.scatter(df[labels == 2, 0], df[labels == 2, 1], s=10, c="green")
# plt.scatter(df[labels == 3, 0], df[labels == 3, 1], s=10, c="brown")
# plt.scatter(df[labels == 4, 0], df[labels == 4, 1], s=10, c="pink")
# plt.scatter(df[labels == 5, 0], df[labels == 5, 1], s=10, c="yellow")
# plt.scatter(df[labels == 6, 0], df[labels == 6, 1], s=10, c="silver")

# plt.xlabel('Annual Income')
# plt.ylabel('Spending Score')
# plt.show()






# print(dbscan)
# print(unq)