# Hierachical Clustering


import matplotlib.pyplot as plt
import pandas as pd

from scipy.cluster import hierarchy
from sklearn.cluster import AgglomerativeClustering


# Importing the mall dataset.
data = pd.read_csv('Mall_Customers.csv')
X = data.iloc[:, [3, 4]].values

# Using the dendrogram to find the optimal number of clusters.
hierarchy.dendrogram(hierarchy.linkage(X, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

# Fitting hierachical clustering to the mall dataset.
hc = AgglomerativeClustering(
    n_clusters=5, affinity="euclidean", linkage='ward'
)
y_hc = hc.fit_predict(X)

# Visualising the clusters
plt.scatter(
    X[y_hc == 0, 0], X[y_hc == 0, 1],
    s=100, c='red', label='Cluster 1'
)
plt.scatter(
    X[y_hc == 1, 0], X[y_hc == 1, 1],
    s=100, c='blue', label='Cluse 2'
)
plt.scatter(
    X[y_hc == 2, 0], X[y_hc == 2, 1],
    s=100, c='green', label='Cluse 3'
)
plt.scatter(
    X[y_hc == 3, 0], X[y_hc == 3, 1],
    s=100, c='cyan', label='Cluster 4'
)
plt.scatter(
    X[y_hc == 4, 0], X[y_hc == 4, 1],
    s=100, c='magenta', label='Cluster 5'
)
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
