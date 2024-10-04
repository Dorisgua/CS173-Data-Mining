import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np


data = pd.read_csv('../data/GTDdata200_clustered.csv')

coordinates = data[['latitude', 'longitude']]

clusters = data['cluster']

kmeans = KMeans(n_clusters=len(clusters.unique())).fit(coordinates)
centroids = kmeans.cluster_centers_

wss = 0
for i in range(len(coordinates)):
    cluster_index = clusters[i]
    centroid = centroids[cluster_index]
    wss += np.linalg.norm(coordinates.iloc[i] - centroid) ** 2


bss = 0
for i in range(len(centroids)):
    for j in range(i+1, len(centroids)):
        bss += np.linalg.norm(centroids[i] - centroids[j]) ** 2


print("WSS (Within-Cluster Sum of Squares):", wss)
print("BSS (Between-Cluster Sum of Squares):", bss)
