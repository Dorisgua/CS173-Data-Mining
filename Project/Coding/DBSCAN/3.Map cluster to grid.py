import pandas as pd
import numpy as np
import pickle

data = pd.read_csv("../data/GTDdata200_clustered_86_filtered_li.csv")

min_lat, max_lat = 32, 38
min_lon, max_lon = 35, 43
grid_size = 0.5


grid_cells = [(lat, lon) for lat in np.arange(min_lat, max_lat + grid_size, grid_size)
                           for lon in np.arange(min_lon, max_lon + grid_size, grid_size)]


data['grid'] = data.apply(lambda row: (int((row['latitude'] - min_lat) / grid_size),
                                       int((row['longitude'] - min_lon) / grid_size)), axis=1)

data.to_csv("../data/GTDdata200_clustered_86_filtered_li_grid.csv", index=False)


grid_to_cluster = {cluster: set() for cluster in data['cluster'].unique()}
for _, row in data.iterrows():
    grid_to_cluster[row['cluster']].add(row['grid'])


with open("../data/grid_to_cluster.pkl", "wb") as f:
    pickle.dump(grid_to_cluster, f)
for key, value in grid_to_cluster.items():
    print("Key:", key,"Value:", value)

