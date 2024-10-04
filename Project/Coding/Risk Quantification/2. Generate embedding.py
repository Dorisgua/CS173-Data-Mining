import numpy as np
import pandas as pd
import json

distance_df = pd.read_csv('../data/distance_edge.csv')

input_file = '../data/datetime_grid_dict.json'
with open(input_file, 'r') as f:
    datetime_grid_dict = json.load(f)

# 读取每个地点每天的事件数量信息
df = pd.read_csv('../data/GTDdata200_clustered_86_filtered_li.csv')
df = df[df['cluster'] != -1]
df.dropna(subset=['latitude', 'longitude'], inplace=True)
df['datetime'] = df['iyear'] * 365 + df['imonth'] * 30 + df['iday']
grouped = df.groupby(['datetime', 'cluster'])['scores'].sum().unstack(fill_value=0)  # 修改此行为对'scores'列求和

# 确定时间序列长度
seq_length = 30

num_clusters = len(grouped.columns)
num_edges = len(distance_df)
adj_matrix = np.zeros((num_clusters, num_clusters))
edge_weights = np.zeros((num_clusters, num_clusters))

for i, row in distance_df.iterrows():
    cluster1 = int(row['cluster1'])
    cluster2 = int(row['cluster2'])
    distance = row['1/std_distance']
    edge_weights[cluster1][cluster2] = distance
    edge_weights[cluster2][cluster1] = distance


# np.save('edge_weights.npy', edge_weights)
X = []
y = []
y_grid = []
for i in range(len(grouped) - seq_length):
    X.append(grouped.values[i:i+seq_length])
    y.append(grouped.values[i+seq_length])
    y_grid.append(datetime_grid_dict[str(grouped.index[i+seq_length])])
X = np.array(X)
y = np.array(y)
y_grid = np.array(y_grid)
np.save('../embedding/X_scores.npy', X)
np.save('../embedding/y_scores.npy', y)
np.save('../embedding/y_grid.npy', y_grid)



# 生成时间衰减的权重
decay_weights = np.exp(-0.01 * (30 - np.arange(30)))

for i in range(len(X)):
    a = decay_weights[:, np.newaxis]
    X[i] *= decay_weights[:, np.newaxis]
np.save('../embedding/X_decay.npy', X)

