import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from tqdm import tqdm

# 读取event_pairs.csv文件
event_pairs_df = pd.read_csv('../data/event_pairs.csv')

# 读取vector200.csv文件
vector_df = pd.read_csv('../data/GTDdata200_features.csv')

# 存储时间差和相似度对的列表
time_similarity_pairs = []

# 计算总共需要迭代的次数
total_iterations = len(range(0, 501, 5))

# 使用tqdm显示进度条
with tqdm(total=total_iterations) as pbar:
    # 选择时间差为0、5、10、15、20、25……500的事件对
    for time_diff in range(0, 501, 5):
        # 筛选时间差为time_diff的事件对
        filtered_pairs = event_pairs_df[event_pairs_df['datetime'] == time_diff]

        # 遍历每个事件对
        for _, pair in filtered_pairs.iterrows():
            # 找到事件对在vector200.csv中的对应行
            event1_row = vector_df[vector_df['eventid'] == pair['eventid1']].iloc[0]
            event2_row = vector_df[vector_df['eventid'] == pair['eventid2']].iloc[0]

            # 提取除了eventid和datetime之外的特征
            event1_features = event1_row.drop(['eventid', 'datetime'])
            event2_features = event2_row.drop(['eventid', 'datetime'])

            # 计算余弦相似度
            similarity = cosine_similarity([event1_features.values], [event2_features.values])[0][0]

            # 存储时间差和相似度对
            time_similarity_pairs.append([time_diff, similarity])
        pbar.update(1)  # 更新进度条

# 转换为DataFrame
time_similarity_df = pd.DataFrame(time_similarity_pairs, columns=['Time Difference', 'Similarity'])

# 存储为CSV文件
# time_similarity_df.to_csv('time_similarity_pairs.csv', index=False)

from matplotlib.colors import LinearSegmentedColormap
# Custom colormap

# Define custom colors
light_blue = (0.6, 0.8, 1.0)
red = (1, 0, 0)  # Red color
# colors = [(0, 'blue'), (1, 'red')]
# Custom colormap
colors = [(0, light_blue), (1, red)]
cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)

# Set the figure size
plt.figure(figsize=(16, 6))
# Convert scatter plot to heatmap
plt.subplot(1, 2, 1)
plt.hexbin(time_similarity_df['Time Difference'], time_similarity_df['Similarity'], gridsize=20, cmap=cmap, reduce_C_function=np.mean)
plt.colorbar(label='Mean Similarity')
plt.title('Time Difference vs Similarity (Event Pairs)')
plt.xlabel('Time Difference')
plt.ylabel('Similarity')
plt.grid(True)

# 设置纵坐标轴范围
plt.ylim(0.9993, 1)
# 设置横坐标轴范围
plt.xlim(time_similarity_df['Time Difference'].min(), time_similarity_df['Time Difference'].max())
# 计算每个时间差下的平均相似度
mean_similarity_df = time_similarity_df.groupby('Time Difference')['Similarity'].mean().reset_index()

# 绘制第二张散点图
plt.subplot(1, 2, 2)
plt.scatter(mean_similarity_df['Time Difference'], mean_similarity_df['Similarity'], marker='o', color='r', alpha=0.5)
plt.title('Time Difference vs Mean Similarity')
plt.xlabel('Time Difference')
plt.ylabel('Mean Similarity')
plt.grid(True)

plt.tight_layout()
plt.show()


'''读入event_pairs.csv，选择时间差为0、5、10、15、20、25……500的事件对
	找到其在vector200.csv中对应的行
	计算两者去掉eventid和datetime之后的余弦相似度
	存储为[时间差、相似度]对
绘制[时间差、相似度]散点图'''