'''
验证2：在多个cluster里ABC，clusterA距离近的clusterB比更加距离远的clusterC传播的多
先进行数据预处理
# 读取数据
df = pd.read_csv('../data/GTDdata200_clustered_86.csv', encoding='latin-1')

# 0、预处理数据
#去掉存在nan的行
columns_to_check = ['attacktype1', 'targtype1', 'targsubtype1', 'ransom', 'nkill', 'nwound', 'property', 'weaptype1']
df.dropna(subset=columns_to_check, inplace=True)
df['datetime'] = df['iyear'] * 365 + df['imonth'] * 30 + df['iday']

# # 使用平均值填充空值
# df.fillna(df.mean(), inplace=True)

# 归一化（除了eventid和latitude、longitude之外的列）
scaler = StandardScaler()
columns_to_normalize = ['extended', 'specificity', 'vicinity', 'crit1', 'crit2', 'crit3',
                        'success', 'suicide', 'attacktype1', 'targtype1', 'targsubtype1',
                        'nkill', 'nwound', 'property', 'propextent', 'ishostkid', 'ransom']
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

1、选取一个事件数量比较多的clusterA，选取一个发生较早的事件一，
2、找到事件一发生100天内的其他事件，print事件数量
3、如果有100天内的其他事件的cluster不和clusterA相同且多于两个cluster，继续往下；如果没有，重新选取事件一
3、选取一个事件数量比较多的clusterB，选取一个事件数量比较多的clusterC

4、计算事件一和clusterB里一百天之内的发生事件的平均相似度
计算事件一和clusterC里一百天之内的发生事件的平均相似度

5、找到clusterA的中心和clusterB的中心的距离，找到clusterA的中心和clusterC的中心的距离
'''
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from geopy.distance import geodesic
import matplotlib.pyplot as plt
# 读取数据
df = pd.read_csv('../data/GTDdata200_clustered_86_filtered_li.csv', encoding='latin-1')

# 0、预处理数据
#去掉存在nan的行
columns_to_check = ['attacktype1', 'targtype1', 'targsubtype1', 'ransom', 'nkill', 'nwound', 'property', 'weaptype1']
df.dropna(subset=columns_to_check, inplace=True)
df['datetime'] = df['iyear'] * 365 + df['imonth'] * 30 + df['iday']

# 归一化（除了eventid和latitude、longitude之外的列）
scaler = StandardScaler()
columns_to_normalize = ['extended', 'specificity', 'vicinity', 'crit1', 'crit2', 'crit3',
                        'success', 'suicide', 'attacktype1', 'targtype1', 'targsubtype1',
                        'nkill', 'nwound', 'property', 'propextent', 'ishostkid', 'ransom']
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])


# 遍历所有cluster
min_count = 3
clusterA = None
points = []
for cluster, count in df['cluster'].value_counts().items():
    # 统计每个cluster中的事件数量
    if count < min_count:
        continue
    else:
        clusterA = cluster
    # 1、选取一个事件数量比较多的clusterA
    # clusterA = df['cluster'].value_counts().idxmax()

    whether_co = 0 #如果地理和cluster有关系则加一
    num_co=0 #判断地理和cluster关系的次数

    # # 存储点的列表
    # points = []

    # 选取一个事件一
    for k in range(len(df[df['cluster'] == clusterA])):
        earliest_event = df[df['cluster'] == clusterA].sort_values('datetime').iloc[k]

        # 2、找到事件一发生100天内的其他事件，print事件数量
        events_within_100_days = df[(df['datetime'] >= earliest_event['datetime']) &
                                    (df['datetime'] <= earliest_event['datetime'] + 100)]

        # print("Events within 100 days of the earliest event:", len(events_within_100_days))
        # print(events_within_100_days)
        # 3、如果有100天内的其他事件的cluster不和clusterA相同且多于两个cluster，继续往下；如果没有，重新选取事件一
        other_clusters = events_within_100_days['cluster'].unique()
        other_clusters = other_clusters[other_clusters != clusterA]  # 去掉 clusterA
        # print(other_clusters)
        if len(other_clusters) > 2 :
            # 3、在other_clusters中随机选取一个cluster作为clusterB，选取一个不是clusterB的cluster作为clusterC

            clusterB = other_clusters[0]
            clusterC = other_clusters[1]

            # 4、计算事件一和clusterB里一百天之内的发生事件的平均相似度
            events_clusterB = events_within_100_days[events_within_100_days['cluster'] == clusterB]
            # print("events_clusterB\n",events_clusterB)
            similarities_B=[]
            main_event_features=earliest_event[
                    ['attacktype1', 'targtype1', 'targsubtype1', 'ransom', 'nkill', 'nwound', 'property', 'weaptype1']]
            for _, event in events_clusterB.iterrows():
                event_features = event[
                    ['attacktype1', 'targtype1', 'targsubtype1', 'ransom', 'nkill', 'nwound', 'property', 'weaptype1']]
                similarities_B.append(
                    cosine_similarity([main_event_features.values], [event_features.values])[0][0])
            mean_similarity_B = np.mean(similarities_B)
            # 计算事件一和clusterC里一百天之内的发生事件的平均相似度
            events_clusterC = events_within_100_days[events_within_100_days['cluster'] == clusterC]
            similarities_C=[]
            for _, event in events_clusterC.iterrows():
                event_features = event[
                    ['attacktype1', 'targtype1', 'targsubtype1', 'ransom', 'nkill', 'nwound', 'property', 'weaptype1']]
                similarities_C.append(
                    cosine_similarity([main_event_features.values], [event_features.values])[0][0])
            mean_similarity_C = np.mean(similarities_C)
            # print("Average similarity between event one and cluster B:", mean_similarity_B)
            # print("Average similarity between event one and cluster C:", mean_similarity_C)

            # 5、计算clusterA中心和clusterB中心的距离，使用df['longitude']和df['latitude']
            # 计算 Cluster A 的中心点
            clusterA_all_data = df[df['cluster'] == clusterA]
            centerA_longitude = clusterA_all_data['longitude'].mean()
            centerA_latitude = clusterA_all_data['latitude'].mean()

            # 计算 Cluster B 的中心点
            clusterB_all_data = df[df['cluster'] == clusterB]
            centerB_longitude = clusterB_all_data['longitude'].mean()
            centerB_latitude = clusterB_all_data['latitude'].mean()

            # 计算 Cluster C 的中心点
            clusterC_all_data = df[df['cluster'] == clusterC]
            centerC_longitude = clusterC_all_data['longitude'].mean()
            centerC_latitude = clusterC_all_data['latitude'].mean()

            # 计算 Cluster A 和 Cluster B 中心点之间的地理距离
            centerA_coords = (centerA_latitude, centerA_longitude)
            centerB_coords = (centerB_latitude, centerB_longitude)
            distance_geo_A_B = geodesic(centerA_coords, centerB_coords).kilometers

            # print("Geographical distance between center of Cluster A and center of Cluster B:", distance_geo_A_B, "km")

            # 计算 Cluster A 和 Cluster C 中心点之间的地理距离
            # centerA_coords = (centerA_latitude, centerA_longitude)
            centerC_coords = (centerC_latitude, centerC_longitude)
            distance_geo_A_C = geodesic(centerA_coords, centerC_coords).kilometers

            # print("Geographical distance between center of Cluster A and center of Cluster C:", distance_geo_A_C, "km")
            # break
            num_co += 1

            # 将距离和相似度存储为一个点
            points.append((distance_geo_A_B, mean_similarity_B))
            points.append((distance_geo_A_C, mean_similarity_C))

            if ((distance_geo_A_B-distance_geo_A_C)>=0 and (mean_similarity_B-mean_similarity_C)<=0)\
                    or ((distance_geo_A_B-distance_geo_A_C)<=0 and (mean_similarity_B-mean_similarity_C)>=0):
                whether_co+=1
        else:
            # print("No events within 100 days meet the criteria. Please choose another earliest event.")
            print("")
    # print(whether_co/num_co)

# 绘制散点图
x = [point[0] for point in points]  # 距离
y = [point[1] for point in points]  # 相似度
# print(points)

from scipy.optimize import curve_fit
from sklearn.linear_model import Lasso

def locally_weighted_regression(x, y, query_points, tau):
    x = np.array(x)
    y = np.array(y)
    query_points = np.array(query_points)

    m = x.shape[0]  # 样本数量
    n = query_points.shape[0]  # 查询点数量

    predictions = np.zeros(n)  # 存储预测结果

    for i in range(n):
        query_point = query_points[i]
        weights = np.exp(-(x - query_point) ** 2 / (2 * tau ** 2))  # 计算权重

        X = np.vstack((np.ones(m), x)).T
        W = np.diag(weights)
        theta = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ y

        query_point = np.hstack(([1], query_point))
        prediction = query_point @ theta
        predictions[i] = prediction

    return predictions


# 定义其他拟合函数形式
def func(x, a, b, c):
    return a * np.array(x)**2 + b * np.array(x) + c
# 定义对数函数形式
def log_func(x, a, b, c):
    return a * np.log(x) + b
# 定义幂函数形式
def power_func(x, a, b, c):
    return a * np.power(x, -b) + c
# 使用新的拟合函数拟合曲线
popt, pcov = curve_fit(func, x, y)
# 示例数据
# x = [1, 2, 3, 4, 5]
# y = [2, 3, 4, 5, 6]
query_points = np.arange(501)  # 查询点范围：0到500
tau = 10

# 绘制散点图
plt.scatter(x, y, label='Data')

plt.scatter(x, y)
# 预测
predictions = locally_weighted_regression(x, y, query_points, tau)
# 绘制折线图
plt.plot(query_points, predictions, label='Predictions', color='r')
plt.xlabel('Distance between clusters (km)')
plt.ylabel('Similarity')
plt.title('Relationship between Distance and Similarity')
# plt.title('Locally Weighted Linear Regression Predictions')
plt.legend()
# plt.ylim(0.6, 1)  # 限制 y 轴范围在 0 到 1
plt.grid(True)
plt.show()
