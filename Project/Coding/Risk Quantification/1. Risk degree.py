import pandas as pd
from sklearn.preprocessing import StandardScaler

file_path = "../data/GTDdata200_clustered_86_filtered_li.csv"
df = pd.read_csv(file_path, encoding='ISO-8859-1')

# Removes a row with a value of -9 (invalid) in the 'vicinity' column
df = df[df['vicinity'] != -9]
selected_rows = df

# 3. 筛选出指定列
# selected_columns = ['crit1', 'crit2', 'crit3', 'vicinity', 'extended', 'ishostkid',
#                     'suicide', 'propextent', 'nkill', 'nwound', 'success']
# selected_columns = ['attacktype1',#1武器类型
#                     'nwound', #2受伤
#                     'ishostkid',#3绑架
#                     'propextent',#4财产
#                      'targtype1',   # 5  目标类型
#                      'region',   # 6地区类型
#                     'suicide',# 7
#                     'crit2', # 8 意图恐吓
#                     'crit1', #9政治
#                     'vicinity', #10 发生在城市
#                     'crit3'#11是否超出国际人道主义法律范围
#                     ]

selected_columns = ['attacktype1', 'nwound', 'ishostkid','propextent',
                    'targtype1', 'region',
                    'suicide','crit2', 'crit1', 'vicinity', 'crit3']
filtered_df = selected_rows[selected_columns]

filtered_df.loc[filtered_df['nwound'] > 1000, 'nwound'] = 1000

means = filtered_df.mean()
filtered_df = filtered_df.fillna(means)
missing_values = filtered_df.isnull().sum()
if missing_values.any():
    print("DataFrame中存在空值。")
else:
    print("DataFrame中没有空值。")


scaler = StandardScaler()
data_scaled = scaler.fit_transform(filtered_df.T).T
print(data_scaled)
# data_scaled = concatenated_df.values

df_scaled = pd.DataFrame(data_scaled)

import numpy as np
# 假设特征权重向量
weights = np.array([0.1510, 0.1824, 0.1110,
                    0.1168, 0.0797, 0.0604,
                    0.0734, 0.0592, 0.0555,
                    0.0548, 0.0556])
scores = np.dot(data_scaled, weights)
normalized_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
print("normalized_scores")
print(normalized_scores)

normalized_scores = np.array(normalized_scores)

df['scores'] = normalized_scores

df.to_csv('../data/GTDdata200_clustered_86_filtered_li.csv', index=False)