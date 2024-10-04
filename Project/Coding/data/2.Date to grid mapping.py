import json
import pandas as pd


file_path = 'GTDdata200_clustered_86_filtered_li_grid.csv'
df = pd.read_csv(file_path)

df['datetime'] = df['iyear'] * 365 + df['imonth'] * 30 + df['iday']

#按照 datetime 和 grid 进行聚类
grouped = df.groupby(['datetime', 'grid'])

#生成字典，键为 datetime，值为当天的 grid 列表
datetime_grid_dict = {}
for (datetime, grid), group in grouped:
    if datetime not in datetime_grid_dict:
        datetime_grid_dict[str(datetime)] = []
    datetime_grid_dict[str(datetime)].append(grid)

output_file = 'datetime_grid_dict.json'
with open(output_file, 'w') as f:
    json.dump(datetime_grid_dict, f)


