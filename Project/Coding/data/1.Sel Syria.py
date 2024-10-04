import pandas as pd
# Sift through the data from Syria
df = pd.read_csv("GTDdata.csv", encoding='ISO-8859-1')

df_200 = df[df['country'] == 200]

df_200.to_csv("GTDdata200.csv", index=False)

print("已将筛选后的数据保存为GTDdata200.csv")
