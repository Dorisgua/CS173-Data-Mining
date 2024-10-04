import numpy as np
import pickle
from sklearn.linear_model import LinearRegression

# 加载数据
X = np.load('Encoding/X_scores.npy')  # 输入数据，形状为(1422, 30, 86)，表示1422份数据
y = np.load('Encoding/y_scores.npy')  # 输出数据，形状为(1422, 86)，表示1422份数据
y_grid = np.load('Encoding/y_grid.npy')

# 加载字典对象
with open("grid_to_cluster.pkl", "rb") as f:
    grid_to_cluster = pickle.load(f)

# 设定阈值
t = 0.3
correct = 0
incorrect = 0
num = []
iteration = 200

# 计算后40%的索引
split_index = int(0.4 * len(X))

# 分割数据集
X_train = X[:split_index]
y_train = y[:split_index]
X_test = X[split_index:]
y_test = y_grid[split_index:]  # 使用 y_grid 作为测试集的真实值

# 创建并拟合线性回归模型
model = LinearRegression()
model.fit(X_train.reshape(-1, X.shape[1] * X.shape[2]), y_train)
coefficients = model.coef_
print(coefficients)
# 遍历测试集
for i, (x, y_true) in enumerate(zip(X, y_grid)):
    if iteration == 0:
        break
    y_true = tuple(map(int, y_true[0].strip('()').split(', ')))
    # 将输入数据转换为3D张量，因为LSTM层期望输入为3D张量
    x_flat = x.reshape(1, -1)  # 展平 x 变量为二维数组
    # 预测
    y_pred = model.predict(x_flat)

    # 存储大于阈值的索引
    happen = np.where(y_pred > t)[1]

    # 转换为对应的 grid
    grids = [grid_to_cluster[key] for key in happen]
    grids = [point for sublist in grids for point in sublist]
    num.append(len(grids))
    print('true', y_true, 'predict', grids)
    # 与真实值比较
    if any(np.array_equal(y_true, np.array(grid)) for grid in grids):
        print(f"Prediction for sample {i + 1}: Correct")
        correct += 1
    else:
        print(f"Prediction for sample {i + 1}: Incorrect")
        incorrect += 1
    iteration -= 1

print(correct, incorrect)


# 计算平均值
average = sum(num) / len(num)
print("预测得到的格子数量:", average)