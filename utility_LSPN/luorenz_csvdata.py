
import numpy as np
from scipy.integrate import solve_ivp
from tqdm import tqdm, trange
import csv

#定义洛伦兹系统的微分方程
def lorenz_system(t, xyz, sigma=10, rho=28, beta=8/3):
    x, y, z = xyz
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]

# 生成数据集
def generate_lorenz_dataset(num_samples=100000, num_steps=100, dt=0.01):
    # 时间向量
    t_span = (0, (num_steps-1) * dt)
    t_eval = np.linspace(*t_span, num_steps)

    # 初始化数据集
    X = np.zeros((num_samples, num_steps, 3))

    # 逐个生成样本
    for i in trange(num_samples, desc="training", unit="samples"):
        # 随机生成初始条件
        initial_conditions = np.random.randn(3) * 10
        
        # 解微分方程
        solution = solve_ivp(lorenz_system, t_span, initial_conditions, t_eval=t_eval)
        
        # 提取解
        x, y, z = solution.y

        # 存储数据
        X[i] = np.array([x, y, z]).T

    return X

# 生成数据集
X = generate_lorenz_dataset(num_samples=100000, num_steps=100)

# 输出数据集形状
print("数据集形状:", X.shape)
# 将数据保存到 CSV 文件
with open('lorenz_data.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['x', 'y', 'z'])  # 写入列名
    for sample in X:
        for row in sample:
            writer.writerow(row)
pass

# import numpy as np
# import csv

# # import numpy as np
# # import csv

# # 读取 CSV 文件并保持原始形状
# def read_lorenz_dataset_original_shape(file_path, num_samples=100000, num_steps=100):
#     data = np.zeros((num_samples, num_steps, 3))
#     with open(file_path, 'r', newline='') as csvfile:
#         reader = csv.reader(csvfile)
#         next(reader)  # 跳过列名行
#         for i in trange(num_samples, desc="drawing", unit="samples"):
#             for j in range(num_steps):
#                 row = next(reader)
#                 data[i, j] = [float(val) for val in row]
#     return data
# steps = 100
# # 从 CSV 文件读取数据集并保持原始形状
# X_original_shape = read_lorenz_dataset_original_shape('utility_LSPN/lorenz_data.csv',num_steps=steps)

# # 输出数据集形状
# print("读取的数据集形状（保持原始形状）:", X_original_shape.shape)
# # print(X_original_shape[2, 2, :])
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # 绘制拟合的曲线
# ax.plot(X_original_shape[-1, :, 0], X_original_shape[-1, :, 1], X_original_shape[-1, :, 2])

# # plt.plot(X_original_shape[1, :, 0], X_original_shape[1, :, 1], X_original_shape[1, :, 2])
# plt.show()
pass