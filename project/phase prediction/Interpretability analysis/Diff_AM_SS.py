import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap

# 从CSV文件读取数据
data = pd.read_csv('four_demention.csv')  # 替换为你的CSV文件路径

# 获取xyz轴数据和标签列
x = data.iloc[:, 0]
y = data.iloc[:, 1]
z = data.iloc[:, 2]
labels = data.iloc[:, 3]

# 创建绘图窗口
fig = plt.figure(dpi=600)
ax = fig.add_subplot(111, projection='3d')
cmap_custom = ListedColormap(['red', 'blue','green'])
# 绘制散点图
scatter = ax.scatter(x, y, z, c=labels, cmap=cmap_custom)

# 设置坐标轴标签
ax.set_xlabel('Delta')
ax.set_ylabel('Hmix')
ax.set_zlabel('Sid')

# 添加图例
legend = ax.legend(*scatter.legend_elements(), title="Labels")
ax.add_artist(legend)

ax.view_init(elev=10, azim=30)
# 显示图形
plt.show()