# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np
#
# df = pd.read_csv(r'./IM_SS_AM2.csv')
#
# feature=df.iloc[:,:]
#
# correlation_matrix = feature.corr()
# fig=plt.figure(figsize=(11, 10))
# heatmap=sns.heatmap(correlation_matrix, annot=True, cmap="magma", linewidths=0.5)
# plt.title("Feature correlation heatmap")
# plt.xticks(fontsize=18)  # 设置x轴刻度标签字体大小
# plt.yticks(fontsize=18)  # 设置y轴刻度标签字体大小
# # fig.subplots_adjust(top=0.9)
# plt.show()
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# import matplotlib.font_manager
# # 设置字体为Arial
# font_list = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
# for font in font_list:
#     print(font)
# plt.rcParams['font.sans-serif'] = ['C:/Windows/Fonts/Arial.ttf']  # 指定中文字体为宋体
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

df = pd.read_csv(r'./IM_SS_AM2.csv')

feature = df.iloc[:,:]

correlation_matrix = feature.corr()
fig = plt.figure(figsize=(11, 10))
heatmap = sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm",square=True, linewidths=0.5, annot_kws={"size": 10,"weight": "bold"},cbar_kws={"shrink": 0.8})
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=16, width=2)
plt.title("Feature correlation heatmap", fontsize=16,fontweight='bold')
plt.xticks(fontsize=14,fontweight='bold',rotation=60)
plt.yticks(fontsize=14,fontweight='bold',rotation=30)
plt.show()