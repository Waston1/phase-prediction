import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
# 加载数据集（这里使用的是Seaborn内置的鸢尾花数据集）
# iris = sns.load_dataset("iris")
gen_nums = ["gen 0",'gen 100', 'gen 200', 'gen 300', 'gen 400', 'gen 500']
accuracy = [ 0.928,0.933, 0.938, 0.932, 0.930,0.934]
Ath_df2 = pd.DataFrame( {'gen_nums':gen_nums,"accuracy":accuracy})
print(Ath_df2)

# 绘制柱状图
sns.barplot(x=gen_nums,y=accuracy )



sns.despine()  # 去除图例和轴线，使文本标签更清晰可见
g = sns.barplot(x = 'gen_nums',y = "accuracy",data = Ath_df2[:6],orient = 'v')
for index,row in Ath_df2[:6].iterrows():
    g.text(row.name, row.accuracy, row.accuracy, ha="center",fontsize=13)

# 添加标题和轴标签
plt.ylim(0.92, 0.94)   #设置y轴为0.9-1的刻度，不设置删除默认0-1
plt.xticks(fontsize=13)

plt.yticks(fontsize=13)
# plt.title("Iris Dataset Barplot")
plt.xlabel("gen_nums",fontsize=14)
plt.ylabel("accuracy",fontsize=14)

# 显示图形
plt.show()