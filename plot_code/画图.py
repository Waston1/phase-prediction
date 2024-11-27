

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# x轴
x = ['Gen 0', "Gen 100", "Gen 200", "Gen 300", "Gen 400", "Gen 500"]
# y轴
y1 = [0.927, 0.946, 0.943, 0.945, 0.944, 0.941]
y2 = [0.928, 0.933, 0.938, 0.932, 0.930, 0.934]
y3 =[0.921, 0.941, 0.936, 0.943,0.935,0.934]

fig = plt.figure(figsize=(10,8),dpi=600) #画图
ax1 = fig.add_subplot(111) # 加子图
# yminorLocator = MultipleLocator(0.05)  # 将此y轴次刻度标签设置为0.1的倍数
# #设置次刻度标签的位置,没有标签文本格式
# ax1.yaxis.set_minor_locator(yminorLocator)
# plt.minorticks_on()  # 显示次刻度线
# plt.minorticks_off() #不显示次刻度线
line1 = ax1.plot(x,y1,'#FF0000',linestyle="-.",label='SVM',linewidth=2,marker='d')
line2 = ax1.plot(x,y2,'#00008B',linestyle="-.",label='RF',linewidth=2,marker='>')
line3 = ax1.plot(x,y3,'#FF8C00',linestyle="-.",label='ANN',linewidth=2,marker='<')
# 标签位置:loc   frameon:是否显示标签边框
ax1.legend(loc='best',frameon=True)
# 刻度朝向:direction (in,out,inout) , 主次刻度线:which ['major','minor'],设置刻度长宽及四周显示刻度
ax1.tick_params(direction='inout',which='major',length=5,width=2,bottom=True,top=True,left=True,right=True)
plt.tick_params(direction='inout',which='minor',length=2,bottom=True,top=True,left=True,right=True)
#设置y轴显示的刻度值
plt.xticks(['Gen 0', "Gen 100", "Gen 200", "Gen 300", "Gen 400", "Gen 500"], fontsize=16)
plt.yticks([0.90,0.91,0.92,0.92,0.93,0.94,0.95], fontsize=16)
# #设置x轴标签
plt.xlabel('Number of samples generated',fontsize=16)
# #设置y轴标签
plt.ylabel('Accuracy (%)',fontsize=16)
# plt.grid(axis='both')
plt.grid(True, which='both',color='k', linestyle='--', linewidth=1,alpha=0.1)
#b=0 or False 不显示. axis='x'只显示x轴线. axis='y'只显示y轴线
plt.show()

