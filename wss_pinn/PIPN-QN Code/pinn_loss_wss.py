import pandas as pd  # 导入Pandas库
import numpy as np  # 导入NumPy库
import os  # 导入操作系统接口模块
import re  # 导入正则表达式模块
import pickle  # 导入Pickle模块
import pdb  # 导入Python调试器
import matplotlib.pyplot as plt  # 导入绘图库
from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # 导入子图定位工具
from matplotlib.patches import ConnectionPatch  # 导入连接线工具
# 设置坐标轴刻度线方向向内
plt.rcParams['xtick.direction'] = 'in'  
plt.rcParams['ytick.direction'] = 'in'  
# 设置字体
font1 = {'family' : 'Times New Roman','weight' : 'normal','size' : 15,'frameon':'False', 'borderpad':0.1, 'labelspacing':0.1}

# 设置matplotlib配置
config = {
    "font.family":'serif',
    "font.size": 12,
    "mathtext.fontset":'stix',
    "font.serif": ['Times New Roman'],
    'axes.unicode_minus': False
}
from matplotlib import rcParams
rcParams.update(config)

# 定义文件路径
path0='weight/history_1500.csv'
path='weight/history_1500_res.csv'

def read_csv(fn):
    """
    读取CSV文件数据
    
    参数:
        fn: 文件路径
    
    返回:
        data: 读取的数据数组
    """
    data = []
    for i,line in enumerate(open(fn)):
        try:
            nd = line.split(',')
            _ = float(nd[0])+float(nd[2])
            data.append(nd)
        except:
            continue
    data = np.array(data,dtype='float64')
    return data

# 读取损失数据
loss_data=read_csv(path0)
loss_res=read_csv(path)

# 提取总损失数据
total_res = loss_res[:,0]
total = loss_data[:,0]

# 提取方程损失数据
equ_res = loss_res[:,3]
equ = loss_data[:,8]

# 设置绘图参数
j=0.75

# 提取边界条件损失数据
bc_res = loss_res[:,5]
bc = loss_data[:,9]

# 创建训练轮数序列
epochs = range(1,len(total)+1)

# 创建图形和子图
figss=plt.figure(figsize = (10.5,3.5))
axss=figss.add_subplot(1,3,1)
# 绘制总损失曲线
axss.plot(epochs,total,'b',label='PINN',linewidth=j)
axss.plot(epochs,total_res,'r',label='rPINN',linewidth=j)
axss.legend(frameon=False, borderpad=0.1, labelspacing=0.1,fontsize = 12)
axss.set_yscale('log')
axss.set_xlabel('Epochs',fontsize = 15)
labels = ['2E+1','1.2E+1','0.8E+1','0.5E+1','0.3E+1']
axss.set_yticks([2e+1,1.3e+1,0.8e+1,0.5e+1,0.3e+1],labels)
# 设置标题
bold_font = {'weight': 'bold'}
axss.set_title('Total loss',fontsize = 20, fontdict=bold_font)

# 绘制方程损失子图
axss=figss.add_subplot(1,3,2)
axss.plot(epochs,equ,'b',label='PINN',linewidth=j)
axss.plot(epochs,equ_res,'r',label='rPINN',linewidth=j)
axss.legend(frameon=False, borderpad=0.1, labelspacing=0.1,fontsize = 12)
axss.set_yscale('log')
axss.set_xlabel('Epochs',fontsize = 15)
labels = ['0.5E+1','0.2E+1','8E-1','3E-1','1E-1']
axss.set_yticks([0.5e+1,0.2e+1,8e-1,3e-1,1e-1],labels)
axss.set_title('Eqn loss',fontsize = 20, fontdict=bold_font)

# 绘制边界条件损失子图
axss=figss.add_subplot(1,3,3)
axss.plot(epochs,bc,'b',label='PINN',linewidth=j)
axss.plot(epochs,bc_res,'r',label='rPINN',linewidth=j)
axss.legend(frameon=False, borderpad=0.1, labelspacing=0.1,fontsize = 12)
axss.set_yscale('log')
labels = ['1E-2','3E-3','1E-3','3E-4','1E-4']
axss.set_yticks([1E-2,3E-3,1E-3,3E-4,1E-4],labels)
axss.set_xlabel('Epochs',fontsize = 15)
axss.set_title('BC loss',fontsize = 20, fontdict=bold_font)

# 调整子图布局
plt.tight_layout()
plt.show()

# 定义更多文件路径
path0='weight/history_0.05.csv'
path2='weight/history_0.25.csv'
path3='weight/history_0.5.csv'
path4='weight/history_0.5_0.75.csv'
path5='weight/history_0.75_0.5.csv'
path6='weight/history_0.75.csv'
path7='weight/history_1_0.5.csv'
path8='weight/history_1.csv'

# 读取更多损失数据
loss0=read_csv(path0)
loss2=read_csv(path2)
loss3=read_csv(path3)
loss4=read_csv(path4)
loss5=read_csv(path5)
loss6=read_csv(path6)
loss7=read_csv(path7)
loss8=read_csv(path8)

# 创建新的图形
fig1=plt.figure(figsize = (8,7))
axss=fig1.add_subplot(3,3,1)
# 绘制不同参数下的损失曲线
axss.plot(epochs,loss0[:,-2],'r',label=r'$\alpha$'+' = 0.05',linewidth=j)
axss.plot(epochs,loss0[:,-1],'b',label=r'$\beta$'+' = 0.05',linewidth=j)
axss.yaxis.set_major_formatter('{:.2f}'.format)
axss.set_yticks([0.3,0.6,0.9,1.2])
axss.set_xticks([0,500,1000,1500])
axss.set_title(r'$\alpha$'+' = 0.05'+'\t'+r'$\beta$'+' = 0.05',fontsize = 15)

axss=fig1.add_subplot(3,3,2)
axss.plot(epochs,loss_res[:,-2],'r',label=r'$\alpha$'+' = 0.1',linewidth=j)
axss.plot(epochs,loss_res[:,-1],'b',label=r'$\beta$'+' = 0.1',linewidth=j)
axss.yaxis.set_major_formatter('{:.2f}'.format)
axss.set_yticks([0.3,0.6,0.9,1.2])
axss.set_xticks([0,500,1000,1500])
axss.set_title(r'$\alpha$'+' = 0.1'+'\t'+r'$\beta$'+' = 0.1',fontsize = 15)

axss=fig1.add_subplot(3,3,3)
axss.plot(epochs,loss2[:,-2],'r',label=r'$\alpha$'+' = 0.25',linewidth=j)
axss.plot(epochs,loss2[:,-1],'b',label=r'$\beta$'+' = 0.25',linewidth=j)
axss.yaxis.set_major_formatter('{:.2f}'.format)
axss.set_yticks([0.3,0.6,0.9,1.2])
axss.set_xticks([0,500,1000,1500])
axss.set_title(r'$\alpha$'+' = 0.25'+'\t'+r'$\beta$'+' = 0.25',fontsize = 15)

axss=fig1.add_subplot(3,3,4)
axss.plot(epochs,loss3[:,-2],'r',label=r'$\alpha$'+' = 0.5',linewidth=j)
axss.plot(epochs,loss3[:,-1],'b',label=r'$\beta$'+' = 0.5',linewidth=j)
axss.yaxis.set_major_formatter('{:.2f}'.format)
axss.set_yticks([0.3,0.6,0.9,1.2])
axss.set_xticks([0,500,1000,1500])
axss.set_title(r'$\alpha$'+' = 0.5'+'\t'+r'$\beta$'+' = 0.5',fontsize = 15)

axss=fig1.add_subplot(3,3,5)
axss.plot(epochs,loss4[:,-2],'r',label=r'$\alpha$'+' = 0.5',linewidth=j)
axss.plot(epochs,loss4[:,-1],'b',label=r'$\beta$'+' = 0.75',linewidth=j)
axss.yaxis.set_major_formatter('{:.2f}'.format)
axss.set_yticks([0.3,0.6,0.9,1.2])
axss.set_xticks([0,500,1000,1500])
axss.set_title(r'$\alpha$'+' = 0.5'+'\t'+r'$\beta$'+' = 0.75',fontsize = 15)

axss=fig1.add_subplot(3,3,6)
axss.plot(epochs,loss5[:,-2],'r',label=r'$\alpha$'+' = 0.75',linewidth=j)
axss.plot(epochs,loss5[:,-1],'b',label=r'$\beta$'+' = 0.5',linewidth=j)
axss.yaxis.set_major_formatter('{:.2f}'.format)
axss.set_yticks([0.3,0.6,0.9,1.2])
axss.set_xticks([0,500,1000,1500])
axss.set_title(r'$\alpha$'+' = 0.75'+'\t'+r'$\beta$'+' = 0.5',fontsize = 15)

axss=fig1.add_subplot(3,3,7)
axss.plot(epochs,loss6[:,-2],'r',label=r'$\alpha$'+' = 0.75',linewidth=j)
axss.plot(epochs,loss6[:,-1],'b',label=r'$\beta$'+' = 0.75',linewidth=j)
axss.yaxis.set_major_formatter('{:.2f}'.format)
axss.set_yticks([0.3,0.6,0.9,1.2])
axss.set_xticks([0,500,1000,1500])
axss.set_title(r'$\alpha$'+' = 0.75'+'\t'+r'$\beta$'+' = 0.75',fontsize = 15)
axss.set_xlabel('Epochs',fontsize = 15)
fig1.savefig('alpha.jpg', dpi= 600)

axss=fig1.add_subplot(3,3,8)
axss.plot(epochs,loss7[:,-2],'r',label=r'$\alpha$'+' = 1.0',linewidth=j)
axss.plot(epochs,loss7[:,-1],'b',label=r'$\beta$'+' = 0.5',linewidth=j)
axss.yaxis.set_major_formatter('{:.2f}'.format)
axss.set_yticks([0.3,0.6,0.9,1.2])
axss.set_xticks([0,500,1000,1500])
axss.set_title(r'$\alpha$'+' = 1.0'+'\t'+r'$\beta$'+' = 0.5',fontsize = 15)
axss.set_xlabel('Epochs',fontsize = 15)

axss=fig1.add_subplot(3,3,9)
axss.plot(epochs,loss8[:,-2],'r',label=r'$\alpha$'+' = 1.0',linewidth=j)
axss.plot(epochs,loss8[:,-1],'b',label=r'$\beta$'+' = 1.0',linewidth=j)
axss.yaxis.set_major_formatter('{:.2f}'.format)
axss.set_yticks([0.3,0.6,0.9,1.2])
axss.set_xticks([0,500,1000,1500])
axss.set_xlabel('Epochs',fontsize = 15)
axss.set_title(r'$\alpha$'+' = 1.0'+'\t'+r'$\beta$'+' = 1.0',fontsize = 18)
plt.tight_layout()
fig1.savefig('alpha.jpg', dpi= 600)

# 创建另一个新的图形
fig1=plt.figure(figsize = (6,5))
axss=fig1.add_subplot(1,1,1)
# 绘制不同参数组合下的总损失曲线
axss.plot(epochs,loss0[:,0],'b',label=r'$\alpha = 0.05$'+'\t'+r'$\beta = 0.05$',linewidth=j,ls='-')
axss.plot(epochs,loss_res[:,0],'pink',label=r'$\alpha = 0.1$'+'\t'+r'$\beta = 0.1$',linewidth=j,ls='-')
axss.plot(epochs,loss2[:,0],'r',label=r'$\alpha = 0.25$'+'\t'+r'$\beta = 0.25$',linewidth=j,ls='-')
axss.plot(epochs,loss3[:,0],'y',label=r'$\alpha = 0.5$'+'\t'+r'$\beta = 0.5$',linewidth=j,ls='-')
axss.plot(epochs,loss4[:,0],'g',label=r'$\alpha = 0.5$'+'\t'+r'$\beta = 0.75$',linewidth=j,ls='-')
axss.plot(epochs,loss5[:,0],'m',label=r'$\alpha = 0.75$'+'\t'+r'$\beta = 0.5$',linewidth=j,ls='-')
axss.plot(epochs,loss6[:,0],'c',label=r'$\alpha = 0.75$'+'\t'+r'$\beta = 0.75$',linewidth=j,ls='-')
axss.plot(epochs,loss7[:,0],'k',label=r'$\alpha = 1.0$'+'\t'+r'$\beta = 0.5$',linewidth=j,ls='-')
axss.plot(epochs,loss8[:,0],'orange',label=r'$\alpha = 1.0$'+'\t'+r'$\beta = 1.0$',linewidth=j,ls='-')
axss.legend(frameon=False, borderpad=0.1, labelspacing=0.1,fontsize = 18)
axss.set_xlabel('Epochs',fontsize = 18)
axss.set_yscale('log')
labels = [r'20',r'$13$',r'$8$',r'$5$',r'$3$']
axss.set_yticks([2e+1,1.3e+1,0.8e+1,0.5e+1,0.3e+1],labels)
axss.set_xticks([0,250,500,750,1000,1250,1500])
fig1.tight_layout()
fig1.savefig('beta.jpg', dpi= 600)
plt.show()

# 定义更多文件路径
path0='weight/history_50.csv'
path2='weight/history_100.csv'
path3='weight/history_250.csv'
path4='weight/history_500.csv'
path5='weight/history_1500_res.csv'
path6='weight/history_3000.csv'
path7='weight/history_1000.csv'

# 读取更多损失数据
loss0=read_csv(path0)
loss2=read_csv(path2)
loss3=read_csv(path3)
loss4=read_csv(path4)
loss5=read_csv(path5)
loss6=read_csv(path6)
loss7=read_csv(path7)

# 创建最后一个图形
fig1=plt.figure(figsize = (6,5))
axss=fig1.add_subplot(1,1,1)
# 绘制不同点数下的损失曲线
axss.plot(epochs,loss0[:,0],'b',label='N = 50',linewidth=j,ls='-')
axss.plot(epochs,loss2[:,0],'r',label='N = 100',linewidth=j,ls='-')
axss.plot(epochs,loss3[:,0],'y',label='N = 250',linewidth=j,ls='-')
axss.plot(epochs,loss4[:,0],'g',label='N = 500',linewidth=j,ls='-')
axss.plot(epochs,loss7[:,0],'m',label='N = 1000',linewidth=j,ls='-')
axss.plot(epochs,loss5[:,0],'c',label='N = 1500',linewidth=j,ls='-')
axss.plot(epochs,loss6[:,0],'k',label='N = 3000',linewidth=j,ls='-')
axss.legend(frameon=False, borderpad=0.1, labelspacing=0.1,fontsize = 18)
axss.set_xlabel('Epochs',fontsize = 18)
axss.set_yscale('log')
labels = [r'20',r'$13$',r'$8$',r'$5$',r'$3$']
axss.set_yticks([2e+1,1.3e+1,0.8e+1,0.5e+1,0.3e+1],labels)
axss.set_xticks([0,250,500,750,1000,1250,1500])
fig1.tight_layout()
plt.show()
fig1.savefig('N_num.jpg', dpi= 600)