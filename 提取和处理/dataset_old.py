"""
数据集处理模块
用于处理CFD仿真数据，准备PINN神经网络的训练和测试数据
"""
import pandas as pd  # 导入Pandas库，用于数据处理和分析
import os  # 导入操作系统接口模块，用于文件和目录操作
from torch.utils.data import Dataset, DataLoader  # 导入PyTorch数据集相关模块，用于创建数据集和数据加载器
import numpy as np  # 导入NumPy库，用于数值计算和数组操作
import pickle  # 导入Pickle模块，用于序列化和反序列化Python对象
import matplotlib.pyplot as plt  # 导入绘图库，用于数据可视化
# 设置随机种子，确保结果可重现
np.random.seed(1234)

def read_ti(fn):
    """
    读取ti格式文件数据
    
    参数:
        fn: 文件路径
    
    返回:
        data: 读取的数据数组
    """
    data = []  # 初始化空列表存储数据
    with open(fn) as f:  # 打开文件
        file = f.readlines()  # 读取所有行
        for line in file:  # 遍历每一行
            line = line.strip()  # 去除行首行尾空白字符
            dd = line.split()  # 按空白字符分割行
            temp = list(dd)  # 转换为列表
            data.append(temp)  # 添加到数据列表
    data = np.delete(data, 0, axis=0)  # 删除第一行（通常是标题行）
    data = np.array(data, dtype='float64')  # 转换为浮点数数组
    return data

def data_cut_point(datesets, idx, point):
    """
    数据切割点函数，用于控制数据集大小
    
    参数:
        datesets: 数据集
        idx: 索引
        point: 点数
    
    返回:
        处理后的数据集
    """
    if idx.shape[0] >= point:  # 如果索引数量大于等于指定点数
        idx = idx[:point]  # 截取前point个索引
        datesets= np.delete(datesets, idx, axis= 0)  # 删除这些索引对应的数据
    else:  # 如果索引数量小于指定点数
        datesets = np.delete(datesets, idx, axis= 0)  # 先删除这些索引对应的数据
        length_model = datesets.shape[0]  # 获取剩余数据数量
        shuffled_indices = np.random.choice(length_model, point-idx.shape[0], replace=False)  # 随机选择补充点
        datesets = np.delete(datesets, shuffled_indices, axis=0)  # 删除补充点
    return datesets

def data_set_model(datesets):
    """
    模型数据集处理函数，过滤无效数据
    
    参数:
        datesets: 数据集
    
    返回:
        处理后的数据集
    """
    datesets = datesets[datesets[:,6]!=0]  # 保留第7列(索引6)不为0的数据
    return datesets
    
def data_set_models(datesets):
    """
    多模型数据集处理函数，对数据进行采样和过滤
    
    参数:
        datesets: 数据集
    
    返回:
        处理后的数据集
    """
    #idx = np.where(datesets[:, 3] < -0.75)[0]  # 查找第4列小于-0.75的索引（被注释掉）
    #datesets = np.delete(datesets, idx, axis= 0)  # 删除这些索引对应的数据（被注释掉）
    datesets = datesets[datesets[:,6]!=0]  # 保留第7列不为0的数据
    length_model = datesets.shape[0]  # 获取数据集大小
    shuffled_indices_model = np.random.choice(length_model, int(length_model * 0.45), replace=False)  # 随机选择45%的数据
    datesets = datesets[shuffled_indices_model, :]  # 采样数据
    point_w = 20000  # 设置点数阈值
    w_kk = np.abs(datesets[:, 8])  # 计算第9列的绝对值
    idx_w = np.where(w_kk < 1e-3)[0]  # 查找绝对值小于1e-3的索引
    datesets = data_cut_point(datesets, idx_w,point_w)  # 调用数据切割函数
    return datesets

def data_set_pv(datesets):
    """
    PV数据集处理函数，对数据进行采样
    
    参数:
        datesets: 数据集
    
    返回:
        处理后的数据集
    """
    #idx = np.where(datesets[:, 1] < 0)[0]  # 查找第2列小于0的索引（被注释掉）
    #datesets = np.delete(datesets, idx, axis=0)  # 删除这些索引对应的数据（被注释掉）
    length_model = datesets.shape[0]  # 获取数据集大小
    shuffled_indices_model = np.random.choice(length_model, int(datesets * 0.5), replace=False)  # 随机选择50%的数据（注意：此处代码有误，应为int(length_model * 0.5)）
    datesets = datesets[shuffled_indices_model, :]  # 采样数据
    return datesets

def norm_data(datapath):
    """
    数据归一化函数，计算数据集的最小值和最大值用于归一化
    
    参数:
        datapath: 数据文件夹路径
    
    返回:
        min_val: 最小值
        max_val: 最大值
        num_data: 数据数量
        num_points: 点数量列表
    """
    min_val = []  # 初始化最小值列表
    max_val = []  # 初始化最大值列表
    num_data = 0  # 初始化数据计数器
    num_points = []  # 初始化点数列表
    for f in os.listdir(datapath):  # 遍历数据目录中的所有文件
        if f[-3:] == 'txt':  # 只处理txt文件
            data = read_ti(os.path.join(datapath,f))  # 读取数据
            min_val.append(data.min(axis=0))  # 计算每列最小值并添加到列表
            max_val.append(data.max(axis=0))  # 计算每列最大值并添加到列表
            min_val = np.max(min_val,axis=0)  # 更新全局最小值
            max_val = np.min(max_val,axis=0)  # 更新全局最大值
            num_data += 1  # 增加数据计数
            num_points.append(data.shape[0])  # 添加数据点数
    return min_val,max_val, num_data,num_points

if __name__ == '__main__':
    """
    主函数，用于处理和保存数据集
    """
    np.random.seed(1234)  # 设置随机种子确保结果可重现
    pv_path = 'pv_data/'  # PV数据路径
    model_path = 'model_data/'  # 模型数据路径
    pv_model_path = 'pv_model/'  # PV模型数据路径
    period = 10  # 时间周期
    people = 50  # 人数
    time_list_info = list(range(168, 241, 8))  # 时间列表，从168到240，步长为8

    # 训练集数据进行收集
    if not os.path.exists("pv_model_50.pkl"):  # 如果训练数据pkl文件不存在
        wss_data, pressure_data ,u_data ,v_data, w_data = [], [], [], [], []  # 初始化各类数据列表
        xyz ,data= [] ,[]  # 初始化坐标和数据列表
        x = 0  # 初始化计数器
        for i in range(21,21+people):  # 遍历每个人的数据（从21到70）
            for j in time_list_info:  # 遍历每个时间点
                model_fn =str(i)+'-0'+str(j)+'.txt'  # 构造文件名
                if j == time_list_info[0]:  # 如果是第一个时间点
                    point_datas = read_ti(pv_model_path + model_fn)  # 读取数据
                    #point_datas = data_set_model(point_datas)  # 数据处理（被注释掉）
                    pressure_data, wss_data = point_datas[:, 4:5],point_datas[:, 9:10]  # 提取压力和WSS数据
                    u_data, v_data, w_data  = point_datas[:, 6:7], point_datas[:, 7:8], point_datas[:, 8:9]  # 提取速度分量数据
                    xyz = point_datas[:, 1:4]  # 提取坐标数据
                    x = x + 1  # 增加计数器
                    print('loading %s (%d/%d)' % (model_fn, x, people * period))  # 打印加载进度
                    continue  # 继续下一个时间点
                point_datas = read_ti(pv_model_path + model_fn)  # 读取数据
                #point_datas = data_set_model(point_datas)  # 数据处理（被注释掉）
                pressure, wss = point_datas[:, 4:5], point_datas[:, 9:10]  # 提取压力和WSS数据
                u, v, w = point_datas[:, 6:7], point_datas[:, 7:8], point_datas[:, 8:9]  # 提取速度分量数据
                wss_data = np.concatenate([wss_data, wss], axis=1)  # 拼接WSS数据
                pressure_data = np.concatenate([pressure_data, pressure], axis=1)  # 拼接压力数据
                u_data = np.concatenate([u_data, u], axis=1)  # 拼接u速度数据
                v_data = np.concatenate([v_data, v], axis=1)  # 拼接v速度数据
                w_data = np.concatenate([w_data, w], axis=1)  # 拼接w速度数据
                x = x + 1  # 增加计数器
                print('loading %s (%d/%d)'%(model_fn,x,people*period))  # 打印加载进度
            data.append([pressure_data, u_data, v_data, w_data, wss_data, xyz])  # 将处理后的数据添加到总数据列表
        pickle.dump(data, open('pv_model_50.pkl', 'wb'))  # 保存数据到pkl文件
    else:
        data = pickle.load(open('pv_model_50.pkl', 'rb'))  # 如果pkl文件存在，则直接加载

    # 测试集数据进行收集
    test_people = 1  # 测试集人数
    test_path = 'test_data/'  # 测试数据路径
    period = 10  # 时间周期
    if not os.path.exists("test_model_data.pkl"):  # 如果测试数据pkl文件不存在
       wss_data, pressure_data ,u_data ,v_data, w_data = [], [], [], [], []  # 初始化各类数据列表
       xyz ,data= [] ,[]  # 初始化坐标和数据列表
       x = 0  # 初始化计数器
       for i in range(102, 102 + test_people):  # 遍历测试人员数据（从102开始）
           for j in time_list_info:  # 遍历每个时间点
               model_fn =str(i)+'-0'+str(j)+'.txt'  # 构造文件名
               if j == time_list_info[0]:  # 如果是第一个时间点
                   point_datas = read_ti(test_path + model_fn)  # 读取数据
                   #point_datas = data_set_model(point_datas)  # 数据处理（被注释掉）
                   pressure_data, wss_data = point_datas[:, 4:5], point_datas[:, 9:10]  # 提取压力和WSS数据
                   u_data, v_data, w_data  = point_datas[:, 6:7], point_datas[:, 7:8], point_datas[:, 8:9]  # 提取速度分量数据
                   xyz = point_datas[:, 1:4]  # 提取坐标数据
                   x = x + 1  # 增加计数器
                   print('loading %s (%d/%d)' % (model_fn, x, test_people * period))  # 打印加载进度
                   continue  # 继续下一个时间点
               point_datas = read_ti(test_path + model_fn)  # 读取数据
               #point_datas = data_set_model(point_datas)  # 数据处理（被注释掉）
               pressure, wss = point_datas[:, 4:5], point_datas[:, 9:10]  # 提取压力和WSS数据
               u, v, w = point_datas[:, 6:7], point_datas[:, 7:8], point_datas[:, 8:9]  # 提取速度分量数据
               pressure_data = np.concatenate([pressure_data, pressure], axis=1)  # 拼接压力数据
               wss_data = np.concatenate([wss_data, wss], axis=1)  # 拼接WSS数据
               u_data = np.concatenate([u_data, u], axis=1)  # 拼接u速度数据
               v_data = np.concatenate([v_data, v], axis=1)  # 拼接v速度数据
               w_data = np.concatenate([w_data, w], axis=1)  # 拼接w速度数据
               x = x + 1  # 增加计数器
               print('loading %s (%d/%d)'%(model_fn,x,test_people*period))  # 打印加载进度
           data.append([pressure_data, u_data, v_data, w_data, wss_data, xyz])  # 将处理后的数据添加到总数据列表
       pickle.dump(data, open('test_model_data.pkl', 'wb'))  # 保存数据到pkl文件
    else:
       data = pickle.load(open('test_model_data.pkl', 'rb'))  # 如果pkl文件存在，则直接加载

    # 注释掉的PV数据处理部分
    '''
    if not os.path.exists("pv_data_50.pkl"):  # 如果pv数据pkl文件不存在
        xyz ,data= [] ,[]  # 初始化坐标和数据列表
        x = 0  # 初始化计数器
        for i in range(21,21+people):  # 遍历每个人的数据
            pv_fn = str(i)+'-0'+'168'+'.txt'  # 构造文件名
            point_datas = read_ti(pv_path + pv_fn)  # 读取数据
            #point_datas = data_set_pv(point_datas)  # 数据处理（被注释掉）
            xyz = point_datas[:, 1:4]  # 提取坐标数据
            x = x+1  # 增加计数器
            print('loading %s (%d/%d)' % (pv_fn, x, people))  # 打印加载进度
            data.append([xyz])  # 将坐标数据添加到总数据列表
        pickle.dump(data, open('pv_data_50.pkl', 'wb'))  # 保存数据到pkl文件
    else:
        data = pickle.load(open('pv_data_50.pkl', 'rb'))  # 如果pkl文件存在，则直接加载
    '''