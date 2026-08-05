import torch  # 导入PyTorch深度学习框架，用于构建和训练神经网络
from torch import nn  # 从torch中导入神经网络模块，包含各种神经网络层和函数
import pandas as pd  # 导入Pandas数据分析库，用于数据处理和分析
import numpy as np  # 导入NumPy科学计算库，用于处理多维数组和矩阵运算
# 导入自定义的PointNet库，这是一个专门处理点云数据的神经网络模型
import pointnet_wss as pointnet_coef
import pickle  # 导入pickle模块，用于序列化和反序列化Python对象
import datetime  # 导入datetime模块，用于处理日期和时间
import matplotlib.pyplot as plt  # 导入matplotlib绘图库，用于数据可视化
import math  # 导入math数学库，提供各种数学函数

# 定义绘制点云数据的函数，用于三维可视化点云数据
def draw_pc(data,color,title):
    """
    绘制点云数据的三维散点图
    
    参数:
        data: 点云数据，形状为(N, 3)，N为点的数量
        color: 颜色值，可以是单一颜色或每个点对应的颜色数组
        title: 图形标题字符串
    """
    fig=plt.figure(dpi=120)  # 创建一个分辨率为120dpi的图形窗口
    ax=fig.add_subplot(111,projection='3d')  # 添加一个3D子图
    plt.title(title)  # 设置图形标题
    # 绘制三维散点图，x、y、z坐标分别来自data的第0、1、2列
    ax.scatter(data[:,0],data[:,1],data[:,2],c=color,marker='.',s=2,linewidth=0,alpha=1,cmap='hot')

# 定义预测单个时间步的函数，用于对CFD数据的一个时间片进行预测
def predict_one(xyzt, batch_size, N, u_cfd, v_cfd, w_cfd, p_cfd, wss_cfd):
    """
    预测单个时间步的数据，将整个数据集分批处理以节省内存
    
    参数:
        xyzt: 坐标和时间数据，形状为(N, 4)
        batch_size: 批次大小，控制每次处理的数据量
        N: 数据总数量
        u_cfd, v_cfd, w_cfd: 分别表示x、y、z方向的速度CFD数据
        p_cfd: 压力CFD数据
        wss_cfd: 壁面剪切应力CFD数据
    
    返回:
        各物理量的预测结果和误差指标(MSE和MAE)
    """
    batch = math.ceil(N / batch_size)  # 计算需要分成多少个批次
    step = 0  # 初始化已处理的批次计数器
    # 数据归一化参数，包括均值、最大值、最小值和标准差
    data_mean = [-36.0, 0.0024, 0.0010, -0.042, 0.7]  # [f, u, v, w, wss]的均值
    data_max = [300, 0.8, 0.8, 0.3, 30]  # [f, u, v, w, wss]的最大值
    data_min = [-1200.0, -0.7, -0.7, -1.0, 0.002]  # [f, u, v, w, wss]的最小值
    data_std = [420.0, 0.06, 0.06, 0.09, 1.4]  # [f, u, v, w, wss]的标准差
    f_mean, u_mean, v_mean, w_mean, wss_mean = data_mean  # 解包均值参数
    f_max, u_max, v_max, w_max, wss_max = data_max  # 解包最大值参数
    f_min, u_min, v_min, w_min, wss_min = data_min  # 解包最小值参数
    f_std, u_std, v_std, w_std, wss_std = data_std  # 解包标准差参数
    data_former = [1, 1, 1, 1]  # 数据格式参数(未使用)
    u_former, v_former, w_former, p_former = data_former  # 解包数据格式参数
    # 初始化各物理量的均方误差和平均绝对误差
    u_mse, v_mse, w_mse, p_mse, wss_mse =0.0,0.0,0.0,0.0,0.0
    u_mae, v_mae, w_mae, p_mae, wss_mae = 0.0, 0.0, 0.0, 0.0, 0.0
    # 初始化预测数据存储变量
    u_one, v_one, w_one, p_one, wss_one = None, None, None, None, None
    # 按批次进行预测处理
    for b in range(batch):
        # 计算当前批次的索引范围
        N_number = np.arange(b*batch_size, (b+1)*batch_size)
        N_number_info = int(N- b* batch_size)  # 当前批次剩余数据数量
        torch.cuda.empty_cache()  # 清空CUDA缓存，释放GPU内存
        # 处理最后一个不完整的批次
        if N_number_info < batch_size:
            xyz = xyzt[-N_number_info:, :]  # 获取最后一批数据
            xyz = _tensor(xyz)  # 将数据转换为GPU张量
            # 获取当前批次的CFD数据
            u_cfd_batch, v_cfd_batch, w_cfd_batch, p_cfd_batch = u_cfd[-N_number_info:], v_cfd[-N_number_info:], w_cfd[-N_number_info:], p_cfd[-N_number_info:]
            wss_cfd_batch = wss_cfd[-N_number_info:]  # 获取壁面剪切应力数据
            uvwp,_ = model(xyz)  # 使用模型进行预测
            # 计算损失
            xyz = xyz.detach().cpu()  # 将数据从GPU转移到CPU
            # 注释掉的代码是另一种数据反归一化方法
            #u_batch, v_batch, w_batch, p_batch=data_outformer(uvwp[0]/10, u_mean, u_max, u_min),data_outformer(uvwp[1]/10, v_mean, v_max, v_min),data_outformer(uvwp[2]/5, w_mean, w_max, w_min),data_outformer(uvwp[3], f_mean, f_max, f_min)
            #uvw, p_batch, wss_batch = uvwp[0],uvwp[1],uvwp[2]
            # 获取模型预测的各物理量
            u_batch, v_batch, w_batch, p_batch, wss_batch = uvwp[0], uvwp[1], uvwp[2], uvwp[3], uvwp[4]
            # 计算速度u的均方误差和平均绝对误差
            u_batch_mse,u_batch_mae = compute_data_loss_mse(u_cfd_batch, u_batch.squeeze(1))
            u_batch = u_batch.detach().cpu()  # 将结果转移到CPU
            # 计算速度v的均方误差和平均绝对误差
            v_batch_mse,v_batch_mae  = compute_data_loss_mse(v_cfd_batch, v_batch.squeeze(1))
            v_batch = v_batch.detach().cpu()  # 将结果转移到CPU
            # 计算速度w的均方误差和平均绝对误差
            w_batch_mse,w_batch_mae  = compute_data_loss_mse(w_cfd_batch, w_batch.squeeze(1))
            w_batch = w_batch.detach().cpu()  # 将结果转移到CPU
            # 对压力数据进行归一化处理
            p_cfd_batch = data_informer(p_cfd_batch, f_max, f_min)
            # 计算压力的均方误差和平均绝对误差
            p_batch_mse,p_batch_mae  = compute_data_loss_mse(p_cfd_batch, p_batch.squeeze(1))
            # 对压力预测结果进行反归一化
            p_batch = data_outformer(p_batch, f_mean, f_max, f_min)
            p_batch = p_batch.detach().cpu()  # 将结果转移到CPU
            # 对壁面剪切应力数据进行归一化处理
            wss_cfd_batch = data_informer(wss_cfd_batch, wss_max, wss_min)
            # 计算壁面剪切应力的均方误差和平均绝对误差
            wss_batch_mse,wss_batch_mae  = compute_data_loss_mse(wss_cfd_batch, wss_batch.squeeze(1))
            # 对壁面剪切应力预测结果进行反归一化
            wss_batch = data_outformer(wss_batch, wss_mean, wss_max, wss_min)
            wss_batch = wss_batch.detach().cpu()  # 将结果转移到CPU
            # 累计各物理量的误差
            u_mse += u_batch_mse
            v_mse += v_batch_mse
            w_mse += w_batch_mse
            p_mse += p_batch_mse
            wss_mse += wss_batch_mse
            u_mae += u_batch_mae
            v_mae += v_batch_mae
            w_mae += w_batch_mae
            p_mae += p_batch_mae
            wss_mae += wss_batch_mae
            # 累计预测数据
            if u_one is None:
                u_one, v_one, w_one, p_one, wss_one = u_batch, v_batch, w_batch, p_batch, wss_batch
            else:
                u_one = np.concatenate((u_one, u_batch), axis=0)
                v_one = np.concatenate((v_one, v_batch), axis=0)
                w_one = np.concatenate((w_one, w_batch), axis=0)
                p_one = np.concatenate((p_one, p_batch), axis=0)
                wss_one = np.concatenate((wss_one, wss_batch), axis=0)
            step +=1
        else:
            # 处理完整批次的数据
            xyz = xyzt[N_number, :]  # 获取当前批次数据
            xyz = _tensor(xyz)  # 将数据转换为GPU张量
            # 获取当前批次的CFD数据
            u_cfd_batch, v_cfd_batch, w_cfd_batch, p_cfd_batch = u_cfd[N_number], v_cfd[N_number],w_cfd[N_number], p_cfd[N_number]
            wss_cfd_batch = wss_cfd[N_number]  # 获取壁面剪切应力数据
            uvwp,_ = model(xyz)  # 使用模型进行预测
            xyz = xyz.detach().cpu()  # 将数据从GPU转移到CPU
            # 注释掉的代码是另一种数据处理方式
            #u_batch,v_batch,w_batch,p_batch=data_outformer(uvwp[0]/10, u_mean, u_max, u_min),data_outformer(uvwp[1]/10, v_mean, v_max, v_min),data_outformer(uvwp[2]/5, w_mean, w_max, w_min),data_outformer(uvwp[3], f_mean, f_max, f_min)
            #uvw, p_batch, wss_batch = uvwp[0], uvwp[3], uvwp[4]
            # 获取模型预测的各物理量
            u_batch, v_batch, w_batch, p_batch, wss_batch = uvwp[0], uvwp[1], uvwp[2], uvwp[3], uvwp[4]
            # 计算各物理量的误差
            u_batch_mse,u_batch_mae = compute_data_loss_mse(u_cfd_batch, u_batch.squeeze(1))
            u_batch = u_batch.detach().cpu()  # 将结果转移到CPU
            v_batch_mse,v_batch_mae = compute_data_loss_mse(v_cfd_batch, v_batch.squeeze(1))
            v_batch = v_batch.detach().cpu()  # 将结果转移到CPU
            w_batch_mse,w_batch_mae = compute_data_loss_mse(w_cfd_batch, w_batch.squeeze(1))
            w_batch = w_batch.detach().cpu()  # 将结果转移到CPU
            p_cfd_batch = data_informer(p_cfd_batch, f_max, f_min)  # 对压力数据进行归一化
            p_batch_mse,p_batch_mae = compute_data_loss_mse(p_cfd_batch, p_batch.squeeze(1))
            p_batch = data_outformer(p_batch, f_mean, f_max, f_min)  # 对压力预测结果进行反归一化
            p_batch = p_batch.detach().cpu()  # 将结果转移到CPU
            wss_cfd_batch = data_informer(wss_cfd_batch, wss_max, wss_min)  # 对壁面剪切应力数据进行归一化
            wss_batch_mse,wss_batch_mae = compute_data_loss_mse(wss_cfd_batch, wss_batch.squeeze(1))
            wss_batch = data_outformer(wss_batch, wss_mean, wss_max, wss_min)  # 对壁面剪切应力预测结果进行反归一化
            wss_batch = wss_batch.detach().cpu()  # 将结果转移到CPU
            # 累计各物理量的误差
            u_mse += u_batch_mse
            v_mse += v_batch_mse
            w_mse += w_batch_mse
            p_mse += p_batch_mse
            wss_mse += wss_batch_mse
            u_mae += u_batch_mae
            v_mae += v_batch_mae
            w_mae += w_batch_mae
            p_mae += p_batch_mae
            wss_mae += wss_batch_mae
            torch.cuda.empty_cache()  # 清空CUDA缓存，释放GPU内存
            # 累计预测数据
            if u_one is None :
                u_one, v_one, w_one, p_one, wss_one = u_batch, v_batch, w_batch, p_batch, wss_batch
            else:
                u_one = np.concatenate((u_one, u_batch), axis=0)
                v_one = np.concatenate((v_one, v_batch), axis=0)
                w_one = np.concatenate((w_one, w_batch), axis=0)
                p_one = np.concatenate((p_one, p_batch), axis=0)
                wss_one = np.concatenate((wss_one, wss_batch), axis=0)
            step += 1
    # 计算平均误差
    u_mse = u_mse / step
    v_mse = v_mse / step
    w_mse = w_mse / step
    p_mse = p_mse / step
    wss_mse = wss_mse / step
    u_mae = u_mae / step
    v_mae = v_mae / step
    w_mae = w_mae / step
    p_mae = p_mae / step
    wss_mae = wss_mae / step
    print(u_mse, v_mse, w_mse, p_mse, wss_mse)  # 打印各物理量的均方误差
    assert u_one.shape[0] == N,'predict is not completed'  # 断言预测结果数量与输入数据数量一致
    # 返回预测结果和误差指标
    return u_one, v_one, w_one, p_one, wss_one, u_mse, v_mse, w_mse, p_mse, wss_mse,u_mae, v_mae, w_mae, p_mae, wss_mae

# 定义计算数据损失（MAPE）的函数
def compute_data_loss_mape(u_cfd, u):
    """
    计算平均绝对百分比误差(MAPE)
    
    参数:
        u_cfd: CFD真实数据
        u: 模型预测数据
    
    返回:
        MAPE值
    """
    u_cfd = _tensor(u_cfd)  # 将数据转换为GPU张量
    mape = torch.mean(torch.abs((u - u_cfd) / u_cfd))  # 计算MAPE
    u_cfd = u_cfd.detach().cpu()  # 将数据从GPU转移到CPU
    return  mape

# 定义计算数据损失（MSE和MAE）的函数
def compute_data_loss_mse(u_cfd, u):
    """
    计算均方误差(MSE)和平均绝对误差(MAE)
    
    参数:
        u_cfd: CFD真实数据
        u: 模型预测数据
    
    返回:
        MSE和MAE值
    """
    u_cfd = _tensor(u_cfd)  # 将数据转换为GPU张量
    mse = MSE(u_cfd, u)  # 使用MSE损失函数计算均方误差
    mae = MAE(u_cfd, u)  # 使用MAE损失函数计算平均绝对误差
    #mse = torch.mean((u - u_cfd)**2)  # 另一种计算MSE的方法
    u_cfd = u_cfd.detach().cpu()  # 将数据从GPU转移到CPU
    return  mse.item(),mae.item()  # 返回MSE和MAE的数值

def data_outformer(u, u_mean, u_max, u_min):
    """
    数据反归一化函数，将归一化后的数据还原为原始数据范围
    
    参数:
        u: 归一化后的数据
        u_mean: 均值
        u_max: 最大值
        u_min: 最小值
    
    返回:
        反归一化后的数据
    """
    return (u+1)/2*(u_max - u_min)+u_min

def data_informer(u, u_max, u_min):
    """
    数据归一化函数，将原始数据归一化到[-1, 1]范围
    
    参数:
        u: 原始数据
        u_max: 最大值
        u_min: 最小值
    
    返回:
        归一化后的数据
    """
    return (u - u_min) / (u_max - u_min) * 2 - 1

# 定义将数据转换为张量的函数
def _tensor(cfd):
    """
    将NumPy数组转换为GPU张量
    
    参数:
        cfd: NumPy数组数据
    
    返回:
        GPU上的浮点型张量
    """
    cfd = torch.from_numpy(cfd).float().cuda(device)  # 转换为GPU张量
    return cfd

# 定义预测步骤函数
def predict_step(model, xyzt, batch_size, u_cfd, v_cfd, w_cfd,  p_cfd, wss_cfd):
    """
    预测步骤函数，设置模型为评估模式并调用预测函数
    
    参数:
        model: 训练好的模型
        xyzt: 坐标和时间数据
        batch_size: 批次大小
        u_cfd, v_cfd, w_cfd: 速度CFD数据
        p_cfd: 压力CFD数据
        wss_cfd: 壁面剪切应力CFD数据
    
    返回:
        各物理量的预测结果和误差指标
    """
    model.eval()  # 设置模型为评估模式
    ## 数据转化格式
    N = xyzt.shape[0]  # 获取数据数量
    # 调用predict_one函数进行预测
    u, v, w, p, wss , u_mse, v_mse, w_mse, p_mse, wss_mse,u_mae, v_mae, w_mae, p_mae, wss_mae= predict_one(xyzt, batch_size, N, u_cfd, v_cfd, w_cfd, p_cfd, wss_cfd)
    ## 进行预测
    return u, v, w, p, wss , u_mse, v_mse, w_mse, p_mse, wss_mse,u_mae, v_mae, w_mae, p_mae, wss_mae

# 程序主入口
if __name__ == "__main__":
    # 模型配置参数
    ues_coef = True  # 是否使用系数
    use_res  = False  # 是否使用残差连接
    # 模型权重文件路径
    res_s = 'weight/PINN_CFD_BLOOD_weight_1500.pth'
    res_T = 'weight/PINN_CFD_BLOOD_weight_1500_res.pth'
    res_F = 'weight/PINN_CFD_BLOOD_weight_1500.pth'
    res_a = 'weight/PINN_CFD_BLOOD_weight_1500_a.pth'
    res_r = 'weight/PINN_CFD_BLOOD_weight_1500_r.pth'
    res = [0.1, 0.1]  # 残差参数

    # 创建模型实例
    model = pointnet_coef.get_model(ues_coef, use_res, res[0], res[1])
    device = torch.device('cuda:0')  # 设置使用第一个GPU
    #model.load_state_dict(torch.load('weight/PINN_CFD_BLOOD_weight.pth'))  # 注释掉的模型加载方式
    model.load_state_dict(torch.load( res_s ))  # 加载训练好的模型权重
    model.to(device)  # 将模型移动到GPU

    # 定义损失函数
    MSE = nn.MSELoss()  # 均方误差损失函数
    MAE = nn.L1Loss()   # 平均绝对误差损失函数
    # 加载CFD数据
    model_data = pickle.load(open('pv_model_50.pkl', 'rb'))[0]
    p_cfd, u_cfd, v_cfd, w_cfd, wss_cfd ,xyz = model_data  # 解包CFD数据
    start = 0     # 起始时间步
    period = 10   # 预测周期
    batch_size = 50000  # 批次大小
    # 初始化预测结果和误差存储变量
    u_pred, u_mse_total, u_mae_total = None, [], []
    v_pred, v_mse_total, v_mae_total = None, [], []
    w_pred, w_mse_total, w_mae_total = None, [], []
    p_pred, p_mse_total, p_mae_total = None, [], []
    wss_pred, wss_mse_total, wss_mae_total = None, [], []
    ## continue to predict
    # 遍历每个时间步进行预测
    for idx in range(start,period+start):
        N, B = p_cfd.shape  # 获取数据维度
        # 构造时间维度数据
        time = (idx+1) * 0.08 * (np.ones_like(p_cfd[:,0:1]) + np.finfo(np.float32).eps)
        # 拼接坐标和时间数据
        xyzt = np.concatenate((xyz,time),axis = -1)
        xyzt = xyzt * 5  # 对数据进行缩放
        # 调用预测函数
        u,v,w,p,wss, u_mse, v_mse, w_mse, p_mse, wss_mse,u_mae, v_mae, w_mae, p_mae, wss_mae = predict_step(model, xyzt, batch_size, u_cfd[:, idx], v_cfd[:, idx], w_cfd[:, idx],  p_cfd[:, idx], wss_cfd[:,idx])

        ## save predict and cfd data
        # 保存预测结果
        if u_pred is None:
            u_pred = u
            v_pred = v
            w_pred = w
            p_pred = p
            wss_pred = wss
        else:
            u_pred = np.concatenate((u_pred, u), axis=1)
            v_pred = np.concatenate((v_pred, v), axis=1)
            w_pred = np.concatenate((w_pred, w), axis=1)
            p_pred = np.concatenate((p_pred, p), axis=1)
            wss_pred = np.concatenate((wss_pred, wss), axis=1)
        ## save mse value
        # 保存误差值
        u_mse_total.append(u_mse)
        v_mse_total.append(v_mse)
        w_mse_total.append(w_mse)
        p_mse_total.append(p_mse)
        wss_mse_total.append(wss_mse)
        u_mae_total.append(u_mae)
        v_mae_total.append(v_mae)
        w_mae_total.append(w_mae)
        p_mae_total.append(p_mae)
        wss_mae_total.append(wss_mae)

    # 计算预测结果和CFD数据的平均值
    u_mean_pred, u_mean_cfd = np.mean(u_pred,axis=0), np.mean(u_cfd,axis=0)
    v_mean_pred, v_mean_cfd = np.mean(v_pred,axis=0), np.mean(v_cfd,axis=0)
    w_mean_pred, w_mean_cfd = np.mean(w_pred,axis=0), np.mean(w_cfd,axis=0)
    p_mean_pred, p_mean_cfd = np.mean(p_pred,axis=0), np.mean(p_cfd,axis=0)
    wss_mean_pred, wss_mean_cfd = np.mean(wss_pred, axis=0), np.mean(wss_cfd, axis=0)
    # 计算预测误差
    p_error = p_pred - p_cfd
    wss_error = wss_pred - wss_cfd

    # 打印结果
    print('p:',p_mean_pred)
    print('p_cfd:',p_mean_cfd[start:period+start])
    print('wss:',wss_mean_pred)
    print('wss_cfd:',wss_mean_cfd[start:period + start])

    print('p_mse_total:', p_mse_total)
    print('p_mae_total:', p_mae_total)
    print('wss_mse_total:', wss_mse_total)
    print('wss_mae_total:', wss_mae_total)
    print('MSE')
    print(np.mean(p_mse_total))
    print(np.mean(wss_mse_total))
    print('MAE')
    print(np.mean(p_mae_total))
    print(np.mean(wss_mae_total))

    # 绘制压力和壁面剪切应力的平均值对比图
    kk = list(range(len(p_mse_total)))
    figk = plt.figure(dpi=150)
    ax = figk.add_subplot(1, 2, 1)
    ax.plot(kk, p_mean_pred, c='y', label='p_pred', ls='--')
    ax.plot(kk, p_mean_cfd[start:period+start], c='r', label='p_cfd')
    ax.set_title('p')
    ax.legend(frameon=False)
    ax = figk.add_subplot(1, 2, 2)
    ax.plot(kk, wss_mean_pred, c='y', label='wss_pred', ls='--')
    ax.plot(kk, wss_mean_cfd[start:period + start], c='r', label='wss_cfd')
    ax.set_title('wss')
    ax.legend(frameon=False)

    # 绘制压力和壁面剪切应力的误差对比图
    figw = plt.figure(dpi=150)
    ax = figw.add_subplot(1, 2, 1)
    ax.plot(kk, p_mse_total, c='r', label='L2', ls='--')
    ax.plot(kk, p_mae_total, c='y', label='L1', ls='--')
    ax.set_title('p')
    ax.legend(frameon=False)
    ax = figw.add_subplot(1, 2, 2)
    ax.plot(kk, wss_mse_total, c='r', label='L2', ls='--')
    ax.plot(kk, wss_mae_total, c='y', label='L1', ls='--')
    ax.set_title('wss')
    ax.legend(frameon=False)

    # 绘制壁面剪切应力的3D可视化图
    fig = plt.figure(dpi=150)
    fig1 = plt.figure(dpi=150)
    fige = plt.figure(dpi=150)
    len_seq =  period
    for location in range(1,len_seq+1):
        ax = fig.add_subplot(2, 5, location, projection='3d')
        ax.axis('off')
        ax1 = fig1.add_subplot(2, 5, location, projection='3d')
        ax1.axis('off')
        ax2 = fige.add_subplot(2, 5, location, projection='3d')
        ax2.axis('off')
        title1 = "wss Time:{}s".format(location * 0.08)
        title = "wss Time:{}s {:.4f}".format(location * 0.08, wss_mse_total[location - 1])
        real_data = wss_cfd[:, location - 1]
        pred_data = wss_pred[:, location - 1]
        error_data = wss_error[:, location - 1]
        ax.set_title(title1)
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=real_data, marker='.', s=2, linewidth=0, alpha=1, cmap='hot')
        ax1.set_title(title)
        ax1.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=pred_data, marker='.', s=2, linewidth=0, alpha=1, cmap='hot')
        ax2.set_title(title1)
        ax2.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=error_data, marker='.', s=2, linewidth=0, alpha=1, cmap='hot')

    # 绘制压力的3D可视化图
    fig2 = plt.figure(dpi=150)
    fig3 = plt.figure(dpi=150)
    fig4 = plt.figure(dpi=150)
    len_seq = period
    for location in range(1, len_seq + 1):
        ax = fig2.add_subplot(2, 5, location, projection='3d')
        ax.axis('off')
        ax1 = fig3.add_subplot(2, 5, location, projection='3d')
        ax1.axis('off')
        ax2 = fig4.add_subplot(2, 5, location, projection='3d')
        ax2.axis('off')
        title = "p Time:{}s {:.4f}".format(location * 0.08,p_mse_total[location-1])
        title1 = "p Time:{}s".format(location * 0.08)
        real_data = p_cfd[:, location - 1]
        pred_data = p_pred[:, location - 1]
        error_data = p_error[:, location - 1]
        ax.set_title(title1)
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=real_data, marker='.', s=2, linewidth=0, alpha=1, cmap='hot')
        ax1.set_title(title,fontsize =10)
        ax1.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=pred_data, marker='.', s=2, linewidth=0, alpha=1, cmap='hot')
        ax2.set_title(title1)
        ax2.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=error_data, marker='.', s=2, linewidth=0, alpha=1, cmap='hot')

    # 注释掉的数据保存代码
    #xyz = pd.DataFrame(xyz).to_csv('N/xyz.csv', header=False, index=False)
    #p_pred = pd.DataFrame(p_pred).to_csv('N/p_pred_1000.csv', header=False, index=False)
    #p_cfd = pd.DataFrame(p_cfd).to_csv('N/p_cfd.csv', header=False, index=False)
    #wss_pred = pd.DataFrame(wss_pred).to_csv('N/wss_pred_1000.csv', header=False, index=False)
    #wss_cfd = pd.DataFrame(wss_cfd).to_csv('N/wss_cfd.csv', header=False, index=False)
    #plt.show()