import os
# 设置TensorFlow日志级别，0表示显示所有日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
# 设置CUDA设备顺序
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# 设置使用的GPU ID
GPU_ID = "0"
# 设置可见的CUDA设备
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
import numpy as np  # 导入NumPy库，用于数值计算
import pandas as pd  # 导入Pandas库，用于数据处理
import time  # 导入时间模块
import math  # 导入数学模块
import glob  # 导入文件路径匹配模块
from datetime import datetime  # 导入日期时间模块
import torch  # 导入PyTorch库，用于深度学习
import torch.nn as nn  # 导入神经网络模块
import csv  # 导入CSV模块
import pickle  # 导入Pickle模块，用于序列化
# 导入自定义的PointNet库
import pointnet_wss as pointnet_coef
from sklearn.metrics import mean_absolute_error  # 导入平均绝对误差计算函数
from sklearn.metrics import mean_squared_error  # 导入均方误差计算函数
from math import sqrt  # 导入平方根函数

# 设置随机种子，确保结果可重现
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)

class Pinn_Cfd_Blood:
    def __init__(self, loss_weights_Data, loss_weights_Equ, Re, point_number, data_former,
                 epochs, batch_sizes, learning_rates, net_params, opt_params, start, period, use_coef, use_res, res):
        """
        初始化Pinn_Cfd_Blood类
        
        参数:
        loss_weights_Data: 数据损失权重
        loss_weights_Equ: 方程损失权重
        Re: 雷诺数
        point_number: 点数量
        data_former: 数据格式化参数
        epochs: 训练轮数列表
        batch_sizes: 批大小列表
        learning_rates: 学习率列表
        net_params: 网络参数
        opt_params: 优化器参数
        start: 开始位置
        period: 周期
        use_coef: 是否使用系数
        use_res: 是否使用残差连接
        res: 残差参数
        """
        self.epoch_loss_checkpoints = 1e10

        # 各方程数据相关参数
        self.learning_rates = learning_rates  # 学习率列表
        self.epochs = epochs  # 训练轮数列表
        self.batch_sizes = batch_sizes  # 批大小列表
        self.loss_weights_Data = loss_weights_Data  # 数据损失权重
        self.loss_weights_Equ = loss_weights_Equ  # 方程损失权重
        self.data_former = data_former  # 数据格式化参数
        
        # 损失曲线相关参数
        self.loss_list = ["total", "f_loss", "wss_loss","loss_equ", "loss_continuous", "loss_bc"]  # 损失类型列表
        self.epoch_loss = dict.fromkeys(self.loss_list, 0)  # 初始化每个损失类型为0
        self.loss_history = dict((loss, []) for loss in self.loss_list)  # 初始化损失历史记录
        
        # 系数相关参数
        self.coef_list = ["p_coef",'wss_coef']  # 系数类型列表
        self.coef_history = dict((coef, []) for coef in self.coef_list)  # 初始化系数历史记录

        # 定义网络相关参数
        self.device = torch.device('cuda:0')  # 设置使用的设备为GPU
        self.use_coef = use_coef  # 是否使用系数
        if use_coef:
            # 获取PointNet模型
            self.net = pointnet_coef.get_model(use_coef, use_res, res[0], res[1])
        else:
            # 获取PointNet模型
            self.net = pointnet_coef.get_model(use_coef, use_res, res[0], res[1])
        self.loss = pointnet_coef.get_loss()  # 获取损失函数
        self.net_params = net_params  # 网络参数
        self.opt_params = opt_params  # 优化器参数
        self.net.to(self.device)  # 将网络移动到指定设备

        # 定义其他参数
        self.loss_epochs = 1e5  # 设置初始的损失轮数为一个较大的值
        self.Re = Re  # 雷诺数
        self.point_number = point_number  # 点数量
        self.pv = 0.2  # 设置pv值
        self.xyzt = 5  # 设置xyzt值

        self.strat = start  # 开始位置
        self.period = period  # 周期
        self.mse = nn.MSELoss()  # 初始化均方误差损失函数
        
    # 连续性条件
    def fun_u_0(self, u_x, v_y, w_z):
        """
        连续性条件函数，计算速度场的散度
        
        参数:
        u_x: u方向速度对x的偏导数
        v_y: v方向速度对y的偏导数
        w_z: w方向速度对z的偏导数
        
        返回:
        速度场的散度
        """
        return u_x + v_y + w_z

    # Define boundary condition   边界条件
    def fun_u_b(self, u, v, w):
        """
        边界条件函数，计算速度场在边界的模长平方和
        
        参数:
        u: u方向速度
        v: v方向速度
        w: w方向速度
        
        返回:
        速度场在边界的模长平方和
        """
        return torch.mean(u**2) + torch.mean(v**2) + torch.mean(w**2)

    # Define residual of the PDE 残差
    def fun_z(self, u, v, w, u_t, u_x, u_y, u_z, f_x, u_xx, u_yy, u_zz):
        """
        PDE残差函数，计算z方向动量方程的残差
        
        参数:
        u, v, w: 速度分量
        u_t: u对时间的偏导数
        u_x, u_y, u_z: u对空间坐标的偏导数
        f_x: 压力对x的偏导数
        u_xx, u_yy, u_zz: u的二阶偏导数
        
        返回:
        z方向动量方程的残差
        """
        #u_former, v_former, w_former, f_former = self.data_former
        return u_t + u * u_x + v * u_y + w * u_z + f_x  - (u_xx + u_yy + u_zz) / self.Re + 9.81/5
        
    def fun_xy(self, u, v, w, u_t, u_x, u_y, u_z, f_x, u_xx, u_yy, u_zz):
        """
        PDE残差函数，计算xy平面动量方程的残差
        
        参数:
        u, v, w: 速度分量
        u_t: u对时间的偏导数
        u_x, u_y, u_z: u对空间坐标的偏导数
        f_x: 压力对x的偏导数
        u_xx, u_yy, u_zz: u的二阶偏导数
        
        返回:
        xy平面动量方程的残差
        """
        #u_former, v_former, w_former, f_former = self.data_former
        return u_t + u * u_x + v * u_y + w * u_z + f_x  - (u_xx + u_yy + u_zz) / self.Re

    ## gradient 梯度
    def fun_gradient(self,u, v, w, u_tx, u_x, u_y, u_z, u_x2, v_x, w_x, u_xx, u_yx, u_zx, f_xx, u_xxx, u_yyx, u_zzx):
        """
        梯度计算函数，计算高阶梯度相关项
        
        参数:
        u, v, w: 速度分量
        u_tx: u对时间和x的混合偏导数
        u_x, u_y, u_z: u对空间坐标的偏导数
        u_x2: u_x的平方
        v_x, w_x: v和w对x的偏导数
        u_xx, u_yx, u_zx: u的二阶偏导数
        f_xx: 压力对x的二阶偏导数
        u_xxx, u_yyx, u_zzx: u的三阶偏导数
        
        返回:
        梯度相关项
        """
        return u_tx+u_x2*u_x+u*u_xx+v_x*u_y+v*u_yx+w_x*u_z+w*u_zx+f_xx-(u_xxx+u_yyx+u_zzx)/ self.Re

    ## Data of loss 损失数据
    def compute_data_loss_mape(self, u, u_cfd):
        """
        计算平均绝对百分比误差
        
        参数:
        u: 预测值
        u_cfd: CFD真实值
        
        返回:
        平均绝对百分比误差
        """
        return torch.mean(torch.abs((u-u_cfd)/u_cfd)) * 100
        
    def compute_data_loss_mse(self, u, u_cfd):
        """
        计算均方误差
        
        参数:
        u: 预测值
        u_cfd: CFD真实值
        
        返回:
        均方误差
        """
        #mse = torch.mean((u - u_cfd) ** 2)
        mse = self.mse(u, u_cfd)
        return mse
        
    def compute_data_loss_rmse(self, u, u_cfd):
        """
        计算均方根误差
        
        参数:
        u: 预测值
        u_cfd: CFD真实值
        
        返回:
        均方根误差
        """
        rmse = torch.sqrt(self.mse(u, u_cfd))
        return rmse
        
    def compute_data_loss_mae(self, u, u_cfd):
        """
        计算平均绝对误差
        
        参数:
        u: 预测值
        u_cfd: CFD真实值
        
        返回:
        平均绝对误差
        """
        mae = mean_absolute_error(u_cfd,u)
        return mae
        
    # data 数据处理
    def data_informer(self,u, u_max, u_min):
        """
        数据归一化函数，将数据归一化到[-1, 1]区间
        
        参数:
        u: 原始数据
        u_max: 数据最大值
        u_min: 数据最小值
        
        返回:
        归一化后的数据
        """
        return (u - u_min) / (u_max - u_min) * 2 - 1
        
    def data_outformer(self,u, u_max, u_min):
        """
        数据反归一化函数，将归一化的数据还原
        
        参数:
        u: 归一化数据
        u_max: 数据最大值
        u_min: 数据最小值
        
        返回:
        反归一化后的数据
        """
        return (u+1)/2*(u_max - u_min)+u_min
        
    def uvw_informer(self,u, u_mean, u_std):
        """
        速度场数据归一化函数
        
        参数:
        u: 原始速度数据
        u_mean: 速度均值
        u_std: 速度标准差
        
        返回:
        归一化后的速度数据
        """
        return u/u_std
        
    def uvw_outformer(self,u, u_mean, u_std):
        """
        速度场数据反归一化函数
        
        参数:
        u: 归一化速度数据
        u_mean: 速度均值
        u_std: 速度标准差
        
        返回:
        反归一化后的速度数据
        """
        return u*u_std #+u_mean

    ## 自动求导机制
    def autograd(self, U, x, create_graph=True):
        """
        自动求导函数，计算张量的梯度
        
        参数:
        U: 待求导的张量
        x: 求导变量
        create_graph: 是否创建计算图
        
        返回:
        梯度值
        """
        return torch.autograd.grad(U, x, grad_outputs=torch.ones_like(U), retain_graph=True, create_graph=create_graph)[0]

    ## compute_fun_loss and data pre-processing 计算函数损失和数据预处理
    def compute_fun_loss(self, model_batches, data_mean, data_max, data_min, data_std):
        """
        计算函数损失和数据预处理函数
        
        参数:
        model_batches: 模型批次数据
        data_mean: 数据均值
        data_max: 数据最大值
        data_min: 数据最小值
        data_std: 数据标准差
        
        返回:
        总损失、各分项损失和系数
        """
        ### data process 数据处理
        f_mean, u_mean, v_mean, w_mean, wss_mean = data_mean
        f_max, u_max, v_max, w_max, wss_max = data_max
        f_min, u_min, v_min, w_min, wss_min = data_min
        f_std, u_std, v_std, w_std,wss_std = data_std
        model_batches[:, 0:4] = model_batches[:, 0:4] * self.xyzt
        # model_data 模型数据
        x, y, z, t = model_batches[:, 0:1], model_batches[:, 1:2], model_batches[:, 2:3], model_batches[:, 3:4]
        f_cfd = self.data_informer(model_batches[:, 4:5], f_max, f_min)
        wss_cfd = self.data_informer(model_batches[:, 8:9], wss_max, wss_min)

        '''
        u_kk = torch.abs(f_cfd)
        idx_u = torch.where(u_kk < 1e-3)[0]
        print(idx_u.shape)
        print(torch.max(f_cfd),torch.max(u_cfd), torch.max(v_cfd), torch.max(w_cfd))
        print(torch.min(f_cfd), torch.min(u_cfd), torch.min(v_cfd), torch.min(w_cfd))
        print(torch.mean(f_cfd), torch.mean(u_cfd), torch.mean(v_cfd), torch.mean(w_cfd))'''

        # Split t and x to compute partial derivatives 分割t和x以计算偏导数
        t.requires_grad = True
        x.requires_grad = True
        y.requires_grad = True
        z.requires_grad = True

        # 网络前向传播
        uvpa ,coef= self.net(torch.concat([x, y, z, t], 1))
        #uvw, f, wss = uvpa[0], uvpa[1], uvpa[2]
        u, v, w , f, wss=  uvpa[0], uvpa[1], uvpa[2], uvpa[3], uvpa[4]
        
        # get data loss 获取数据损失
        f_loss = self.compute_data_loss_mse(f, f_cfd)
        wss_loss = self.compute_data_loss_mse(wss, wss_cfd)
        loss_data = f_loss*self.loss_weights_Data[0] + wss_loss * self.loss_weights_Data[1]

        # boundary loss BC 边界损失
        loss_bc = self.fun_u_b(u, v, w)

        # to compute derivatives u_t and u_x 计算导数u_t和u_x
        # u = torch.tensor(u,requires_grad=True)
        # Compute gradient u_x within the GradientTape 在GradientTape中计算梯度u_x
        # since we need second derivatives 因为我们需要二阶导数
        u_t = self.autograd(u, t, create_graph=True)
        v_t = self.autograd(v, t, create_graph=True)
        w_t = self.autograd(w, t, create_graph=True)
        u_x = self.autograd(u, x, create_graph=True)
        u_y = self.autograd(u, y, create_graph=True)
        u_z = self.autograd(u, z, create_graph=True)
        w_x = self.autograd(w, x, create_graph=True)
        w_y = self.autograd(w, y, create_graph=True)
        w_z = self.autograd(w, z, create_graph=True)
        v_x = self.autograd(v, x, create_graph=True)
        v_y = self.autograd(v, y, create_graph=True)
        v_z = self.autograd(v, z, create_graph=True)
        f_x = self.autograd(f, x, create_graph=True)
        f_y = self.autograd(f, y, create_graph=True)
        f_z = self.autograd(f, z, create_graph=True)
        u_xx = self.autograd(u_x, x, create_graph=True)
        u_yy = self.autograd(u_y, y, create_graph=True)
        u_zz = self.autograd(u_z, z, create_graph=True)
        w_xx = self.autograd(w_x, x, create_graph=True)
        w_yy = self.autograd(w_y, y, create_graph=True)
        w_zz = self.autograd(w_z, z, create_graph=True)
        v_xx = self.autograd(v_x, x, create_graph=True)
        v_yy = self.autograd(v_y, y, create_graph=True)
        v_zz = self.autograd(v_z, z, create_graph=True)
        # 清空GPU缓存
        torch.cuda.empty_cache()

        # N-S equ loss N-S方程损失
        loss_equ_u = self.fun_xy(u, v, w, u_t, u_x, u_y, u_z, f_x, u_xx, u_yy, u_zz)
        loss_equ_v = self.fun_xy(u, v, w, v_t, v_x, v_y, v_z, f_y, v_xx, v_yy, v_zz)
        loss_equ_w = self.fun_z(u, v, w, w_t, w_x, w_y, w_z, f_z, w_xx, w_yy, w_zz)
        loss_equ_u = torch.mean(loss_equ_u ** 2)
        loss_equ_v = torch.mean(loss_equ_v ** 2)
        loss_equ_w = torch.mean(loss_equ_w ** 2)
        loss_equ = loss_equ_u + loss_equ_v + loss_equ_w

        # continuous equ loss 连续性方程损失
        loss_continuous = self.fun_u_0(u_x, v_y, w_z)
        loss_continuous = torch.mean(loss_continuous ** 2)

        # PINN损失
        loss_pinn = loss_equ*self.loss_weights_Equ[0] + loss_continuous*self.loss_weights_Equ[1] + loss_bc*self.loss_weights_Equ[2]
        # total loss 总损失
        torch.cuda.empty_cache()
        loss_fun_total =  loss_data + loss_pinn
        return loss_fun_total, [loss_fun_total, f_loss, wss_loss, loss_equ, loss_continuous, loss_bc], coef

    ## get data 获取数据
    def set_batch_data(self, model_data_batch):
        """
        获取数据的函数
        
        参数:
        model_data_batch: 模型数据批次
        
        返回:
        处理后的模型数据集、数据均值、最大值、最小值和标准差
        """
        pressure_data, u_data, v_data, w_data, wss_data , xyz = model_data_batch
        model_data_sets = None
        '''
        #pv_data_sets = None
        p_mean, u_mean, v_mean, w_mean, wss_mean = np.mean(pressure_data), np.mean(u_data), np.mean(v_data), np.mean(w_data), np.mean(wss_data)
        data_mean = [p_mean, u_mean, v_mean, w_mean, wss_mean]
        p_max, u_max, v_max, w_max, wss_max = np.max(pressure_data), np.max(u_data), np.max(v_data), np.max(w_data), np.max(wss_data)
        data_max = [p_max, u_max, v_max, w_max, wss_max]
        p_min, u_min, v_min, w_min, wss_min = np.min(pressure_data), np.min(u_data), np.min(v_data), np.min(w_data), np.min(wss_data)
        data_min = [p_min, u_min, v_min, w_min, wss_min]
        p_std, u_std, v_std, w_std, wss_std = np.std(pressure_data), np.std(u_data), np.std(v_data), np.std(w_data), np.std(wss_data)
        data_std = [p_std, u_std, v_std, w_std, wss_std]
        print('mean:',data_mean)
        print('max:',data_max)
        print('min:',data_min)
        print('std:', data_std)'''
        
        # 预设的数据统计信息
        data_mean = [-36.0, 0.0024, 0.0010, -0.042, 0.7]
        data_max = [300, 0.8, 0.8, 0.3, 30]
        data_min = [-1200.0, -0.7, -0.7, -1.0, 0.002]
        data_std = [420.0, 0.06, 0.06, 0.09, 1.4]
        
        # 构建时间序列数据
        for idx in range(self.strat, self.period+self.strat):
            if idx < self.strat:
                continue
            if model_data_sets is None:
                pressure,wss = pressure_data[:,idx:idx+1], wss_data[:,idx:idx+1]
                u ,v ,w = u_data[:,idx:idx+1], v_data[:,idx:idx+1], w_data[:,idx:idx+1]
                # model_data 模型数据
                model_time = (idx+1) * 0.08 * (np.ones_like(pressure) + np.finfo(np.float32).eps )
                model_data_sets = np.concatenate([xyz, model_time, pressure, u, v, w, wss],axis= 1)
                # pv_data
                #pv_time = model_time[:int(pv_xyz_batch[0].shape[0]*0.5)]
                #pv = np.random.choice(int(model_time.shape[0]*0.9), int(pv_time.shape[0]), replace=False)
                #pv_xyz_batch = np.array(pv_xyz_batch[0][pv])
                #pv_data_sets = np.concatenate([np.array(pv_xyz_batch[0][pv]), pv_time],axis= 1)
            else:
                if idx == self.period:
                    break
                pressure,wss = pressure_data[:, idx:idx+1], wss_data[:,idx:idx+1]
                u, v, w = u_data[:, idx:idx+1], v_data[:, idx:idx+1], w_data[:, idx:idx+1]
                model_time = (idx + 1) * 0.08 * (np.ones_like(pressure) + np.finfo(np.float32).eps)
                #pv_time = model_time[:int(pv_xyz_batch[0].shape[0]*0.5)]
                #pv = np.random.choice(int(model_time.shape[0]*0.9), int(pv_time.shape[0]), replace=False)
                #pv_xyz_batch = np.array(pv_xyz_batch[0][pv])
                # model_data 模型数据
                model_data_one = np.concatenate([xyz, model_time, pressure, u, v, w, wss], axis=1)
                model_data_sets = np.concatenate([model_data_sets, model_data_one],axis=0)
                # pv_data
                #pv_data_one = np.concatenate([np.array(pv_xyz_batch[0][pv]), pv_time], axis=1)
                #pv_data_sets = np.concatenate([pv_data_sets, pv_data_one],axis=0)
                
        # 验证数据维度
        assert model_data_sets.shape[0] == pressure_data.shape[0] * self.period, 'point_number will not correspond'
        return model_data_sets, data_mean, data_max, data_min, data_std

    ## shuffle data and get self.point_number of data 数据打乱和获取指定数量数据
    def shuffle_data_and_reset_epoch_losses(self, model_data_batch):
        """
        数据打乱和重置epoch损失的函数
        
        参数:
        model_data_batch: 模型数据批次
        
        返回:
        打乱后的模型数据批次
        """
        # pv_data
        #length_pv = pv_data_batch.shape[0]
        #shuffled_indices_pv = np.random.choice(length_pv, int(self.point_number* self.period * self.pv), replace=False)
        # model_data 模型数据
        length_model = model_data_batch.shape[0]
        # 随机选择指定数量的点
        shuffled_indices_model = np.random.choice(length_model, int(self.point_number * self.period), replace=False)
        #pv_data_batch = pv_data_batch[shuffled_indices_pv, :]
        model_data_batch = model_data_batch[shuffled_indices_model, :]
        return model_data_batch

    ## set_optimizers 设置优化器
    def set_optimizers(self, counter):
        """
        设置优化器的函数
        
        参数:
        counter: 计数器
        """
        # 如果存在预训练参数，则加载
        if self.net_params:
            load_params = torch.load(self.net_params)
            self.net.load_state_dict(load_params)
        # 创建Adam优化器
        self.opt = torch.optim.Adam(params=self.net.parameters(),lr=self.learning_rates[counter],weight_decay=0.005)

    ## get batch_size of data and data pre-processing 获取批次数据并进行预处理
    def get_batches(self, model_data_batch, b, batch_sizes):
        """
        获取批次数据的函数
        
        参数:
        model_data_batch: 模型数据批次
        b: 批次索引
        batch_sizes: 批次大小
        
        返回:
        指定批次的数据
        """
        #pv_batch_size = int(batch_sizes * self.pv)
        #pv_batches = torch.tensor(pv_data_batch[b * pv_batch_size:(b + 1) * pv_batch_size , :], dtype = torch.float32).cuda(self.device)
        model_batches = torch.tensor(model_data_batch[b * batch_sizes:(b + 1) * batch_sizes, :], dtype = torch.float32).cuda(self.device)

        return model_batches

    ## save batch_size of loss 保存批次损失
    def assign_batch_losses(self, batch_losses):
        """
        保存批次损失的函数
        
        参数:
        batch_losses: 批次损失列表
        """
        for loss_values, key in zip(batch_losses, self.epoch_loss):
            self.epoch_loss[key] += loss_values

    ## save batch_size of history loss 保存历史损失
    def append_loss_and_activation_coeff_history(self, tr_step, coef):
        """
        保存损失和系数历史的函数
        
        参数:
        tr_step: 训练步数
        coef: 系数
        """
        for key in self.loss_history:
            if self.epoch_loss[key] == 0:
                self.loss_history[key].append(self.epoch_loss[key])
                continue
            else:
                self.epoch_loss[key] = self.epoch_loss[key].detach().cpu().numpy() / tr_step
            self.loss_history[key].append(self.epoch_loss[key])
        #if self.use_coef:
        for i,key in enumerate(self.coef_history):
            #self.epoch_coef[key] = self.epoch_coef[key].detach().cpu().numpy()
            self.coef_history[key].append(coef[i].item())

    ## training 训练
    def train(self, mdoel_data_sets):
        """
        训练函数
        
        参数:
        mdoel_data_sets: 模型数据集
        """
        print('\nSTART---training----')
        print("\nEPOCHS: ", self.epochs, " BATCH SIZES: ", self.batch_sizes, " LEARNING RATES: ", self.learning_rates)
        start_total = time.time()
        epoch_list = [1]
        coef = None
        # 遍历不同阶段的训练参数
        for counter, epoch_value in enumerate(self.epochs):
            self.set_optimizers(counter)
            epoch_list.append(epoch_value)
            # 训练指定轮数
            for e in range(1, epoch_value + 1):
                start_epoch = time.time()
                tr_step = 0
                # 处理每个数据批次
                for model_data_batch  in zip(mdoel_data_sets):
                    model_data_batch, data_mean, data_max, data_min, data_std = self.set_batch_data(model_data_batch[0])
                    model_data_batch = self.shuffle_data_and_reset_epoch_losses(model_data_batch)
                    num_batches = int(model_data_batch.shape[0] / self.batch_sizes[0])
                    # 分批次训练
                    for b in range(num_batches):
                        self.opt.zero_grad()
                        model_batches = self.get_batches(model_data_batch, b, self.batch_sizes[0])
                        batch_losses, loss, coef = self.compute_fun_loss(model_batches, data_mean, data_max, data_min, data_std)
                        batch_losses.backward()
                        self.opt.step()
                        tr_step += 1
                        self.assign_batch_losses(loss)
                # 保存损失历史
                self.append_loss_and_activation_coeff_history(tr_step,coef)
                print(time.time() - start_epoch, "s")
                print("current lr is {}".format(self.opt.state_dict()['param_groups'][0]['lr']))
                number_epochs = epoch_list[-2] * counter + e - 1
                # 打印训练信息
                print("epoch/num_epoch: %d/%d %d  loss_total[Adam]: %.7f p_loss: %.7f wss_loss: %.7f loss_equ: %.7f"
                      "loss_continuous: %.7f loss_bc: %.7f p_coef: %.7f wss_coef: %.7f"
                      %(e,epoch_value, counter+1,self.loss_history['total'][number_epochs],self.loss_history['f_loss'][number_epochs],self.loss_history['wss_loss'][number_epochs],self.loss_history['loss_equ'][number_epochs],
                        self.loss_history['loss_continuous'][number_epochs],self.loss_history['loss_bc'][number_epochs],self.coef_history['p_coef'][number_epochs],self.coef_history['wss_coef'][number_epochs]))
                # 保存最佳模型
                if self.loss_history['total'][number_epochs] < self.loss_epochs:
                    torch.save(self.net.state_dict(),'weight/' + 'PINN_CFD_BLOOD_weight_50.pth')
                    self.loss_epochs = self.loss_history['total'][number_epochs]
                self.net_params = 'weight/' + 'PINN_CFD_BLOOD_weight_50.pth'
        print(time.time() - start_total, "s")
        # 保存训练历史
        data = {}
        data.update(self.loss_history)
        data.update(self.coef_history)
        loss_history = pd.DataFrame(data = data)
        loss_history.to_csv('weight/history_50.csv', index=False)

def main():
    """
    主函数
    """
    # 加载模型数据
    model_data = pickle.load(open('pv_model_50.pkl', 'rb'))

    opt_params = None
    net_params = None
    use_coef = True
    use_res = True

    # HYPERPARAMETERS FOR TRAINING 训练超参数
    res = [0.1, 0.1]#p,wss
    loss_weights_Data =  [40.0, 25.0]#p,wss
    loss_weights_Equ = [5.0, 1.0, 40.0, 1.0]
    data_former = [1, 1, 1, 1]
    epochs = [500, 500, 500]
    batch_sizes = [500]
    learning_rates = [3e-4,1e-4,1e-5]
    Re = 300
    point_number = 50
    period = 10
    start = 0

    # 创建Pinn_Cfd_Blood实例
    PINN = Pinn_Cfd_Blood(loss_weights_Data, loss_weights_Equ, Re, point_number, data_former,
                        epochs, batch_sizes, learning_rates, net_params, opt_params, start, period, use_coef, use_res, res)

    # TRAINING 训练
    PINN.train(model_data)

if __name__ == "__main__":
    main()