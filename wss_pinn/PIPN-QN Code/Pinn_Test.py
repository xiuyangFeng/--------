import torch  # 导入PyTorch库
from torch import nn  # 导入神经网络模块
import pandas as pd  # 导入Pandas库
import numpy as np  # 导入NumPy库
# 导入自定义的PointNet库
import pointnet_wss as pointnet_coef
import pickle  # 导入Pickle模块
import datetime  # 导入日期时间模块
import matplotlib.pyplot as plt  # 导入绘图库
import math  # 导入数学模块

# 定义Pinn_test类，用于血流仿真测试
class Pinn_test:
    def __init__(self):
        """
        初始化Pinn_test类
        """
        # 初始化均方误差（MSE）和平均绝对误差（MAE）相关参数
        self.mse_list = ["u", "v", "w", "p", "wss"]  # MSE列表
        #self.epoch_loss = dict.fromkeys(self.mse_list, 0)
        self.mse_history = dict((loss, []) for loss in self.mse_list)  # MSE历史记录

        self.mae_list = ["u", "v", "w", "p", "wss"]  # MAE列表
        # self.epoch_loss = dict.fromkeys(self.mse_list, 0)
        self.mae_history = dict((loss, []) for loss in self.mae_list)  # MAE历史记录

        # 初始化预测和CFD数据
        self.p_pred = None  # 压力预测数据
        self.wss_pred = None  # 壁面剪切应力预测数据
        self.p_cfd = None  # 压力CFD数据
        self.wss_cfd = None  # 壁面剪切应力CFD数据

        # 设置设备为GPU
        self.device = torch.device('cuda:0')
        # 定义损失函数
        self.MSE = nn.MSELoss()  # 均方误差损失函数
        self.MAE = nn.L1Loss()  # 平均绝对误差损失函数
        # 设置数据范围
        self.sa = 0  # 起始时间步
        self.pe = 10  # 结束时间步

    # 定义计算数据损失（MAPE）的函数
    def compute_data_loss_mape(self,u_cfd, u):
        """
        计算平均绝对百分比误差
        
        参数:
        u_cfd: CFD真实值
        u: 预测值
        
        返回:
        平均绝对百分比误差
        """
        u_cfd = self._tensor(u_cfd)
        mape = torch.mean(torch.abs((u - u_cfd) / u_cfd))
        u_cfd = u_cfd.detach().cpu()
        return mape

    # 定义计算数据损失（MSE和MAE）的函数
    def compute_data_loss_mse(self,u_cfd, u):
        """
        计算均方误差和平均绝对误差
        
        参数:
        u_cfd: CFD真实值
        u: 预测值
        
        返回:
        均方误差和平均绝对误差
        """
        u_cfd = torch.from_numpy(u_cfd).float().cuda(self.device)
        u = torch.from_numpy(u).float().cuda(self.device)
        mse = self.MSE(u_cfd, u)
        mae = self.MAE(u_cfd, u)
        u_cfd = u_cfd.detach().cpu()
        u = u.detach().cpu()
        return mse.item(), mae.item()

    # 定义数据反归一化函数
    def data_outformer(self,u, u_max, u_min):
        """
        数据反归一化函数
        
        参数:
        u: 归一化数据
        u_max: 数据最大值
        u_min: 数据最小值
        
        返回:
        反归一化后的数据
        """
        return (u + 1) / 2 * (u_max - u_min) + u_min

    # 定义数据归一化函数
    def data_informer(self,u, u_max, u_min):
        """
        数据归一化函数
        
        参数:
        u: 原始数据
        u_max: 数据最大值
        u_min: 数据最小值
        
        返回:
        归一化后的数据
        """
        return (u - u_min) / (u_max - u_min) * 2 - 1

    # 定义将数据转换为张量的函数
    def _tensor(self,cfd):
        """
        将数据转换为张量
        
        参数:
        cfd: 原始数据
        
        返回:
        张量数据
        """
        cfd = torch.from_numpy(cfd).float().cuda(self.device)
        return cfd

    # 定义数据信息处理函数
    def data_info(self,data,p_history):
        """
        数据信息处理函数
        
        参数:
        data: 数据
        p_history: 历史记录
        """
        for j,i in enumerate(p_history):
            if np.array(p_history[i]).shape == 0:
                p_history[i] = data[j].detach().cpu().squeeze()
            else:
                p_history[i] = np.concatenate((p_history[i], data[j].detach().cpu().squeeze()), axis=0)

    # 定义CFD数据信息处理函数
    def cfd_info(self,data,p_history):
        """
        CFD数据信息处理函数
        
        参数:
        data: 数据
        p_history: 历史记录
        """
        for j,i in enumerate(p_history):
            p_history[i] = data[j]

    # 定义预测步骤函数
    def pred_step(self,model,xyzt, batch_size,N, u_cfd, v_cfd, w_cfd,  p_cfd, wss_cfd):
        """
        预测步骤函数
        
        参数:
        model: 模型
        xyzt: 坐标和时间数据
        batch_size: 批次大小
        N: 数据数量
        u_cfd, v_cfd, w_cfd: 速度CFD数据
        p_cfd: 压力CFD数据
        wss_cfd: 壁面剪切应力CFD数据
        
        返回:
        预测历史记录和CFD历史记录
        """
        batch = math.ceil(N / batch_size)
        step = 0
        # 数据归一化参数
        data_mean = [-36.0, 0.0024, 0.0010, -0.042, 0.7]
        data_max = [300.0, 0.8, 0.8, 0.3, 30]
        data_min = [-1200.0, -0.7, -0.7, -1.0, 0.002]
        data_std = [420.0, 0.06, 0.06, 0.09, 1.4]
        f_max, u_max, v_max, w_max, wss_max = data_max
        f_min, u_min, v_min, w_min, wss_min = data_min
        # 初始化预测数据
        u_one, v_one, w_one, p_one, wss_one = None, None, None, None, None
        prediction = ["u", "v", "w", "p", "wss"]
        p_history = dict((i, []) for i in prediction)
        # 按批次进行预测
        for b in range(batch):
            N_number = np.arange(b * batch_size, (b + 1) * batch_size)
            N_number_info = int(N - b * batch_size)
            torch.cuda.empty_cache()
            if N_number_info < batch_size:
                xyz = xyzt[-N_number_info:, :]
                xyz = self._tensor(xyz)
                u_cfd_batch, v_cfd_batch, w_cfd_batch, p_cfd_batch = u_cfd[-N_number_info:], v_cfd[-N_number_info:], w_cfd[-N_number_info:], p_cfd[-N_number_info:]
                wss_cfd_batch = wss_cfd[-N_number_info:]
                uvwp, _ = model(xyz)
                self.data_info(uvwp,p_history)
            else:
                xyz = xyzt[N_number, :]
                xyz = self._tensor(xyz)
                u_cfd_batch, v_cfd_batch, w_cfd_batch, p_cfd_batch = u_cfd[N_number], v_cfd[N_number], w_cfd[N_number], p_cfd[N_number]
                wss_cfd_batch = wss_cfd[N_number]
                uvwp, _ = model(xyz)
                self.data_info(uvwp,p_history)
        # 处理CFD数据
        cfd = ["u", "v", "w", "p", "wss"]
        cfd_history = dict((i, []) for i in cfd)
        self.cfd_info([u_cfd, v_cfd, w_cfd,  p_cfd, wss_cfd], cfd_history)
        # 对压力和壁面剪切应力数据进行归一化处理
        cfd_history["p"] = self.data_informer(cfd_history["p"], f_max, f_min)
        cfd_history["wss"] = self.data_informer(cfd_history["wss"], wss_max, wss_min)
        return p_history, cfd_history

    # 定义绘图函数
    def draw(self,pred,cfd,title):
        """
        绘图函数
        
        参数:
        pred: 预测数据
        cfd: CFD数据
        title: 图标题
        """
        time = list(range(len(self.mse_history["p"])))
        figk = plt.figure(dpi=150)
        ax = figk.add_subplot(1, 1, 1)
        ax.plot(time, pred, c='y', label='pred', ls='--',linewidth = 2)
        ax.plot(time, cfd[self.sa:self.pe + self.sa], c='r', label='cfd')
        ax.set_title(title)
        ax.legend(frameon=False)

    # 定义PINN测试函数
    def pinn_test(self,data,batch_size,model):
        """
        PINN测试函数
        
        参数:
        data: 测试数据
        batch_size: 批次大小
        model: 模型
        """
        p_cfd, u_cfd, v_cfd, w_cfd, wss_cfd, xyz = data
        model.eval()
        for idx in range(self.sa, self.pe + self.sa):
            N, B = p_cfd.shape
            time = (idx + 1) * 0.08 * (np.ones_like(p_cfd[:, 0:1]) + np.finfo(np.float32).eps)
            xyzt = np.concatenate((xyz, time), axis=-1)
            xyzt = xyzt * 5
            p_history, cfd_history = self.pred_step(model,xyzt, batch_size,N, u_cfd[:, idx], v_cfd[:, idx], w_cfd[:, idx],  p_cfd[:, idx], wss_cfd[:,idx])
            # 处理预测和CFD数据
            if self.p_pred is None:
                self.p_pred = np.array(p_history["p"])
                self.p_cfd = np.array(cfd_history["p"])
            else:
                self.p_pred = np.column_stack((self.p_pred, np.array(p_history["p"])))
                self.p_cfd = np.column_stack((self.p_cfd, np.array(cfd_history["p"])))
            if self.wss_pred is None:
                self.wss_pred = np.array(p_history["wss"])
                self.wss_cfd = np.array(cfd_history["wss"])
            else:
                self.wss_pred = np.column_stack((self.wss_pred, np.array(p_history["wss"])))
                self.wss_cfd = np.column_stack((self.wss_cfd, np.array(cfd_history["wss"])))

            # 计算损失
            for key in ["u", "v", "w", "p", "wss"]:
                mse ,mae = self.compute_data_loss_mse( np.array(cfd_history[key]), np.array(p_history[key]))
                self.mse_history[key].append(mse)
                self.mae_history[key].append(mae)
        # 数据反归一化
        self.p_pred = self.data_outformer(self.p_pred, 300.0, -1200.0)
        self.p_cfd = self.data_outformer(self.p_cfd, 300.0, -1200.0)
        self.wss_pred = self.data_outformer(self.wss_pred, 30.0, 0.002)
        self.wss_cfd = self.data_outformer(self.wss_cfd, 30.0, 0.002)

        # 绘制结果图
        self.draw(np.mean(self.p_pred, axis=0), np.mean(self.p_cfd, axis=0), 'Pressure')
        self.draw(np.mean(self.wss_pred, axis=0), np.mean(self.wss_cfd, axis=0), 'WSS')

        # 打印损失结果
        print("Pressure")
        print('MSE：%.6f'%(np.mean(self.mse_history["p"])))
        print('MAE：%.6f'%(np.mean(self.mae_history["p"])))
        print("WSS")
        print('MSE：%.6f'%(np.mean(self.mse_history["wss"])))
        print('MAE：%.6f'%(np.mean(self.mae_history["wss"])))
        plt.show()

def main():
    """
    主函数
    """
    print("START---test---")
    ues_coef = True
    use_res = True
    # 设置权重文件路径
    res_s = 'weight/PINN_CFD_BLOOD_weight_3000.pth'
    res_T = 'weight/PINN_CFD_BLOOD_weight_1500_res.pth'
    res_F = 'weight/PINN_CFD_BLOOD_weight_1500.pth'
    res_a = 'weight/PINN_CFD_BLOOD_weight_1500_a.pth'
    res_r = 'weight/PINN_CFD_BLOOD_weight_1500_r.pth'
    res = [0.1, 0.1]
    # 获取模型
    model = pointnet_coef.get_model(ues_coef, use_res, res[0], res[1])
    device = torch.device('cuda:0')
    # 加载模型权重
    # model.load_state_dict(torch.load('weight/PINN_CFD_BLOOD_weight.pth'))
    model.load_state_dict(torch.load(res_T))
    model.to(device)

    model_data = pickle.load(open('pv_model_50.pkl', 'rb'))[0]
    batch_size = 50000

    test = Pinn_test()
    test.pinn_test(model_data,batch_size,model)

if __name__ == "__main__":
    main()