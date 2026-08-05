import torch.nn as nn  # 导入神经网络模块
import torch  # 导入PyTorch库
import torch.nn.functional as F  # 导入函数模块
import pdb  # 导入Python调试器

class get_model(nn.Module):
    """
    PointNet模型类，用于处理点云数据
    """
    def __init__(self):
        """
        初始化PointNet模型
        """
        super(get_model, self).__init__()
        #self.ues_coef = ues_coef
        # 定义压力场处理层
        self.layer_p1 = nn.Sequential(nn.Linear(4, 32), nn.Tanh(), nn.Linear(32, 64), nn.Tanh(), nn.Linear(64, 128))
        self.layer_p2 = nn.Sequential(nn.Linear(128, 256), nn.Tanh(), nn.Linear(256, 512))
        # 激活函数
        self.layer_u1 = nn.Sequential(nn.Linear(512 + 128, 128), nn.Tanh(), nn.Linear(128, 32), nn.Tanh(), nn.Linear(32, 5))

    def forward(self, xyz):
        """
        前向传播函数
        
        参数:
            xyz: 输入的坐标数据 [N, 4] (x, y, z, t)
        
        返回:
            outputs: 网络输出
            _: 占位符
        """
        N, _ = xyz.shape
        # 处理压力场
        y1 = self.layer_p1(xyz)
        y2 = self.layer_p2(y1)
        # 全局最大池化
        y = torch.max(y2, 0)[0]
        y = y.unsqueeze(0)
        y = torch.broadcast_to(y, [N, 512])
        # 特征拼接
        y = torch.concat((y, y1), dim=-1)
        # 输出层
        outputs = self.layer_u1(y)

        return outputs, _

    def decoder(self,uvwp, moudel, coef, ues_coef):
        """
        解码器函数
        
        参数:
            uvwp: 输入数据
            moudel: 模型
            coef: 系数
            ues_coef: 是否使用系数
        
        返回:
            处理后的数据
        """
        if ues_coef:
            for layer in moudel:
                uvwp = torch.tanh( coef * layer(uvwp))
        else:
            for layer in moudel:
                uvwp = torch.tanh(layer(uvwp))
        return uvwp

class get_loss(nn.Module):
    """
    损失函数类
    """
    def __init__(self):
        """
        初始化损失函数
        """
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        """
        前向传播计算损失
        
        参数:
            pred: 预测值
            target: 目标值
        
        返回:
            mse: 均方误差
            mae: 平均绝对误差
        """
        e = pred - target
        mse = torch.mean(e**2)
        mae= torch.mean(torch.abs(e))
        return mse,mae