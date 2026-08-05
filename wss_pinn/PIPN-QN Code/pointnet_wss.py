import torch.nn as nn  # 导入神经网络模块
import torch  # 导入PyTorch库
import torch.nn.functional as F  # 导入函数模块
import pdb  # 导入Python调试器

class get_model(nn.Module):
    """
    PointNet模型类，用于处理点云数据并预测血流参数
    
    该模型基于PointNet架构，专门设计用于处理三维空间中的点云数据，
    预测流体动力学中的速度分量、压力和壁面剪切应力(WSS)等参数。
    模型支持两种模式：使用残差连接(Qres)或普通深度网络(dnet)。
    """
    def __init__(self, ues_coef, use_res, p_res, wss_res):
        """
        初始化PointNet模型
        
        参数:
            ues_coef: bool类型，是否在计算中使用系数缩放
            use_res: bool类型，是否使用Qres残差网络结构
            p_res: float类型，压力参数的初始值
            wss_res: float类型，壁面剪切应力参数的初始值
        """
        super(get_model, self).__init__()
        self.ues_coef = ues_coef  # 是否使用系数进行缩放
        self.use_res = use_res  # 是否使用残差网络结构
        
        # 定义基础特征提取网络层
        # 输入维度为4 (x, y, z, t)，逐步提取更高维特征
        self.layer_p1 = nn.Sequential(
            nn.Linear(4, 32), nn.Tanh(), 
            nn.Linear(32, 64), nn.Tanh(), 
            nn.Linear(64, 128)
        )
        
        # 进一步提取全局特征
        self.layer_p2 = nn.Sequential(
            nn.Linear(128, 256), nn.Tanh(), 
            nn.Linear(256, 512)
        )
        
        # 定义各个物理量的输出网络层
        # 输入维度为512+128=640，结合全局和局部特征
        self.layer_u1 = nn.Sequential(
            nn.Linear(512 + 128, 128), nn.Tanh(), 
            nn.Linear(128, 32), nn.Tanh(), 
            nn.Linear(32, 1)  # 输出u速度分量
        )
        
        self.layer_v1 = nn.Sequential(
            nn.Linear(512 + 128, 128), nn.Tanh(), 
            nn.Linear(128, 32), nn.Tanh(), 
            nn.Linear(32, 1)  # 输出v速度分量
        )
        
        self.layer_w1 = nn.Sequential(
            nn.Linear(512 + 128, 128), nn.Tanh(), 
            nn.Linear(128, 32), nn.Tanh(), 
            nn.Linear(32, 1)  # 输出w速度分量
        )
        
        # 压力(p)输出相关层
        self.p_6 = nn.Linear(32, 1)  # 将32维特征映射到1维输出
        self.p = nn.Parameter(torch.tensor([p_res]))  # 可学习的压力参数
        
        # 壁面剪切应力(wss)输出相关层
        self.wss_6 = nn.Linear(32, 1)  # 将32维特征映射到1维输出
        self.wss = nn.Parameter(torch.tensor([wss_res]))  # 可学习的wss参数
        
        if self.use_res:
            # 使用Qres残差网络结构时的参数配置
            # 压力分支的残差网络参数
            self.bias_p1 = nn.Parameter(torch.Tensor(128))  # 第一层偏置
            self.bias_p2 = nn.Parameter(torch.Tensor(32))   # 第二层偏置
            self.bias_p = [self.bias_p1, self.bias_p2]      # 偏置参数列表
            
            # 两个并行的隐藏层序列，用于Qres计算
            self.hidden_p1 = nn.ModuleList([
                nn.Linear(512 + 128, 128, bias=False),  # 无偏置的线性变换
                nn.Linear(128, 32, bias=False)
            ])
            
            self.hidden_p2 = nn.ModuleList([
                nn.Linear(512 + 128, 128, bias=False),
                nn.Linear(128, 32, bias=False)
            ])
            
            # 壁面剪切应力分支的残差网络参数
            self.bias_w1 = nn.Parameter(torch.Tensor(128))  # 第一层偏置
            self.bias_w2 = nn.Parameter(torch.Tensor(32))   # 第二层偏置
            self.bias_w = [self.bias_w1, self.bias_w2]      # 偏置参数列表
            
            # 两个并行的隐藏层序列，用于Qres计算
            self.hidden_w1 = nn.ModuleList([
                nn.Linear(512 + 128, 128, bias=False),
                nn.Linear(128, 32, bias=False)
            ])
            
            self.hidden_w2 = nn.ModuleList([
                nn.Linear(512 + 128, 128, bias=False),
                nn.Linear(128, 32, bias=False)
            ])
        else:
            # 不使用残差网络时，使用普通的深度网络结构
            # 压力分支的网络层
            self.moudle_y_3 = nn.ModuleList([
                nn.Linear(512 + 128, 128),  # 包含偏置的标准线性层
                nn.Linear(128, 32)
            ])
            
            # 壁面剪切应力分支的网络层
            self.moudle_wss_3 = nn.ModuleList([
                nn.Linear(512 + 128, 128),
                nn.Linear(128, 32)
            ])

    def forward(self, xyz):
        """
        前向传播函数
        
        参数:
            xyz: 输入张量，形状为[N, 4]，包含N个点的坐标和时间信息
                 其中4维分别代表x, y, z坐标和时间t
        
        返回:
            outputs: 包含5个元素的列表，分别是[u, v, w, p, wss]的预测值
            coef: 包含[p, wss]可学习参数的列表
        """
        N, _ = xyz.shape  # 获取输入点的数量N
        
        # 特征提取阶段
        y1 = self.layer_p1(xyz)  # 通过第一层网络提取特征，得到[N, 128]的输出
        y2 = self.layer_p2(y1)   # 通过第二层网络进一步提取特征，得到[N, 512]的输出
        
        # 全局最大池化操作，提取全局特征
        # 对每个特征维度取所有点中的最大值，得到[512]的全局描述符
        y = torch.max(y2, 0)[0]
        y = y.unsqueeze(0)  # 增加批次维度，变为[1, 512]
        
        # 将全局特征广播到每个点
        # 将[1, 512]的全局特征复制N份，变为[N, 512]
        y = torch.broadcast_to(y, [N, 512])
        
        # 特征拼接：将全局特征与局部特征结合
        # 将[N, 512]的全局特征与[N, 128]的局部特征在最后一维拼接
        # 得到[N, 640]的综合特征表示
        y = torch.concat((y, y1), dim=-1)
        
        # 初始化输出列表
        outputs = []
        
        # 计算三个速度分量
        outputs.append(self.layer_u1(y))  # u速度分量
        outputs.append(self.layer_v1(y))  # v速度分量
        outputs.append(self.layer_w1(y))  # w速度分量
        
        # 根据是否使用残差网络选择不同的计算路径
        if self.use_res:
            # 使用Qres残差网络计算压力和wss
            # 通过Qres_net处理特征，然后通过线性层得到最终输出
            outputs.append(self.p_6(self.Qres_net(y, self.p, self.hidden_p1, self.hidden_p2, self.bias_p)))
            outputs.append(self.wss_6(self.Qres_net(y, self.wss, self.hidden_w1, self.hidden_w2, self.bias_w)))
        else:
            # 使用普通深度网络计算压力和wss
            # 通过dnet处理特征，然后通过线性层得到最终输出
            outputs.append(self.p_6(self.dnet(y, self.moudle_y_3, self.p)))
            outputs.append(self.wss_6(self.dnet(y, self.moudle_wss_3, self.wss)))
            
        # 收集可学习参数
        coef = [self.p, self.wss]
        
        return outputs, coef

    def dnet(self, uvwp, moudel, coef):
        """
        普通深度网络处理函数
        
        该函数实现了一个简单的前馈神经网络，用于处理特征数据。
        
        参数:
            uvwp: 输入特征张量
            moudel: 网络层列表(ModuleList)
            coef: 系数参数，用于控制网络行为
        
        返回:
            处理后的特征张量
        """
        if self.ues_coef:
            # 如果使用系数，则在激活函数前对线性变换结果进行缩放
            for layer in moudel:
                uvwp = torch.tanh(coef * 10 * layer(uvwp))
        else:
            # 不使用系数时，标准的前馈计算
            for layer in moudel:
                uvwp = torch.tanh(layer(uvwp))
        return uvwp

    def Qres_net(self, uvwp, coef, hidden1, hidden2, bias):
        """
        Qres残差网络处理函数
        
        实现了一种特殊的残差网络结构，用于增强模型的表达能力。
        
        参数:
            uvwp: 输入特征张量
            coef: 系数参数
            hidden1: 第一组隐藏层
            hidden2: 第二组隐藏层
            bias: 偏置参数列表
        
        返回:
            处理后的特征张量
        """
        if self.ues_coef:
            # 使用系数的情况
            for nums in range(len(hidden1)):
                layer1 = hidden1[nums]  # 获取当前层的第一个线性变换
                layer2 = hidden2[nums]  # 获取当前层的第二个线性变换
                bias_val = bias[nums]   # 获取当前层的偏置
                # 应用Qres计算并进行非线性激活
                uvwp = torch.tanh(coef * 10 * self.Qres(uvwp, layer1, layer2, bias_val))
        else:
            # 不使用系数的情况
            for nums in range(len(hidden1)):
                layer1 = hidden1[nums]
                layer2 = hidden2[nums]
                bias_val = bias[nums]
                # 应用Qres计算并进行非线性激活
                uvwp = torch.tanh(self.Qres(uvwp, layer1, layer2, bias_val))
        return uvwp

    def Qres(self, input, layer1, layer2, bias):
        """
        Qres基本计算单元
        
        实现了特殊的残差计算：h₁ ⊙ h₂ + (h₁ + b)
        其中⊙表示逐元素乘法，h₁和h₂是两个并行的线性变换结果。
        
        参数:
            input: 输入张量
            layer1: 第一个线性变换层
            layer2: 第二个线性变换层
            bias: 偏置参数
        
        返回:
            经过Qres计算的输出张量
        """
        h_1 = layer1(input)  # 第一个线性变换
        h_2 = layer2(input)  # 第二个线性变换
        # 返回逐元素乘法结果加上第一个变换结果与偏置的和
        return torch.add(torch.mul(h_1, h_2), layer1(input) + bias)

class get_loss(nn.Module):
    """
    损失函数计算类
    用于计算预测值与真实值之间的均方误差(MSE)和平均绝对误差(MAE)
    """
    def __init__(self):
        """
        初始化损失函数模块
        """
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        """
        计算损失函数
        
        参数:
            pred: 预测值张量
            target: 真实目标值张量
        
        返回:
            mse: 均方误差
            mae: 平均绝对误差
        """
        e = pred - target  # 计算误差
        mse = torch.mean(e**2)  # 均方误差
        mae = torch.mean(torch.abs(e))  # 平均绝对误差
        return mse, mae
