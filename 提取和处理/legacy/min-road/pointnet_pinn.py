import torch
# from pinn import get_pinn
import pandas as pd
import numpy as np
import pointnet2_ssg as pointnet2
from dataset import pointdata, norm_data
from torch.utils.data import Dataset, DataLoader
import pickle
import os
import datetime

# 设置环境变量以减少TensorFlow日志输出
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 定义雷诺数常量（流体仿真中的无量纲参数）
Re = 300

def fun_u_0(u_x, v_y, w_z):
    """
    连续性方程（不可压缩流体的质量守恒）
    在不可压缩流体中，速度场的散度应该为零
    
    参数:
    u_x: u方向速度对x方向的偏导数
    v_y: v方向速度对y方向的偏导数
    w_z: w方向速度对z方向的偏导数
    
    返回:
    连续性方程的残差值
    """
    return u_x + v_y + w_z

# Define boundary condition   边界条件函数
def fun_u_b(u, v, w):
    """
    边界条件函数，计算速度的绝对值之和
    
    参数:
    u: x方向速度分量
    v: y方向速度分量
    w: z方向速度分量
    
    返回:
    速度绝对值之和
    """
    return torch.abs(u) + torch.abs(v) + torch.abs(w)

# Define residual of the PDE 偏微分方程残差函数
def fun_r(u, v, w, u_t, u_x, u_y, u_z, f_x, u_xx, u_yy, u_zz):
    """
    Navier-Stokes方程的残差计算函数（x方向动量方程）
    
    参数:
    u, v, w: 三个方向的速度分量
    u_t: u对时间t的偏导数
    u_x, u_y, u_z: u对空间坐标x, y, z的偏导数
    f_x: 压力对x方向的偏导数
    u_xx, u_yy, u_zz: u对各方向的二阶偏导数
    
    返回:
    NS方程的残差
    """
    # NS方程: ∂u/∂t + (u·∇)u = -∇p/ρ + ν∇²u
    # 其中ν是运动粘度，与雷诺数Re相关
    return u_t + u * u_x + v * u_y + w * u_z + f_x - (u_xx + u_yy + u_zz) / Re

def get_pinn(model, pv, labels):
    """
    计算PINN（物理信息神经网络）的损失函数
    
    参数:
    model: 神经网络模型
    pv: 输入点云数据 (时间t和空间坐标x,y,z)
    labels: 标签数据
    
    返回:
    pde_loss: PDE残差损失
    """
    # 分离时间和空间坐标
    t, x, y, z = pv[:, 0:1, :], pv[:, 1:2, :], pv[:, 2:3, :], pv[:, 3:4, :]
    
    # 设置需要计算梯度的变量
    t.requires_grad = True
    x.requires_grad = True
    y.requires_grad = True
    z.requires_grad = True

    # 重新组合输入数据
    pv = torch.cat([t, x], -2)
    pv = torch.cat([pv, y], -2)
    pv = torch.cat([pv, z], -2)
    
    # 通过模型获取预测值：压力f和三个方向的速度u,v,w
    logits = model(pv)
    f = logits[:, 0:1, :]   # 压力
    u = logits[:, 1:2, :]   # x方向速度
    v = logits[:, 2:3, :]   # y方向速度
    w = logits[:, 3:4, :]   # z方向速度
    
    # 计算一阶偏导数
    # 时间导数
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(t), retain_graph=True)[0]
    v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(t), retain_graph=True)[0]
    w_t = torch.autograd.grad(w, t, grad_outputs=torch.ones_like(t), retain_graph=True)[0]
    
    # 空间一阶导数
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(x), retain_graph=True, create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(y), retain_graph=True, create_graph=True)[0]
    u_z = torch.autograd.grad(u, z, grad_outputs=torch.ones_like(z), retain_graph=True, create_graph=True)[0]
    
    w_x = torch.autograd.grad(w, x, grad_outputs=torch.ones_like(x), retain_graph=True, create_graph=True)[0]
    w_y = torch.autograd.grad(w, y, grad_outputs=torch.ones_like(y), retain_graph=True, create_graph=True)[0]
    w_z = torch.autograd.grad(w, z, grad_outputs=torch.ones_like(z), retain_graph=True, create_graph=True)[0]
    
    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(x), retain_graph=True, create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(y), retain_graph=True, create_graph=True)[0]
    v_z = torch.autograd.grad(v, z, grad_outputs=torch.ones_like(z), retain_graph=True, create_graph=True)[0]
    
    f_x = torch.autograd.grad(f, x, grad_outputs=torch.ones_like(x), retain_graph=True)[0]
    f_y = torch.autograd.grad(f, y, grad_outputs=torch.ones_like(y), retain_graph=True)[0]
    f_z = torch.autograd.grad(f, z, grad_outputs=torch.ones_like(z), retain_graph=True)[0]
    
    # 计算二阶偏导数（用于粘性项）
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(x), retain_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(y), retain_graph=True)[0]
    u_zz = torch.autograd.grad(u_z, z, grad_outputs=torch.ones_like(z), retain_graph=True)[0]
    
    w_xx = torch.autograd.grad(w_x, x, grad_outputs=torch.ones_like(x), retain_graph=True)[0]
    w_yy = torch.autograd.grad(w_y, y, grad_outputs=torch.ones_like(y), retain_graph=True)[0]
    w_zz = torch.autograd.grad(w_z, z, grad_outputs=torch.ones_like(z), retain_graph=True)[0]
    
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(x), retain_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(y), retain_graph=True)[0]
    v_zz = torch.autograd.grad(v_z, z, grad_outputs=torch.ones_like(z), retain_graph=True)[0]
    
    # 计算各个方向的PDE残差
    r1 = fun_r(u, v, w, u_t, u_x, u_y, u_z, f_x, u_xx, u_yy, u_zz)  # x方向动量方程
    r2 = fun_r(u, v, w, v_t, v_x, v_y, v_z, f_y, v_xx, v_yy, v_zz)  # y方向动量方程
    r3 = fun_r(u, v, w, w_t, w_x, w_y, w_z, f_z, w_xx, w_yy, w_zz)  # z方向动量方程
    r4 = fun_u_0(u_x, v_y, w_z)  # 连续性方程
    
    # 返回所有方程残差的均方误差之和
    return torch.mean((r1) ** 2) + torch.mean((r2) ** 2) + torch.mean((r3) ** 2) + torch.mean((r4) ** 2)

def train_step(model, optimizer, x, labels):
    """
    执行一个训练步骤
    
    参数:
    model: 神经网络模型
    optimizer: 优化器
    x: 输入数据
    labels: 标签数据
    
    返回:
    loss.item(): 总损失值
    metric.item(): 评估指标值
    lo.item(): 数据损失值
    func.item(): 物理约束损失值
    """
    # 设置模型为训练模式
    model.train()
    # 梯度清零
    optimizer.zero_grad()
    # 数据类型转换并移至GPU
    x = x.float().cuda()
    labels = labels.float().cuda()
    # 前向传播得到预测结果
    pred = model(x)

    # 计算数据损失和评估指标
    lo, metric = model.loss_func(pred, labels)
    # 计算物理约束损失（PINN损失）
    los = get_pinn(model, x, labels)
    # 总损失 = 数据损失 + 0.02 * 物理约束损失
    loss = lo + los * 0.02

    # 反向传播求梯度
    loss.backward()
    # 更新参数
    optimizer.step()
    return loss.item(), metric.item(), lo.item(), los.item()

def train(model, dl_train, epochs=100, start=1, lr=1e-4):
    """
    训练模型
    
    参数:
    model: 神经网络模型
    dl_train: 训练数据加载器
    epochs: 训练轮数
    start: 起始轮数
    lr: 学习率
    
    返回:
    dfhistory: 训练历史记录
    """
    # 创建用于记录训练历史的DataFrame
    dfhistory = pd.DataFrame(columns=["epoch", "loss", 'mae', 'mse', 'func'])
    # 获取当前时间用于日志记录
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # 设置模型的损失函数
    model.loss_func = pointnet2.get_loss()
    print("Start Training..." + "%s" % nowtime)
    # 设置日志打印频率
    log_step_freq = 20
    # 创建优化器（使用SGD优化器）
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.RMSprop(model.parameters(),lr=lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    # 开始训练循环
    for epoch in range(start, epochs + start):
        # 训练循环
        loss_sum = 0.0
        metric_sum = 0.0
        lo_sum = 0.0
        func_sum = 0.0
        step = 1
        # 遍历训练数据
        for step, (x, labels) in enumerate(dl_train, 1):
            # 执行训练步骤
            loss, metric, lo, func = train_step(model, optimizer, x, labels)
            # 累计损失和评估指标
            loss_sum += loss
            metric_sum += metric
            lo_sum += lo
            func_sum += func

        # 记录日志
        info = (epoch, loss_sum / step, metric_sum / step, lo_sum / step, func_sum / step)
        dfhistory.loc[epoch - 1] = info

        # 打印epoch级别日志
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("epoch = %d, loss = %.5f, mae = %.5f, mse= %.5f, function=%.5f" % info + "\t" + "==========" * 2 + "%s" % nowtime)
        # 每2个epoch保存一次模型
        if epoch % 2 == 0:
            torch.save(model.state_dict(), 'weight/pointnet_pinn_80_%d_%0.5f_%0.5f.pth' % (epoch, loss_sum / step, metric_sum / step))

    return dfhistory

if __name__ == "__main__":
    # 加载数据信息
    data_info = pickle.load(open('data.pkl', 'rb'))
    # 设置数据路径和参数
    input_path = 'pv_txt/'
    output_path = r'D:\pointnet- pinn\out.csv'
    # 设置每个样本使用的点数量
    point_number = 15000
    # 创建训练数据集和数据加载器
    data_train = pointdata(input_path, output_path, data_info, point_number)
    dl_train = DataLoader(data_train, batch_size=4, shuffle=True, num_workers=5)

    # 创建模型并移至GPU
    model = pointnet2.get_model()
    device = torch.device('cuda:0')
    # 如果需要可以加载预训练模型
    # model.load_state_dict(torch.load('weight/pointnet_pinn_300_200_0.01591_0.04843.pth'))
    # # # model.load_state_dict(torch.load('weight/pointnet_pinn_16_0.434_0.879.pth'))
    model.to(device)
    # 分阶段训练模型
    his1 = train(model, dl_train, epochs=30, start=1, lr=3e-3)
    his2 = train(model, dl_train, epochs=80, start=31, lr=1e-3)

    # 合并训练历史并保存到CSV文件
    his = pd.concat([his1, his2])
    his.to_csv("train_loss.csv", index=False)