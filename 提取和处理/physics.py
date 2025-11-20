import torch

def compute_pinn_loss(u_pred, p_pred, pos, mask_inner, 
                      rho=1060.0, mu=0.0035, sample_size=2000):
    """
    计算 Navier-Stokes 方程残差。
    
    参数:
    - u_pred: 预测速度 [N, 3]
    - p_pred: 预测压力 [N, 1]
    - pos: 物理坐标 [N, 3] (必须 requires_grad=True)
    - mask_inner: 内部点掩码 [N]
    - sample_size: 每次计算物理 Loss 采样的点数 (防止显存爆炸)
    """
    
    # 1. 为了节省显存，我们不计算所有 120万个点的物理 Loss
    # 而是随机采样一部分“内部点”来计算
    inner_indices = torch.nonzero(mask_inner).squeeze() # 找出所有内部点的索引
    
    # 随机选择 sample_size 个点
    if len(inner_indices) > sample_size:
        perm = torch.randperm(len(inner_indices))
        idx = inner_indices[perm[:sample_size]]
    else:
        idx = inner_indices
        
    # 提取这些采样点的预测值和坐标
    u_sample = u_pred[idx] # [sample_size, 3]
    p_sample = p_pred[idx] # [sample_size, 1]
    pos_sample = pos[idx]  # [sample_size, 3]
    
    # 分解速度分量
    u = u_sample[:, 0:1]
    v = u_sample[:, 1:2]
    w = u_sample[:, 2:3]
    p = p_sample
    
    # --- 2. 自动微分 (Autograd) 计算一阶导数 ---
    #我们需要计算 du/dx, du/dy, du/dz 等
    
    # 通用求导函数
    def get_grad(y, x):
        # create_graph=True 是为了能算二阶导 (黏性项需要二阶导)
        grad = torch.autograd.grad(
            y, x, 
            grad_outputs=torch.ones_like(y), 
            create_graph=True, 
            retain_graph=True
        )[0]
        return grad

    # 对 u 求导 -> [du/dx, du/dy, du/dz]
    grad_u = get_grad(u, pos_sample)
    u_x, u_y, u_z = grad_u[:, 0:1], grad_u[:, 1:2], grad_u[:, 2:3]
    
    # 对 v 求导
    grad_v = get_grad(v, pos_sample)
    v_x, v_y, v_z = grad_v[:, 0:1], grad_v[:, 1:2], grad_v[:, 2:3]

    # 对 w 求导
    grad_w = get_grad(w, pos_sample)
    w_x, w_y, w_z = grad_w[:, 0:1], grad_w[:, 1:2], grad_w[:, 2:3]

    # 对 p 求导
    grad_p = get_grad(p, pos_sample)
    p_x, p_y, p_z = grad_p[:, 0:1], grad_p[:, 1:2], grad_p[:, 2:3]
    
    # --- 3. 计算二阶导数 (Laplacian) ---
    # nu * (u_xx + u_yy + u_zz)
    # 这是一个简化版，为了显存考虑，这里只演示思路
    # 实际上你需要再对 u_x 求导得到 u_xx...
    
    u_xx = get_grad(u_x, pos_sample)[:, 0:1]
    u_yy = get_grad(u_y, pos_sample)[:, 1:2]
    u_zz = get_grad(u_z, pos_sample)[:, 2:3]
    laplace_u = u_xx + u_yy + u_zz

    v_xx = get_grad(v_x, pos_sample)[:, 0:1]
    v_yy = get_grad(v_y, pos_sample)[:, 1:2]
    v_zz = get_grad(v_z, pos_sample)[:, 2:3]
    laplace_v = v_xx + v_yy + v_zz
    
    w_xx = get_grad(w_x, pos_sample)[:, 0:1]
    w_yy = get_grad(w_y, pos_sample)[:, 1:2]
    w_zz = get_grad(w_z, pos_sample)[:, 2:3]
    laplace_w = w_xx + w_yy + w_zz

    # --- 4. 组装 Navier-Stokes 残差 ---
    # 动量方程 (Momentum): rho * (u.grad)u = -grad(p) + mu * laplace(u)
    # 我们假设是稳态 (Steady State)，没有 du/dt 项
    
    # X 方向动量残差
    res_u = rho*(u*u_x + v*u_y + w*u_z) + p_x - mu*laplace_u
    # Y 方向动量残差
    res_v = rho*(u*v_x + v*v_y + w*v_z) + p_y - mu*laplace_v
    # Z 方向动量残差
    res_w = rho*(u*w_x + v*w_y + w*w_z) + p_z - mu*laplace_w

    # 连续性方程 (Continuity): div(u) = 0
    res_cont = u_x + v_y + w_z

    # --- 5. 汇总 Loss ---
    loss_momentum = torch.mean(res_u**2 + res_v**2 + res_w**2)
    loss_continuity = torch.mean(res_cont**2)
    
    return loss_momentum + loss_continuity