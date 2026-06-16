import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import datetime
import pandas as pd
import pickle
import torch.nn.functional as F
import os

def fun_u_0(u_x,v_y,w_z):
        return u_x+v_y+w_z

    # Define boundary condition   边界
def fun_u_b(u,v,w):
        return torch.abs(u)+torch.abs(v)+torch.abs(w)

    # Define residual of the PDE 残差
def fun_r(u,v,w, u_t, u_x, u_y,u_z,f_x,u_xx,u_yy,u_zz):
        return u_t+u*u_x+v*u_y+w*u_z+f_x-(u_xx+u_yy+u_zz)/Re

def get_pinn(model, pv):

    # pv: (B, 3, N)
    B, _, N = pv.shape

    x = pv[:, 0:1, :]   # (B,1,N)
    y = pv[:, 1:2, :]
    z = pv[:, 2:3, :]

    x.requires_grad = True
    y.requires_grad = True
    z.requires_grad = True

    # Run model
    logits = model(torch.cat([x, y, z], dim=1))   # (B,4,N)

    u = logits[:, 0:1, :]     # (B,1,N)
    v = logits[:, 1:2, :]
    w = logits[:, 2:3, :]
    f = logits[:, 3:4, :]

    # time derivatives = 0
    u_t = torch.zeros_like(u)
    v_t = torch.zeros_like(v)
    w_t = torch.zeros_like(w)

    ones = torch.ones_like(x)

    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(x), retain_graph=True, create_graph=True)[0] # allow 2-order derivatives later
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
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(x), retain_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(y), retain_graph=True)[0]
    u_zz = torch.autograd.grad(u_z, z, grad_outputs=torch.ones_like(z), retain_graph=True)[0]
    w_xx = torch.autograd.grad(w_x, x, grad_outputs=torch.ones_like(x), retain_graph=True)[0]
    w_yy = torch.autograd.grad(w_y, y, grad_outputs=torch.ones_like(y), retain_graph=True)[0]
    w_zz = torch.autograd.grad(w_z, z, grad_outputs=torch.ones_like(z), retain_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(x), retain_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(y), retain_graph=True)[0]
    v_zz = torch.autograd.grad(v_z, z, grad_outputs=torch.ones_like(z), retain_graph=True)[0]

    # u_x = torch.autograd.grad(u.sum(), x, retain_graph=True, create_graph=True)[0] # allow 2-order derivatives later
    # u_y = torch.autograd.grad(u.sum(), y, retain_graph=True, create_graph=True)[0]
    # u_z = torch.autograd.grad(u.sum(), z, retain_graph=True, create_graph=True)[0]
    # w_x = torch.autograd.grad(w.sum(), x, retain_graph=True, create_graph=True)[0]
    # w_y = torch.autograd.grad(w.sum(), y, retain_graph=True, create_graph=True)[0]
    # w_z = torch.autograd.grad(w.sum(), z, retain_graph=True, create_graph=True)[0]
    # v_x = torch.autograd.grad(v.sum(), x, retain_graph=True, create_graph=True)[0]
    # v_y = torch.autograd.grad(v.sum(), y, retain_graph=True, create_graph=True)[0]
    # v_z = torch.autograd.grad(v.sum(), z, retain_graph=True, create_graph=True)[0]
    # f_x = torch.autograd.grad(f.sum(), x, retain_graph=True)[0]
    # f_y = torch.autograd.grad(f.sum(), y, retain_graph=True)[0]
    # f_z = torch.autograd.grad(f.sum(), z, retain_graph=True)[0]
    # u_xx = torch.autograd.grad(u_x.sum(), x, retain_graph=True)[0]
    # u_yy = torch.autograd.grad(u_y.sum(), y, retain_graph=True)[0]
    # u_zz = torch.autograd.grad(u_z.sum(), z, retain_graph=True)[0]
    # w_xx = torch.autograd.grad(w_x.sum(), x, retain_graph=True)[0]
    # w_yy = torch.autograd.grad(w_y.sum(), y, retain_graph=True)[0]
    # w_zz = torch.autograd.grad(w_z.sum(), z, retain_graph=True)[0]
    # v_xx = torch.autograd.grad(v_x.sum(), x, retain_graph=True)[0]
    # v_yy = torch.autograd.grad(v_y.sum(), y, retain_graph=True)[0]
    # v_zz = torch.autograd.grad(v_z.sum(), z, retain_graph=True)[0]

    # Physics residuals
    r1 = fun_r(u, v, w, u_t, u_x, u_y, u_z, f_x, u_xx, u_yy, u_zz)
    r2 = fun_r(u, v, w, v_t, v_x, v_y, v_z, f_y, v_xx, v_yy, v_zz)
    r3 = fun_r(u, v, w, w_t, w_x, w_y, w_z, f_z, w_xx, w_yy, w_zz)
    r4 = fun_u_0(u_x, v_y, w_z)

    return (r1**2).mean() + (r2**2).mean() + (r3**2).mean() + (r4**2).mean()


class get_model(nn.Module):
    def __init__(self):
        super(get_model, self).__init__()

        # Initial MLP1: [3 → 256 → 512]
        self.mlp_convs1 = nn.ModuleList()
        self.mlp_bns1   = nn.ModuleList()
        input_channel = 3
        for output_channel in [256, 512]:
            self.mlp_convs1.append(nn.Conv1d(input_channel, output_channel, 1))
            self.mlp_bns1.append(nn.BatchNorm1d(output_channel))
            input_channel = output_channel

        # Final MLP2: [1024 → 512 → 256 → 4]
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns   = nn.ModuleList()
        last_channel = 1024
        for out_channel in [512, 256, 4]:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            if out_channel != 4:
                self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

        self.dropout = nn.Dropout(0.5)


    def forward(self, xyz):
        B, _, N = xyz.shape
        x = xyz[:, 0:3, :]   # only xyz

        # MLP 1
        for i, conv in enumerate(self.mlp_convs1):
            bn = self.mlp_bns1[i]
            x = F.relu(bn(conv(x)))

        # Global feature
        y = x                           # (B, 512, N)
        x_global = torch.max(x, 2, keepdim=True)[0]  # (B, 512, 1)
        x_global = x_global.expand(-1, -1, N)        # (B, 512, N)
        x = torch.cat([x_global, x], dim=1)          # (B, 1024, N)

        # MLP 2
        for i, conv in enumerate(self.mlp_convs):
            if i < 2:
                bn = self.mlp_bns[i]
                x = F.relu(bn(conv(x)))
            else:
                x = conv(x)

        return x   # (B, 4, N)

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        e = pred - target
        mse = torch.mean(e**2)
        mae= torch.mean(torch.abs(e))
        return mse,mae

# ============================================================
#  2. Train Step (GPU-optimized)
# ============================================================
def train_step(model, optimizer, pv_data, pv_phy, labels, lphy, scaler, device):
    model.train()
    optimizer.zero_grad()

    pv_data = pv_data.to(device, non_blocking=True)       # (B, 3, M)
    pv_phy = pv_phy.to(device, non_blocking=True)       # (B, 3, M)
    labels  = labels.to(device, non_blocking=True)        # (B, 4, M)



    pred = model(pv_data)                             # (B, 4, M)
    loss_data, mae = model.loss_func(pred, labels)

    # Create wall mask
    wall_mask = (
        labels[:, 0, :]**2 + labels[:, 1, :]**2 + labels[:, 2, :]**2
    ) <= 0.01    # (B, N)

    B = pv_data.size(0)
    walls = []

    # select points batch by batch
    for b in range(B):
        mask = wall_mask[b]             # (N,)
        pv_b = pv_data[b][:, mask]      # (3, M_b)
        walls.append(pv_b)

    # Compute wall loss
    loss_wall = 0.0
    for w in walls:
        pred_w = model(w.unsqueeze(0))  # (1,4,M_b)
        u_w = pred_w[:,0]
        v_w = pred_w[:,1]
        w_w = pred_w[:,2]

        loss_wall += torch.mean(u_w**2 + v_w**2 + w_w**2)

    loss_wall = loss_wall / B



    los = get_pinn(model, pv_phy)
    func = los + loss_wall

    loss = loss_data + lphy * func

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    return loss.item(), loss_data.item(), func.item()


# ============================================================
#  3. Main Training Loop (batch-aware, fast)
# ============================================================
def train(model, dl_train, epochs, lr, device):
    dfhistory = pd.DataFrame(columns = ["epoch","loss",'loss_data','func'])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.98, patience=10, min_lr=1e-4
    )
    scaler = GradScaler()
    best_loss = 1e10
    choice_num = 10000
    lphy = 1

    print(f"Start Training at {datetime.datetime.now()}")

    for epoch in range(1, epochs + 1):
        loss_sum, data_sum, func_sum = 0.0, 0.0, 0.0

        for fn, pv_list, label_list in dl_train:

            B = len(pv_list)
            pv_batch = []
            pv_phy_batch = []
            label_batch = []

            for i in range(B):

                pv_i = torch.as_tensor(pv_list[i], dtype=torch.float32)     # (3, Ni)
                lab_i = torch.as_tensor(label_list[i], dtype=torch.float32) # (4, Ni)

                Ni = pv_i.shape[1]
                choice = min(choice_num, Ni)

                idx = torch.randperm(Ni)[:choice]
                pv_batch.append(pv_i[:, idx])
                label_batch.append(lab_i[:, idx])

                idx = torch.randperm(Ni)[:choice]
                pv_phy_batch.append(pv_i[:, idx])

            pv_batch = torch.stack(pv_batch, dim=0)       # (B, 3, choice)
            label_batch = torch.stack(label_batch, dim=0) # (B, 4, choice)
            pv_phy_batch = torch.stack(pv_phy_batch, dim=0)

            loss, loss_data, func = train_step(
                model, optimizer,
                pv_batch, pv_phy_batch, label_batch, lphy,   # pv_phy = pv_batch if same sampling
                scaler, device
            )

            loss_sum += loss
            data_sum += loss_data
            func_sum += func


        epoch_loss = loss_sum / len(dl_train)
        epoch_data = data_sum/len(dl_train)
        epoch_func = func_sum/len(dl_train)

        info = (epoch, epoch_loss, epoch_data, epoch_func)
        dfhistory.loc[epoch-1] = info

        scheduler.step(epoch_loss)

        if epoch % 10 == 0:

            lphy = epoch_data/epoch_func
            # lphy = max( min(epoch_data/(epoch_func+1e-9), 100.0), 0.001 )


            dfhistory.to_csv("train_loss.csv",index=False)
            print(f"Epoch {epoch:04d} | loss={epoch_loss:.5f} | data={epoch_data:.5f} | func={epoch_func:.5f} | lphy={lphy:.5f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), "weight/best_epoch.pth")

    return dhistory


# ============================================================
#  4. Run Training
# ============================================================
def my_collate(batch):
    fns = [item[0] for item in batch]
    pvs = [torch.tensor(item[1]) for item in batch]
    labels = [torch.tensor(item[2]) for item in batch]
    return fns, pvs, labels

Re=300

if __name__ == "__main__":
    device = torch.device('cuda:0')

    data_train = pickle.load(open('../DATA_Nagahama/CHI_voxelized_train.pkl','rb'))

        # =======================================================
    #  NEW: Normalize Pressure Across All Samples
    # =======================================================
    p_all = []

    # Collect all p values
    for fn, pv, label in data_train:
        p_all.append(label[3])  # (Ni,)

    p_all = np.concatenate(p_all)
    p_min = float(p_all.min())
    p_max = float(p_all.max())

    print(">>> Global Pressure Range:", p_min, p_max)

    # Save p_min and p_max to CSV
    df_norm = pd.DataFrame({"p_min": [p_min], "p_max": [p_max]})
    os.makedirs("weight", exist_ok=True)
    df_norm.to_csv("weight/norm_pressure.csv", index=False)
    print(">>> Saved normalization factors to weight/norm_pressure.csv")

    # Normalize each sample
    for i in range(len(data_train)):
        fn, pv, label = data_train[i]
        p = label[3]
        p_norm = (p - p_min) / (p_max - p_min + 1e-9)
        label[3] = p_norm
        data_train[i] = (fn, pv*1000, label)

    # =======================================================

    dl_train = DataLoader(
        data_train,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        collate_fn=my_collate
    )

    model = get_model().to(device)
    model.loss_func = get_loss()

    train(model, dl_train, epochs=20000, lr=3e-3, device=device)
