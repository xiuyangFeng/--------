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
import time

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            # dists = square_distance(xyz1, xyz2)
            dists = torch.cdist(xyz1, xyz2, p=2)  # (B,N,N)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points


def knn_torch(xyz, k):
    """
    xyz: (B, N, 3)
    return:
        idx: (B, N, k)
        dist: (B, N, k)
    """
    # pairwise squared distance
    with torch.no_grad():
        dist = torch.cdist(xyz, xyz, p=2)  # (B,N,N)
        # dist = square_distance(xyz, xyz)
        dist, idx = torch.topk(dist, k=k+1, largest=False)
        idx = idx[:, :, 1:]   # remove self
        dist = dist[:, :, 1:]
    return idx, dist


class LocalFeatureAggregation(nn.Module):
    def __init__(self, d_in, d_out, num_neighbors):
        super(LocalFeatureAggregation, self).__init__()

        self.num_neighbors = num_neighbors

        self.mlp = nn.Sequential(
            nn.Conv2d(d_in + 3, d_out, 1),
            nn.BatchNorm2d(d_out),
            nn.ReLU(),
            nn.Conv2d(d_out, 2*d_out, 1),
            nn.BatchNorm2d(2*d_out),
            nn.ReLU()
        )

    def forward(self, coords, features):
        """
        coords: [B, N, 3]
        features: [B, d_in, N, 1]
        """

        B, N, _ = coords.shape

        # KNN
        idx, _ = knn_torch(coords, self.num_neighbors)

        coords_expand = coords.unsqueeze(2).expand(-1, -1, self.num_neighbors, -1)

        neighbors = index_points(coords, idx)  # [B,N,K,3]

        relative_xyz = neighbors - coords_expand

        # feature gathering
        features = features.squeeze(-1).permute(0,2,1)  # [B,N,d]
        neighbor_features = index_points(features, idx)

        concat = torch.cat([relative_xyz, neighbor_features], dim=-1)

        concat = concat.permute(0,3,1,2)  # [B,C,N,K]

        new_features = self.mlp(concat)

        new_features = torch.max(new_features, dim=3)[0]  # max pooling

        return new_features.unsqueeze(-1)

def random_sample(xyz, npoint):
    """
    xyz: [B, 3, N]
    return: idx [B, npoint] (long)
    使用随机采样（与论文一致）
    """
    B, _, N = xyz.shape
    # 生成 [B, N] 的随机数，然后取每行 topk
    rand = torch.rand(B, N, device=xyz.device)
    _, idx = torch.topk(rand, k=npoint, dim=1)   # idx: [B, npoint]
    return idx

class RandLASA(nn.Module):
    def __init__(self, in_channel, out_channel, k, device, npoint):
        super().__init__()
        self.lfa = LocalFeatureAggregation(
            d_in=in_channel,
            d_out=out_channel,
            num_neighbors=k
        )
        # self.sample_ratio = sample_ratio  # 每层采样比例（或直接给固定 npoint）
        self.npoint = npoint
        
    def forward(self, xyz, points):
        """
        原来返回 (xyz, features) with same N。
        修改后返回下采样后的 (new_xyz, new_features)，
        以便下一层输入使用减少后的点数。
        输入:
            xyz: [B, 3, N]
            points: [B, D, N] or None
        返回:
            new_xyz: [B, 3, S]
            new_points: [B, 2*out_channel, S]
        """
        B, _, N = xyz.shape

        # ---------- 1) 决定采样点数 S ----------
        # S = max(1, int(N * self.sample_ratio))  
        idx = random_sample(xyz, self.npoint)  # [B, S]

        xyz_NC = xyz.transpose(1,2)  # [B, N, 3]
        new_xyz = index_points(xyz_NC, idx)  # [B, S, 3]

        if points is None:
            new_points_feat = new_xyz  
            new_points_for_lfa = new_points_feat.transpose(1,2).unsqueeze(-1)  # [B, 3, S, 1]
        else:
            points_ND = points.transpose(1,2)  # [B, N, D]
            new_points_feat = index_points(points_ND, idx)  # [B, S, D]
            new_points_for_lfa = new_points_feat.transpose(1,2).unsqueeze(-1)  # [B, D, S, 1]

        coords = new_xyz  # [B, S, 3]

        # ---------- 2) 应用 Local Feature Aggregation ----------
        lfa_out = self.lfa(coords, new_points_for_lfa)  # 返回 shape (B, 2*out_channel, S, 1)
        lfa_out = lfa_out.squeeze(-1)  # -> (B, 2*out_channel, S)

        # 返回下采样后的 xyz（恢复到 [B,3,S]）和 feature
        return new_xyz.transpose(1,2), lfa_out  # new_xyz.transpose -> [B,3,S]
        
class SimplePointTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, dim_ff=512, dropout=0.1):
        super().__init__()
        self.pos = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )

        enc = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,          # <<< important
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=1)

    def forward(self, feats_bcn, xyz_b3n):
        x = feats_bcn.permute(0, 2, 1)  # (B,N,C)
        xyz = xyz_b3n.permute(0, 2, 1)  # (B,N,3)
        x = x + self.pos(xyz)
        x = self.encoder(x)
        return x.permute(0, 2, 1)

        
class get_model(nn.Module):
    def __init__(self,device):
        super(get_model, self).__init__()
        
        # ---- Transformer at l2 level (A30 safe) ----
        self.trans_l3vel = SimplePointTransformer(
            d_model=512,    # l3_points has 2*out_channel = 512
            nhead=8,
            dim_ff=1024,
            dropout=0.1
        )
        
        self.trans_l3p = SimplePointTransformer(
            d_model=512,    # l3_points has 2*out_channel = 512
            nhead=8,
            dim_ff=1024,
            dropout=0.1
        )
        
        self.sa1 = RandLASA(
            in_channel=3,
            out_channel=64,
            k=128,
            device=device, npoint=4096
        )

        self.sa2 = RandLASA(
            in_channel=128,
            out_channel=128,
            k=128,
            device=device, npoint=2048
        )

        self.sa3 = RandLASA(
            in_channel=256,
            out_channel=256,
            k=128,
            device=device, npoint=512
        )

        # self.sa4 = PointNetFeaturePropagation(in_channel=512 + 3, mlp=[1024, 1024])
        
        self.fp3 = PointNetFeaturePropagation(
            in_channel=512 + 256,   # l3_points + l2_points
            mlp=[512, 512]
        )

        self.fp2 = PointNetFeaturePropagation(
            in_channel=512 + 128,   # fp3 + l1_points
            mlp=[256, 256]
        )

        self.fp1 = PointNetFeaturePropagation(
            in_channel=256 + 3,     # fp2 + xyz (or early feat)
            mlp=[256, 128]
        )

        
        self.fc1 = nn.Conv1d(128, 128, 1)
        self.fc2 = nn.Conv1d(128, 64, 1)
        self.fc3 = nn.Conv1d(64, 3, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.bn2 = nn.BatchNorm1d(64)
        self.drop2 = nn.Dropout(0.5)
        
        self.fpp1 = PointNetFeaturePropagation(
            in_channel=512 + 3,   # l3_points + xyz
            mlp=[512, 512]
        )
        
        self.fcp1 = nn.Conv1d(512, 128, 1)
        self.fcp2 = nn.Conv1d(128, 64, 1)
        self.fcp3 = nn.Conv1d(64, 1, 1)
        self.bnp1 = nn.BatchNorm1d(128)
        self.dropp1 = nn.Dropout(0.5)
        self.bnp2 = nn.BatchNorm1d(64)
        self.dropp2 = nn.Dropout(0.5)

    def forward(self, x):  # x: [B, 3, N] = [x,y,z]
        B, _, N = x.shape      
        xyz = x

        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)       
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        l3_pointsvel = self.trans_l3vel(l3_points, l3_xyz)
        l3_pointsp = self.trans_l3p(l3_points, l3_xyz)
        
        # velocity prediction
        # l3 → l2
        l2_fp = self.fp3(
            xyz1=l2_xyz,       # target
            xyz2=l3_xyz,       # source
            points1=l2_points,
            points2=l3_pointsvel
        )

        # l2 → l1
        l1_fp = self.fp2(
            xyz1=l1_xyz,
            xyz2=l2_xyz,
            points1=l1_points,
            points2=l2_fp
        )

        # l1 → original xyz
        l0_fp = self.fp1(
            xyz1=xyz,
            xyz2=l1_xyz,
            points1=xyz,       # 或 early feature
            points2=l1_fp
        )


        xvel = self.drop1(F.relu(self.bn1(self.fc1(l0_fp))))
        xvel = self.drop2(F.relu(self.bn2(self.fc2(xvel))))
        xvel = self.fc3(xvel)     # (B, 3, N)
        
        # pressure prediction
        # l3 → original xyz
        l0_fpp = self.fpp1(
            xyz1=xyz,       # target
            xyz2=l3_xyz,       # source
            points1=xyz,
            points2=l3_pointsp
        )


        xp = self.dropp1(F.relu(self.bnp1(self.fcp1(l0_fpp))))
        xp = self.dropp2(F.relu(self.bnp2(self.fcp2(xp))))
        xp = self.fcp3(xp)     # (B, 1, N)
        
        return torch.cat([xvel,xp],dim=1) # (B, 4, N)



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
def train_step(model, optimizer, pv_data,_, labels, scaler, device):
    model.train()
    optimizer.zero_grad()

    pv_data = pv_data.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    
    # Enable Autocast for Mixed Precision
    # with torch.cuda.amp.autocast():
    pred = model(pv_data)
    loss_data, mae = model.loss_func(pred, labels)
    
    scaler.scale(loss_data).backward()
    scaler.step(optimizer)
    scaler.update()

    return loss_data.item(), mae.item()
    
# ============================================================
#  3. Main Training Loop (batch-aware, fast)
# ============================================================
def train(model, dl_train, epochs, lr, device, loss_file): 
    dfhistory = pd.DataFrame(columns = ["epoch","loss",'mae']) 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( optimizer, mode='min', factor=0.98, patience=10, min_lr=1e-4 ) 
    scaler = GradScaler() 
    best_loss = 1e10 
    choice_num = 10000 
    print(f"Start Training at {datetime.datetime.now()}") 
    for epoch in range(1, epochs + 1): 
        loss_sum, mae_sum = 0.0, 0.0 
        for fn, pv_list, label_list in dl_train: 
            B = len(pv_list) 
            pv_batch = [] 
            label_batch = [] 
            for i in range(B): 
                pv_i = torch.as_tensor(pv_list[i], dtype=torch.float32) # (3, Ni) 
                lab_i = torch.as_tensor(label_list[i], dtype=torch.float32) # (4, Ni)
                Ni = pv_i.shape[1] 
                choice = min(choice_num, Ni) 
                idx = torch.randperm(Ni)[:choice] 
                pv_batch.append(pv_i[:, idx]) 
                label_batch.append(lab_i[:, idx]) 
                
            pv_batch = torch.stack(pv_batch, dim=0) # (B, 3, choice) 
            label_batch = torch.stack(label_batch, dim=0) # (B, 4, choice) 
            loss, mae = train_step( model, optimizer, pv_batch, pv_batch, label_batch, scaler, device ) 
            loss_sum += loss 
            mae_sum += mae 
        
        epoch_loss = loss_sum / len(dl_train) 
        epoch_mae = mae_sum/len(dl_train) 
        info = (epoch, epoch_loss, epoch_mae) 
        dfhistory.loc[epoch-1] = info
        
        scheduler.step(epoch_loss) 
        
        if epoch % 10 == 0: 
            dfhistory.to_csv(loss_file,index=False) 
            print(f"Epoch {epoch:04d} | loss={epoch_loss:.5f} | mae={epoch_mae:.5f}") # 
            
            if epoch_loss < best_loss: 
                best_loss = epoch_loss 
                torch.save(model.state_dict(), f"weight/best_epoch.pth") 


# ============================================================ # 4. Run Training # ============================================================ 

def my_collate(batch): 
    fns = [item[0] for item in batch] 
    pvs = [torch.tensor(item[1]) for item in batch] 
    labels = [torch.tensor(item[2]) for item in batch] 
    return fns, pvs, labels 
    
if __name__ == "__main__": 
    device = torch.device('cuda:0') 
    data_train = pickle.load(open('../../DATA_Nagahama/CHI_voxelized_train.pkl','rb')) 
    
# ======================================================= # NEW: Normalize Pressure Across All Samples # ======================================================= 
    p_all = [] # Collect all p values 
    for fn, pv, label in data_train: 
        p_all.append(label[3]) # (Ni,) 
    
    p_all = np.concatenate(p_all) 
    p_min = float(p_all.min()) 
    p_max = float(p_all.max()) 
    print(">>> Global Pressure Range:", p_min, p_max) # Save p_min and p_max to CSV 
    df_norm = pd.DataFrame({"p_min": [p_min], "p_max": [p_max]}) 
    os.makedirs("weight", exist_ok=True) 
    df_norm.to_csv("weight/norm_pressure.csv", index=False) 
    print(">>> Saved normalization factors to weight/norm_pressure.csv") # Normalize each sample 
    
    for i in range(len(data_train)): 
        fn, pv, label = data_train[i] 
        p = label[3]
        p_norm = (p - p_min) / (p_max - p_min + 1e-9) 
        label[3] = p_norm 
        data_train[i] = (fn, pv*1000, label) 
        
# ======================================================= 

    dl_train = DataLoader( data_train, 
                            batch_size=6, 
                            shuffle=True, 
                            num_workers=8, 
                            pin_memory=True, 
                            persistent_workers=True, 
                            prefetch_factor=4, collate_fn=my_collate ) 
                            
    model = get_model(device).to(device) 
    # model.load_state_dict(torch.load("weight/best_epoch30.pth", map_location=device)) 
    
    model.loss_func = get_loss() 
    train(model, dl_train, epochs=150, lr=3e-3, device=device, loss_file="train_loss1.csv")
    train(model, dl_train, epochs=50, lr=1e-4, device=device, loss_file="train_loss2.csv")
