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
def train_step(model, optimizer, pv_data, pv_phy, labels, scaler, device):
    model.train()
    optimizer.zero_grad()

    pv_data = pv_data.to(device, non_blocking=True)       # (B, 3, M)
    labels  = labels.to(device, non_blocking=True)        # (B, 4, M)


    with autocast():
        pred = model(pv_data)                             # (B, 4, M)
        loss_data, mae = model.loss_func(pred, labels)

        loss = loss_data

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    return loss.item(), mae.item()


# ============================================================
#  3. Main Training Loop (batch-aware, fast)
# ============================================================
def train(model, dl_train, epochs, lr, device):
    dfhistory = pd.DataFrame(columns = ["epoch","loss",'mae'])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.98, patience=10, min_lr=1e-4
    )
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

                pv_i = torch.as_tensor(pv_list[i], dtype=torch.float32)     # (3, Ni)
                lab_i = torch.as_tensor(label_list[i], dtype=torch.float32) # (4, Ni)

                Ni = pv_i.shape[1]
                choice = min(choice_num, Ni)

                idx = torch.randperm(Ni)[:choice]
                pv_batch.append(pv_i[:, idx])
                label_batch.append(lab_i[:, idx])

            pv_batch = torch.stack(pv_batch, dim=0)       # (B, 3, choice)
            label_batch = torch.stack(label_batch, dim=0) # (B, 4, choice)

            loss, mae = train_step(
                model, optimizer,
                pv_batch, pv_batch, label_batch,   # pv_phy = pv_batch if same sampling
                scaler, device
            )

            loss_sum += loss
            mae_sum += mae



        epoch_loss = loss_sum / len(dl_train)
        epoch_mae = mae_sum/len(dl_train)
        info = (epoch, epoch_loss, epoch_mae)
        dfhistory.loc[epoch-1] = info

        scheduler.step(epoch_loss)

        if epoch % 10 == 0:

            dfhistory.to_csv("train_loss.csv",index=False)
            print(f"Epoch {epoch:04d} | loss={epoch_loss:.5f} | mae={epoch_mae:.5f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), "weight/best_epoch.pth")

    return history


# ============================================================
#  4. Run Training
# ============================================================
def my_collate(batch):
    fns = [item[0] for item in batch]
    pvs = [torch.tensor(item[1]) for item in batch]
    labels = [torch.tensor(item[2]) for item in batch]
    return fns, pvs, labels


if __name__ == "__main__":
    device = torch.device('cuda:0')

    data_train = pickle.load(open('../DATA_Nagahama/CHI_voxelized_train.pkl','rb'))

    dl_train = DataLoader(
                            data_train,
                            batch_size=16,
                            shuffle=True,
                            num_workers=8,
                            pin_memory=True,
                            persistent_workers=True,
                            prefetch_factor=4,
                            collate_fn=my_collate)


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
        data_train[i] = (fn, pv, label)

    # =======================================================

    dl_train = DataLoader(
        data_train,
        batch_size=16,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        collate_fn=my_collate
    )

    model = get_model().to(device)
    model.loss_func = get_loss()

    train(model, dl_train, epochs=20000, lr=3e-3, device=device)
