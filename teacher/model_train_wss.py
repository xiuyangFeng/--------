import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import datetime
import pandas as pd
import pickle
import torch.nn.functional as F
import os
import json
import math


# ============================================================
#  E2-GLOBAL locked configuration
#  Source: training_wss_min/configs/pointnet_distribution_matrix/e2_global.json
# ============================================================
SEED = 1234
N_POINTS = 2000
BATCH_SIZE = 8
NUM_WORKERS = 4
EPOCHS = 400
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 10
MIN_LR = 1e-5
GRAD_CLIP = 1.0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "outputs", "wss_pkl")
WEIGHT_DIR = os.path.join(SCRIPT_DIR, "weight")
HISTORY_PATH = os.path.join(SCRIPT_DIR, "train_loss.csv")


class get_model(nn.Module):
    def __init__(self):
        super(get_model, self).__init__()

        # Initial MLP1: [6 → 256 → 512]
        self.mlp_convs1 = nn.ModuleList()
        self.mlp_bns1   = nn.ModuleList()
        input_channel = 6
        for output_channel in [256, 512]:
            self.mlp_convs1.append(nn.Conv1d(input_channel, output_channel, 1))
            self.mlp_bns1.append(nn.BatchNorm1d(output_channel))
            input_channel = output_channel

        # Final MLP2: [1024 → 512 → 256 → 1]
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns   = nn.ModuleList()
        last_channel = 1024
        for out_channel in [512, 256, 1]:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            if out_channel != 1:
                self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

        # dropout=0.0
        self.dropout = nn.Identity()

    def forward(self, xyz):
        B, _, N = xyz.shape
        x = xyz[:, 0:6, :]   # xyz + geom

        # MLP 1
        for i, conv in enumerate(self.mlp_convs1):
            bn = self.mlp_bns1[i]
            x = F.relu(bn(conv(x)))
        # Global feature
        x_global = torch.max(x, 2, keepdim=True)[0]  # (B, 512, 1)
        x_global = x_global.expand(-1, -1, N)        # (B, 512, N)

        x = torch.cat([x_global, x], dim=1)          # (B, 1024, N)

        # MLP 2
        for i, conv in enumerate(self.mlp_convs):
            if i < 2:
                bn = self.mlp_bns[i]
                x = F.relu(bn(conv(x)))
            else:
                x = conv(self.dropout(x))

        return x   # (B, 1, N)


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        e = pred - target
        mse = torch.mean(e**2)
        mae = torch.mean(torch.abs(e))
        return mse, mae


# ============================================================
#  2. Train Step (GPU-optimized)
# ============================================================
def train_step(model, optimizer, pv_data, labels, scaler, device):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    pv_data = pv_data.to(device, non_blocking=True)       # (B, 6, M)
    labels  = labels.to(device, non_blocking=True)        # (B, 1, M)

    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        pred = model(pv_data)                             # (B, 1, M)
        loss, mae = model.loss_func(pred, labels)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    scaler.step(optimizer)
    scaler.update()

    return loss.item(), mae.item()


def fps_indices(xyz, n_sample, seed):
    """在归一化 xyz 上执行与 baseline 相同的确定性 FPS。"""
    N = xyz.shape[0]
    n_sample = min(n_sample, N)
    if n_sample >= N:
        return np.arange(N, dtype=np.int64)

    rng = np.random.default_rng(seed)
    sel = np.empty(n_sample, dtype=np.int64)
    sel[0] = rng.integers(N)
    dist = np.linalg.norm(xyz - xyz[sel[0]], axis=1)
    for i in range(1, n_sample):
        sel[i] = int(np.argmax(dist))
        dist = np.minimum(dist, np.linalg.norm(xyz - xyz[sel[i]], axis=1))
    return sel


def build_fixed_fps_dataset(data_train):
    """每例只计算一次 FPS2000；后续 400 epochs 不重采样。"""
    sampled = []
    for case_index, (fn, feat, label) in enumerate(data_train):
        seed = SEED + 7919 * case_index
        idx = fps_indices(feat[:3].T, N_POINTS, seed)
        if len(idx) != N_POINTS:
            raise ValueError(f"{fn}: only {len(idx)} wall points, E2 requires {N_POINTS}")
        sampled.append((fn, feat[:, idx], label[:, idx]))
    return sampled


def lr_multiplier(epoch, epochs):
    """linear warmup + cosine decay。"""
    if epoch < WARMUP_EPOCHS:
        return (epoch + 1) / max(1, WARMUP_EPOCHS)
    progress = (epoch - WARMUP_EPOCHS) / max(1, epochs - WARMUP_EPOCHS)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return (MIN_LR + (LEARNING_RATE - MIN_LR) * cosine) / LEARNING_RATE


# ============================================================
#  3. Main Training Loop (batch-aware, fast)
# ============================================================
def train(model, dl_train, epochs, lr, device):
    dfhistory = pd.DataFrame(columns=["epoch", "loss", "mae"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: lr_multiplier(epoch, epochs)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    best_loss = 1e10

    print(f"Start Training at {datetime.datetime.now()}")

    for epoch in range(epochs):
        loss_sum, mae_sum = 0.0, 0.0

        for fn, pv_list, label_list in dl_train:
            pv_batch = torch.stack(pv_list, dim=0).float()       # (B, 6, 2000)
            label_batch = torch.stack(label_list, dim=0).float() # (B, 1, 2000)

            loss, mae = train_step(model, optimizer, pv_batch, label_batch, scaler, device)

            loss_sum += loss
            mae_sum += mae

        epoch_loss = loss_sum / len(dl_train)
        epoch_mae = mae_sum / len(dl_train)
        info = (epoch, epoch_loss, epoch_mae)
        dfhistory.loc[epoch] = info

        scheduler.step()

        if epoch % 10 == 0 or epoch == epochs - 1:
            dfhistory.to_csv(HISTORY_PATH, index=False)
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:04d}/{epochs} | loss={epoch_loss:.5f} | "
                f"mae={epoch_mae:.5f} | lr={current_lr:.2e}"
            )

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), os.path.join(WEIGHT_DIR, "best_epoch.pth"))

        torch.save(model.state_dict(), os.path.join(WEIGHT_DIR, "last_epoch.pth"))

    return dfhistory


# ============================================================
#  4. Run Training
# ============================================================
def my_collate(batch):
    fns = [item[0] for item in batch]
    pvs = [torch.tensor(item[1]) for item in batch]
    labels = [torch.tensor(item[2]) for item in batch]
    return fns, pvs, labels


def seed_worker(worker_id):
    """与 baseline 一致地为每个 DataLoader worker 派生固定 seed。"""
    worker_seed = SEED + 10007 * (worker_id + 1)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


if __name__ == "__main__":
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_train = pickle.load(open(os.path.join(DATA_DIR, "wss_train.pkl"), "rb"))
    stats = json.load(open(os.path.join(DATA_DIR, "wss_norm_stats.json")))

    if len(data_train) != 61:
        raise ValueError(
            f"E2-GLOBAL requires 61 train cases, got {len(data_train)}. "
            "Please rebuild with build_wss_dataset.py."
        )

    # =======================================================
    #  Normalize WSS
    # =======================================================
    eps = float(stats["eps"])
    mu = float(stats["log"]["mean"])
    sigma = float(stats["log"]["std"])
    print(">>> WSS log_z stats:", eps, mu, sigma)

    os.makedirs(WEIGHT_DIR, exist_ok=True)
    pd.DataFrame({"eps": [eps], "log_mean": [mu], "log_std": [sigma]}).to_csv(
        os.path.join(WEIGHT_DIR, "norm_wss.csv"), index=False
    )

    # 几何通道：curvature 做 signed-log1p；三几何特征 train z-score
    abs_all, rad_all, curv_all = [], [], []
    for fn, feat, label in data_train:
        abs_all.append(feat[3])
        rad_all.append(feat[4])
        curv_all.append(feat[5])
    abs_all = np.concatenate(abs_all).astype(np.float64)
    rad_all = np.concatenate(rad_all).astype(np.float64)
    curv_all = np.concatenate(curv_all).astype(np.float64)
    curv_all = np.sign(curv_all) * np.log1p(np.abs(curv_all))
    curv_clip = float(np.percentile(np.abs(curv_all), 99))
    curv_all = np.clip(curv_all, -curv_clip, curv_clip)
    geom_mu = np.array([abs_all.mean(), rad_all.mean(), curv_all.mean()], np.float64)
    geom_std = np.array([abs_all.std(), rad_all.std(), curv_all.std()], np.float64) + 1e-6

    feature_stats = {
        "abscissa_norm": {
            "mean": float(abs_all.mean()), "std": float(abs_all.std() + 1e-6),
            "transform": "none",
        },
        "local_radius": {
            "mean": float(rad_all.mean()), "std": float(rad_all.std() + 1e-6),
            "transform": "none",
        },
        "curvature": {
            "mean": float(curv_all.mean()), "std": float(curv_all.std() + 1e-6),
            "clip": curv_clip, "transform": "signed_log1p",
        },
    }
    with open(os.path.join(WEIGHT_DIR, "feature_stats.json"), "w") as f:
        json.dump(feature_stats, f, indent=2)

    for i in range(len(data_train)):
        fn, feat, label = data_train[i]
        feat = feat.copy()
        # geom norm
        feat[3] = (feat[3] - geom_mu[0]) / geom_std[0]
        feat[4] = (feat[4] - geom_mu[1]) / geom_std[1]
        c_raw = feat[5].astype(np.float64)
        c = np.sign(c_raw) * np.log1p(np.abs(c_raw))
        c = np.clip(c, -curv_clip, curv_clip)
        feat[5] = (c - geom_mu[2]) / geom_std[2]
        # wss log_z
        wss = label[0]
        wss_n = (np.log(np.clip(wss, 0, None) + eps) - mu) / sigma
        data_train[i] = (fn, feat.astype(np.float32), wss_n.astype(np.float32)[None, :])

    # =======================================================

    print(">>> Precomputing fixed FPS2000 (one time only)...")
    data_train = build_fixed_fps_dataset(data_train)

    dl_train = DataLoader(
        data_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=2,
        collate_fn=my_collate,
        generator=torch.Generator().manual_seed(SEED),
        worker_init_fn=seed_worker,
    )

    model = get_model().to(device)
    model.loss_func = get_loss()

    train(model, dl_train, epochs=EPOCHS, lr=LEARNING_RATE, device=device)
