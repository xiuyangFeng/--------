import torch
from torch import nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
import os
import datetime
import random
import pointnet2_ssg as pointnet2

# ----------------------------------
#  Reproducibility
# ----------------------------------
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


if __name__ == "__main__":

    # ----------------------------------
    # Load model
    # ----------------------------------
    model = pointnet2.get_model()
    device = torch.device("cuda:0")
    model.load_state_dict(torch.load("weight/best_epoch.pth", map_location=device))
    model.to(device)
    model.eval()

    # ----------------------------------
    # Load p_min and p_max for renormalization
    # ----------------------------------
    norm_file = "weight/norm_pressure.csv"
    df_norm = pd.read_csv(norm_file)
    p_min = df_norm.p_min.item()
    p_max = df_norm.p_max.item()

    print(f"Loaded normalization factors: p_min={p_min}, p_max={p_max}")

    # ----------------------------------
    # Load test dataset
    # ----------------------------------
    data_valid = pickle.load(open("../DATA_Nagahama/CHI_voxelized_test.pkl", "rb"))
    dl_valid = DataLoader(data_valid, batch_size=1, shuffle=False, num_workers=1)

    # ----------------------------------
    # Compute global min / max of uvwp in test set
    # ----------------------------------
    print("\nComputing global uvwp min/max from test set ...")

    u_min, v_min, w_min, p_min = np.inf, np.inf, np.inf, np.inf
    u_max, v_max, w_max, p_max = -np.inf, -np.inf, -np.inf, -np.inf

    for fn, pv, labels in data_valid:   # directly iterate dataset

        u = labels[0, :]
        v = labels[1, :]
        w = labels[2, :]
        p = labels[3, :]

        u_min = min(u_min, u.min())
        v_min = min(v_min, v.min())
        w_min = min(w_min, w.min())
        p_min = min(p_min, p.min())

        u_max = max(u_max, u.max())
        v_max = max(v_max, v.max())
        w_max = max(w_max, w.max())
        p_max = max(p_max, p.max())

    global_min = np.array([u_min, v_min, w_min, p_min])
    global_max = np.array([u_max, v_max, w_max, p_max])

    print("Global min (u,v,w,p):", global_min)
    print("Global max (u,v,w,p):", global_max)


    output_dir = "."
    os.makedirs(output_dir, exist_ok=True)

    # ----------------------------------
    # Store NMAE for all cases
    # ----------------------------------
    nmae_records = []

    # ----------------------------------
    # Inference Loop
    # ----------------------------------
    for i, (fn, pv, labels) in enumerate(dl_valid):

        print(f"\nProcessing {fn[0]} ...")
        print("Number of points =", pv.shape[-1])   # ★ show N in this test case

        pv = pv.float().to(device)           # (1, 3, N)
        labels = labels.float()              # (1, 4, N)

        # ---- prediction ----
        with torch.no_grad():
            pred = model(pv)                 # (1, 4, N)

        pred = pred.squeeze(0).cpu().numpy().T      # (N, 4)
        labels = labels.squeeze(0).numpy().T        # (N, 4)
        pv_np = pv.squeeze(0).cpu().numpy().T       # (N, 3)

        # ----------------------------------
        # Renormalize predicted and GT pressure
        # ----------------------------------
        p_pred_norm = pred[:, 3]

        p_pred_real = p_pred_norm * (p_max - p_min) + p_min

        # Replace normalized p with real p
        pred[:, 3] = p_pred_real

        # ----------------------------------
        # Compute NMAE for uvwp (per case)
        # ----------------------------------
        nmae_case = {"case": fn[0]}

        var_names = ["u", "v", "w", "p"]

        for j, var in enumerate(var_names):
            gt = labels[:, j]
            pd_ = pred[:, j]

            denom = global_max[j] - global_min[j]

            if denom == 0:
                nmae = np.nan
            else:
                nmae = np.mean(np.abs(pd_ - gt)) / denom

            nmae_case[f"NMAE_{var}"] = nmae

        nmae_records.append(nmae_case)

# ----------------------------------
# Save all NMAE results
# ----------------------------------
df_nmae = pd.DataFrame(nmae_records)
nmae_csv = os.path.join(output_dir, "NMAE_all_cases.csv")
df_nmae.to_csv(nmae_csv, index=False)

print(f"\nSaved NMAE summary → {nmae_csv}")
