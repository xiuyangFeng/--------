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
    data_valid = pickle.load(open("./CHI_voxelized_test.pkl", "rb"))
    dl_valid = DataLoader(data_valid, batch_size=1, shuffle=False, num_workers=1)

    output_dir = "test"
    os.makedirs(output_dir, exist_ok=True)

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
        # Save result CSV
        # ----------------------------------
        result = np.concatenate([pv_np, pred, labels], axis=1)

        df = pd.DataFrame(
            result,
            columns=[
                "x", "y", "z",
                "u_pred", "v_pred", "w_pred", "p_pred",
                "u_gt", "v_gt", "w_gt", "p_gt"
            ]
        )

        output_file = os.path.join(output_dir, f"result_test_{fn[0]}.csv")
        df.to_csv(output_file, index=False)

        print(f"Saved → {output_file}")
