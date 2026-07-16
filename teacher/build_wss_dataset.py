import os
import json
import pickle
import numpy as np
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_ROOT = os.path.join(ROOT, "data_wss_min", "AG")
SPLIT_JSON = os.path.join(
    ROOT, "training", "splits", "split_AG_wss_min_v1_traintest.json"
)
OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "wss_pkl")


def read_data(bundle_path):
    """从 bundle.npz 读取壁面坐标、几何特征与峰值 WSS。"""
    with np.load(bundle_path, allow_pickle=True) as data:
        steps = data["steps"].tolist()
        peak = int(data["peak_step"])
        peak_idx = steps.index(peak)

        xyz = data["wall_coords_norm"].astype(np.float32)          # (N, 3)
        abscissa = data["wall_abscissa_norm"].astype(np.float32)    # (N,)
        radius = data["wall_local_radius"].astype(np.float32)       # (N,)
        curv = data["wall_curvature"].astype(np.float32)            # (N,)
        wss = data["wall_wss"][peak_idx].astype(np.float32)         # (N,)

    features = np.stack(
        [xyz[:, 0], xyz[:, 1], xyz[:, 2], abscissa, radius, curv],
        axis=0,
    ).astype(np.float32)                                       # (6, N)
    values = wss[None, :]                                      # (1, N)
    return features, values, peak


def process_case(bundle_path, case_id):
    features, values, peak = read_data(bundle_path)
    print(f"Processed: {case_id}  N={features.shape[1]}  peak={peak}")
    return case_id, features, values


def process_directory(data_root, split_json):
    """按 AG 正式 split 白名单读取各病例 bundle.npz。"""
    split = json.load(open(split_json))
    counts = (
        len(split["train_cases"]), len(split["val_cases"]), len(split["test_cases"])
    )
    if counts != (61, 0, 16):
        raise ValueError(f"E2-GLOBAL requires split counts (61, 0, 16), got {counts}")
    labels = split["train_cases"] + split["val_cases"] + split["test_cases"]
    print(len(labels))

    dataset = []
    meta = []
    # split JSON 本身已有固定顺序；保留该顺序，确保训练侧每例 FPS seed
    # 与 E2-GLOBAL baseline（seed + 7919 * case_index）完全对应。
    for label in tqdm(labels, desc="Reading bundles"):
        subset, case = label.split("/", 1)
        bundle_path = os.path.join(data_root, subset, case, "bundle.npz")
        if not os.path.isfile(bundle_path):
            raise FileNotFoundError(bundle_path)
        case_id, feat, wss = process_case(bundle_path, label)
        dataset.append((case_id, feat, wss))
        part = (
            "train" if label in split["train_cases"] else
            "val" if label in split["val_cases"] else "test"
        )
        meta.append({"label": label, "part": part})
    return dataset, meta, split


def split_and_save_dataset(
    dataset, meta, split, output_train, output_val, output_test, stats_json
):
    """按 E2-GLOBAL train/val/test 写出 pkl，并计算 train61-only log_z 统计。"""
    buckets = {"train": [], "val": [], "test": []}
    for item, m in zip(dataset, meta):
        buckets[m["part"]].append(item)

    for path, key in ((output_train, "train"), (output_val, "val"), (output_test, "test")):
        with open(path, "wb") as f:
            pickle.dump(buckets[key], f)
        print(f"Saved: {path} ({len(buckets[key])} cases)")

    # baseline 的统计在 float64 中计算；显式转换可避免 float32 reduction 带来的约 1e-7 偏差。
    wss_all = np.concatenate(
        [item[2].reshape(-1) for item in buckets["train"]]
    ).astype(np.float64)
    eps = 1e-6
    logged = np.log(np.clip(wss_all, 0, None) + eps)
    stats = {
        "method": "log_z",
        "eps": eps,
        "n_points": int(wss_all.size),
        "n_cases": len(buckets["train"]),
        "partitions": ["train"],
        "timesteps_scope": "peak",
        "log": {"mean": float(logged.mean()), "std": float(logged.std() + 1e-12)},
        "split_name": split.get("split_version", "split_AG_wss_min_v1_traintest"),
        "feature_names": ["x", "y", "z", "abscissa_norm", "local_radius", "curvature"],
        "n_train": len(buckets["train"]),
        "n_val": len(buckets["val"]),
        "n_test": len(buckets["test"]),
    }
    with open(stats_json, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved: {stats_json}")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    train_output = os.path.join(OUT_DIR, "wss_train.pkl")
    val_output = os.path.join(OUT_DIR, "wss_val.pkl")
    test_output = os.path.join(OUT_DIR, "wss_test.pkl")
    stats_output = os.path.join(OUT_DIR, "wss_norm_stats.json")

    print("Reading preprocessed bundles...")
    dataset, meta, split = process_directory(DATA_ROOT, SPLIT_JSON)

    print("Splitting dataset by AG official split...")
    split_and_save_dataset(
        dataset, meta, split,
        train_output, val_output, test_output, stats_output,
    )
