import os
import csv
import pickle
import numpy as np
from tqdm import tqdm
import pandas as pd

def read_data(csv_file):
    data = pd.read_csv(csv_file)
    x,y,z,u,v,w,p = data['x'],data['y'],data['z'],data['u'],data['v'],data['w'],data['p'] # mm

    points = np.stack([x, y, z], axis=-1)  # (N, 3)
    values = np.stack([u, v, w, p], axis=-1)  # (N, 4)
    return points, values


def voxelize_and_average(points, features, voxel_size=0.00005):
    """
    Apply voxelization and compute distance-weighted average within each voxel.
    Input:
        points: (N, 3)
        features: (N, 4)
    Output:
        voxel_centers: (M, 3)
        voxel_features: (M, 4)
    """
    voxel_indices = np.floor(points / voxel_size).astype(int)  # (N, 3)
    unique_voxels, inverse_indices = np.unique(voxel_indices, axis=0, return_inverse=True)

    centers = (unique_voxels + 0.5) * voxel_size  # (M, 3)
    voxel_features = []

    for i, voxel in enumerate(unique_voxels):
        mask = (inverse_indices == i)
        voxel_points = points[mask]
        voxel_values = features[mask]

        center = centers[i]
        dists = np.linalg.norm(voxel_points - center, axis=1) + 1e-8  # avoid divide by zero
        weights = 1.0 / dists
        weights = weights / weights.sum()

        weighted_avg = (voxel_values * weights[:, None]).sum(axis=0)  # (4,)
        voxel_features.append(weighted_avg)

    voxel_features = np.stack(voxel_features, axis=0)  # (M, 4)
    return centers, voxel_features

def process_case(input_file, output_dir):
    points, values = read_data(input_file)                            # (N, 3), (N, 4)
    centers, averaged_features = voxelize_and_average(points, values) # (M, 3), (M, 4)

    df_voxelized = pd.DataFrame(
        np.hstack([centers, averaged_features]),
        columns=["x", "y", "z", "u", "v", "w", "p"]
    )

    # # File naming
    basename = os.path.basename(input_file).replace(".csv", "")
    os.makedirs(output_dir, exist_ok=True)
    csv_voxelized = f"{output_dir}/{basename}_voxelized.csv"

    # Save for inspection
    df_voxelized.to_csv(csv_voxelized, index=False)

    print(f"Processed: {input_file}")

    return basename, centers.T, averaged_features.T # (3, M), (4, M)


def process_directory(input_dir, output_dir):
    """
    Recursively search input_dir and its subfolders for CSV files
    containing 'wall' in their filename, and voxelize each case.
    """
    dataset = []
    valid_files = []

    # Recursively traverse all subdirectories
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            print(file)
            valid_files.append(os.path.join(root, file))
    print(len(valid_files))
    # Sort for reproducibility
    valid_files = sorted(valid_files)

    # Process each file
    for case_file in tqdm(valid_files, desc="Voxelizing cases"):
        dataset.append(process_case(case_file, output_dir))

    return dataset


def split_and_save_dataset(dataset, output_train, output_test, test_ratio=0.2):
    from sklearn.model_selection import train_test_split
    train_data, test_data = train_test_split(dataset, test_size=test_ratio, random_state=42)
    with open(output_train, 'wb') as f:
        pickle.dump(train_data, f)
    with open(output_test, 'wb') as f:
        pickle.dump(test_data, f)
    print(f"Saved: {output_train} ({len(train_data)} cases), {output_test} ({len(test_data)} cases)")


if __name__ == "__main__":
    input_dir = 'CROWN_Dataset'
    output_dir = 'CROWN_Dataset/voxelized_lt'
    train_output = 'CHI_voxelized_train.pkl'
    test_output = 'CHI_voxelized_test.pkl'

    print("Processing and voxelizing data...")
    dataset = process_directory(input_dir, output_dir)

    print("Splitting dataset...")
    split_and_save_dataset(dataset, train_output, test_output)
