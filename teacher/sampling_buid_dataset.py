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



def process_case(input_file):
    points, values = read_data(input_file)                            # (N, 3), (N, 4)
    # centers, averaged_features = voxelize_and_average(points, values) # (M, 3), (M, 4)

    # # Combine into full tables for export
    # df_original = pd.DataFrame(
        # np.hstack([points, values]),
        # columns=["x", "y", "z", "u", "v", "w", "p"]
    # )

    # df_voxelized = pd.DataFrame(
        # np.hstack([centers, averaged_features]),
        # columns=["x", "y", "z", "u", "v", "w", "p"]
    # )

    # # File naming
    basename = os.path.basename(input_file).replace(".csv", "")
    # os.makedirs("debug_csv", exist_ok=True)
    # csv_original = f"debug_csv/{basename}_original.csv"
    # csv_voxelized = f"debug_csv/{basename}_voxelized.csv"

    # # Save for inspection
    # df_original.to_csv(csv_original, index=False)
    # df_voxelized.to_csv(csv_voxelized, index=False)

    print(f"Processed: {input_file}")
    # print(f"Saved original:  {csv_original}")
    # print(f"Saved voxelized: {csv_voxelized}")

    return basename, points.T, values.T # (3, M), (4, M)


def process_directory(input_dir):
    """
    Recursively search input_dir and its subfolders for CSV files
    containing 'wall' in their filename, and voxelize each case.
    """
    dataset = []
    valid_files = []

    # Recursively traverse all subdirectories
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".csv") and "wall" in file:
                valid_files.append(os.path.join(root, file))
    print(len(valid_files))
    # Sort for reproducibility
    valid_files = sorted(valid_files)

    # Process each file
    for case_file in tqdm(valid_files, desc="Voxelizing cases"):
        dataset.append(process_case(case_file))

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
    input_dir = '..\processed_data_voxelized'
    train_output = 'CHI_original_train.pkl'
    test_output = 'CHI_original_test.pkl'

    print("Processing and voxelizing data...")
    dataset = process_directory(input_dir)

    print("Splitting dataset...")
    split_and_save_dataset(dataset, train_output, test_output)
