import os
import numpy as np
import pandas as pd
import csv

def read_data(csv_file):
    x, y, z, u, v, w, p = [], [], [], [], [], [], []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for line_num, r in enumerate(reader, start=1):
            if line_num <= 6:
                continue
            x.append(float(r[0]))
            y.append(float(r[1]))
            z.append(float(r[2]))
            p.append(float(r[3]))
            u.append(float(r[5]))
            v.append(float(r[6]))
            w.append(float(r[7]))
    points = np.stack([x, y, z], axis=-1)
    values = np.stack([u, v, w, p], axis=-1)
    return points, values

def process_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if not filename.endswith(".csv"):
            continue
        if "volume" not in filename: # or "wall" not in filename:
            continue

        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        try:
            points, values = read_data(input_path)
            # mirrored_points = mirror_points_through_plane(points, normal, d)

            df = pd.DataFrame(points, columns=['x', 'y', 'z'])
            df['u'] = values[:, 0]
            df['v'] = values[:, 1]
            df['w'] = values[:, 2]
            df['p'] = values[:, 3]

            df.to_csv(output_path, index=False)
            print(f"✅ Mirrored and saved: {filename}")
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")


# === Process both folders ===
folders = [
    ("lt", "processed_lt"),
    ("rt", "processed_rt")
]

for in_folder, out_folder in folders:
    process_folder(in_folder, out_folder)
