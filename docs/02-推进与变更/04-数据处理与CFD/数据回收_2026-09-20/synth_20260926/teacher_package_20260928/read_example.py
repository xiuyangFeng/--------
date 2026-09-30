"""读取示例：python read_example.py AG/fast/PENG_JI_MING~m25
依赖：numpy、h5py（pip install numpy h5py）。只读，不改任何文件。"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np

case_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "AG/fast/PENG_JI_MING~m25")
h5_path = next(case_dir.glob("*.h5"))

with h5py.File(h5_path, "r") as f:
    print("病例", f.attrs["case_id"], "  母例", f.attrs["parent_id"])
    k = int(f["time"].attrs["peak_index"])                       # 峰值帧（step 1162）在 81 帧里的下标
    t = f["time/phase_s"][...]                                   # 周期内相位 0–0.80 s

    # 壁面点云：坐标 (N,3) mm；某一帧的压力 / WSS 三分量 / WSS 标量
    xyz = f["wall/xyz_mm"][...]
    p_wall = f["wall/pressure_pa"][k]                            # (N,)   Pa，表压
    tau_vec = f["wall/wss_vector_pa"][k]                         # (N,3)  Pa
    tau = f["wall/wss_scalar_pa"][k]                             # (N,)   Pa
    tawss = f["wall/wss_scalar_pa"][:80].mean(0)                 # 第 81 帧与第 1 帧同相位 → 取前 80 帧平均（近似 TAWSS）
    print(f"壁面 {len(xyz)} 点；峰值帧 WSS p50/p99 = {np.percentile(tau, 50):.2f}/{np.percentile(tau, 99):.2f} Pa；TAWSS 中位 {np.median(tawss):.2f} Pa")

    # 体点：解剖区网格单元中心；速度 (N,3) m/s
    vel = f["volume/velocity_m_s"][k]
    print(f"体点 {f['volume/xyz_mm'].shape[0]} 个；峰值帧最大流速 {np.linalg.norm(vel, axis=1).max():.2f} m/s")

    # 开口：逐帧流量（出口为正、入口为负）
    for lab in ("inlet", "out-le", "out-li", "out-re", "out-ri"):
        q = f[f"boundary/{lab}/flux_outward_m3s"][:80]            # 周期平均同样取前 80 帧
        print(f"  {lab:7s} {f['boundary'][lab].attrs['vessel_zh']:4s} 周期平均流量 {q.mean() * 1e6:8.2f} mL/s")

    hemo = json.loads(f.attrs["hemodynamics"])                   # 与 *_hemodynamics.json 相同（含 "_说明" 中文注释键）
    print("本例解读：", hemo["本例解读"]["出口分流与压力"])
    print("h5 数据集说明示例：", f["wall/wss_scalar_pa"].attrs["说明"])
    for o in hemo["outlets"]["per_outlet"]:
        print(f"  RCR {o['label']}: R1={o['R1_Pa_s_per_kg']:.4g} R2={o['R2_Pa_s_per_kg']:.4g} Pa·s/kg  C={o['C_kg_per_Pa']:.4g} kg/Pa")
    print("质量标记：", json.loads(f.attrs["quality_flags"])["notes"] or "无")

# STL（二进制，mm，外法向）；顶点与 wall/xyz_mm 同一批点，三角形与 wall/triangles 逐个相同
raw = next(case_dir.glob("*.stl")).read_bytes()
n_tri = int(np.frombuffer(raw[80:84], "<u4")[0])
print("STL 三角形数", n_tri)
