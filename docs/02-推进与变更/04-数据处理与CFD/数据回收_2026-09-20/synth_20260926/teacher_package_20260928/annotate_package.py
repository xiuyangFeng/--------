#!/usr/bin/env python
"""给交老师数据包加中文注释（在 build_package.py 之后运行；幂等，可重复跑）。

1. 每例 <名>_hemodynamics.json：每个参数前插入同名 "<键>_说明" 中文注释键，各块开头加 "说明"，文件开头加 "文件说明" 与
   "本例解读"（本例流量、分流、压力、形变、质量标记的具体数字）。仍是合法 JSON，原有键名与数值逐位不变（脚本自检）。
2. 每例 <名>.h5：根属性 hemodynamics 换成注释版 JSON；每个组 / 数据集加中文属性 "说明"（HDF5 的注释）。只改属性，不动数据。
3. manifest.csv 的 h5_sha256 同步更新。

    /public/newhome/cy/.conda/envs/GNN/bin/python annotate_package.py [--cases ID ...]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np

PKG = Path("/public/newhome/cy/Digital_twin/GNN/outputs/synth59_teacher_package_2026-09-28")
EXTRA_KEYS = {"文件说明", "说明", "本例解读", "viscosity_reference_mPa_s"}   # 注释版新增的键（剥离时去掉）
VIZ = Path("/public/newhome/cy/Digital_twin/GNN/docs/02-推进与变更/04-数据处理与CFD/数据回收_2026-09-20/synth_20260926/viz_sample15_20260928")
LABEL_SEM = {"out-le": "left_external", "out-li": "left_internal", "out-re": "right_external", "out-ri": "right_internal"}
SEM_ZH = {"trunk": "主干", "left_cia": "左髂总", "right_cia": "右髂总", "left_external": "左髂外", "left_internal": "左髂内",
          "right_external": "右髂外", "right_internal": "右髂内"}
ZH = {"out-le": "左髂外", "out-li": "左髂内", "out-re": "右髂外", "out-ri": "右髂内", "inlet": "主动脉入口"}


def strip(o):
    """去掉注释键，还原成 build_package.py 写出的原始参数（用于幂等与自检）。"""
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if k not in EXTRA_KEYS and not k.endswith("_说明")}
    if isinstance(o, list):
        return [strip(v) for v in o]
    return o


def with_notes(d: dict, notes: dict, head: str | None = None, extra_after: dict | None = None) -> dict:
    """按原键顺序输出；每个键前插入 "<键>_说明"（若有）；head 放在块首的 "说明"。"""
    out = {}
    if head:
        out["说明"] = head
    for k, v in d.items():
        if k in notes:
            out[f"{k}_说明"] = notes[k]
        out[k] = v
        if extra_after and k in extra_after:
            for ek, ev in extra_after[k].items():
                out[ek] = ev
    return out


def label_swaps() -> dict:
    """出口标签与母例库内标签相反的病例 → 母例语义 → 本包语义 的映射（来自 verification_60.csv 的 outlet_label_diff）。"""
    out = {}
    for r in csv.DictReader(open(VIZ / "verification_60.csv")):
        if r["outlet_labels_match_parent"] != "True" and r["outlet_label_diff"]:
            m = {}
            for item in r["outlet_label_diff"].split(";"):          # blood2:out-le->out-li
                a, b = item.split(":", 1)[1].split("->")
                m[LABEL_SEM[a]] = LABEL_SEM[b]
            out[r["child"]] = m
    return out


def bump_text(b: dict) -> str:
    return f"{b['segment_zh']}{'狭窄' if b['amplitude'] < 0 else '扩张'} a={b['amplitude']:+.3f} σ={b['sigma_mm']:.1f}mm s={b['s_local_mm']:.1f}mm (seg {b['segment_id']})"


def fmt_ml(x):
    return f"{x * 1e6:.2f} mL/s"


def annotate_case(args) -> dict:
    case_dir, swap = args
    jf = next(case_dir.glob("*_hemodynamics.json")); hf = next(case_dir.glob("*.h5"))
    h = strip(json.loads(jf.read_text()))                                   # 原始参数
    if swap:  # 形变段名原取自母例 atlas 语义；本例母例标签与本包相反 → 改成本包标签（已改过则跳过）
        with h5py.File(hf, "a") as f:
            morph = json.loads(f.attrs["morph"])
            if not morph.get("segment_labels_follow_package"):
                for b in morph["bumps"]:
                    b["segment_name"] = swap.get(b["segment_name"], b["segment_name"]); b["segment_zh"] = SEM_ZH[b["segment_name"]]
                morph["segment_labels_follow_package"] = True
                morph["segment_label_note"] = "本例母例库内的左髂内/左髂外标签与本包相反；segment_name/segment_zh 已按本包出口标签（出口边界名 + UDF）改正，segment_id 仍是母例 atlas 的段编号。"
                f.attrs["morph"] = json.dumps(morph, ensure_ascii=False)
    with h5py.File(hf, "r") as f:
        morph = json.loads(f.attrs["morph"]); flags = json.loads(f.attrs["quality_flags"])
        phase = f["time/phase_s"][...]
        flux = {lab: f[f"boundary/{lab}/flux_outward_m3s"][...] for lab in ZH}
        pmean = {lab: f[f"boundary/{lab}/p_mean_pa"][...] for lab in ZH}
    cid = h["case_id"]; fl, inl, outl, sol, idt = h["fluid"], h["inlet"], h["outlets"], h["solver"], h["identical_to_parent"]
    fr = inl["fourier"]; rho = fl["density_kg_m3"]
    q = np.asarray(inl["q_nominal_m3s_at_frames"]); qc = q[:-1]                # 第 81 帧与第 1 帧同相位，统计时去掉
    kpk, kmin = int(np.argmax(q)), int(np.argmin(qc))
    A = inl["A_udf_m2"]; ratio = inl["inlet_area_ratio_face_over_udf"]
    q_meas = -flux["inlet"][:-1]
    Qf = lambda t: fr["scale"] * (fr["a0"] + sum(fr["a"][k] * np.cos((k + 1) * fr["w"] * t) + fr["b"][k] * np.sin((k + 1) * fr["w"] * t) for k in range(8)))  # noqa: E731
    visc = {f"{g:g} 1/s": round(1000 * (fl["mu_inf_Pa_s"] + (fl["mu_0_Pa_s"] - fl["mu_inf_Pa_s"]) * (1 + (fl["lambda_s"] * g) ** fl["a"]) ** ((fl["n"] - 1) / fl["a"])), 1)
            for g in (0.1, 1, 10, 100, 1000)}
    po = {o["label"]: o for o in outl["per_outlet"]}
    rsum = {k: o["R1_Pa_s_per_kg"] + o["R2_Pa_s_per_kg"] for k, o in po.items()}; g = {k: 1 / v for k, v in rsum.items()}; gt = sum(g.values())
    share = {k: g[k] / gt for k in g}; r_total = 1 / gt
    q_out = {k: flux[k][:-1].mean() for k in po}; q_out_tot = sum(q_out.values())
    area = {k: o["opening_area_m2"] for k, o in po.items()}
    murray = {side: ((g[e] / g[i]), (area[e] / area[i]) ** 1.5) for side, (e, i) in {"左": ("out-le", "out-li"), "右": ("out-re", "out-ri")}.items()}
    murray_txt = "；".join(f"{s}侧 髂外/髂内 导纳比 {a:.3f}，按面积^1.5 应为 {b:.3f}（{'吻合' if abs(np.log(a / b)) < 0.05 else '不吻合'}）" for s, (a, b) in murray.items())
    tau = {k: rsum[k] * o["C_kg_per_Pa"] for k, o in po.items()}
    p_kpa = {k: pmean[k][:-1].mean() / 1000 for k in po}

    # ---- 形变概况
    bumps = morph["bumps"]
    btxt = "；".join(f"{b['segment_zh']}{'狭窄' if b['amplitude'] < 0 else '扩张'} {abs(b['amplitude']) * 100:.0f}%（中心在该段弧长 {b['s_local_mm']:.1f} mm 处，"
                    f"当地半径 {b['radius_at_center_mm']:.1f} mm，宽度 σ = {b['sigma_mm']:.1f} mm）" for b in bumps)
    cancel = ""
    if len(bumps) == 2 and bumps[0]["segment_id"] == bumps[1]["segment_id"] and np.sign(bumps[0]["amplitude"]) != np.sign(bumps[1]["amplitude"]) \
            and abs(bumps[0]["s_local_mm"] - bumps[1]["s_local_mm"]) < max(b["sigma_mm"] for b in bumps):
        cancel = "两个凸包位置几乎重合且一缩一扩，大部分相互抵消，实际几何与母例很接近。"
    summary = {
        "入口流量": f"标称周期平均 {fmt_ml(qc.mean())}，峰值 {fmt_ml(q.max())}（第 {kpk + 1} 个导出帧 = 峰值帧 step {sol['peak_frame']['step']}，相位 {phase[kpk]:.2f} s，"
                    f"入口平均流速 {q.max() / A:.3f} m/s），最小 {fmt_ml(qc.min())}（相位 {phase[kmin]:.2f} s）；h5 实测入流周期平均 {fmt_ml(q_meas.mean())}"
                    + ("" if abs(ratio - 1) < 1e-3 else f"。注意：本例实际入口面积是 UDF 面积常数的 {ratio:.3f} 倍，实际入流 = 标称 × {ratio:.3f}"),
        "出口分流与压力": "；".join(f"{ZH[k]} 按 RCR 应分 {share[k] * 100:.1f}%、实测 {q_out[k] / q_out_tot * 100:.1f}%（{fmt_ml(q_out[k])}），平均压力 {p_kpa[k]:.2f} kPa"
                              for k in ("out-le", "out-li", "out-re", "out-ri")),
        "RCR 规律核对": f"四出口并联总阻力 {r_total:.4g} Pa·s/kg；左右两侧导纳各占 {(g['out-le'] + g['out-li']) / gt * 100:.1f}% / {(g['out-re'] + g['out-ri']) / gt * 100:.1f}%；"
                      f"每个出口 (R1+R2)·C = {min(tau.values()):.3f}–{max(tau.values()):.3f} s；{murray_txt}",
        "形变": f"{btxt}。节点最大位移 {morph['max_displacement_mm']:.2f} mm，解剖区体积为母例的 {morph['anatomy_volume_ratio']:.3f} 倍。{cancel}",
        "质量标记": flags["notes"] or ["无"],
    }

    # ---- 各块注释
    top_notes = {
        "case_id": "本例编号：队列/亚组/病例名（ILO 为 ILO/姓名-k~mNN/before|after）。~mNN 表示由母例形变得到的第 NN 号合成子例。",
        "parent_id": "母例编号：本例由这个真实病例的 CFD 网格形变得到。",
        "note": "一句话说明：合成子例与母例的唯一区别是网格节点坐标。",
        "fluid": "血液物性（59 例全部相同）。",
        "inlet": "入口边界条件：主动脉入口处给定随时间周期变化的流量。",
        "outlets": "出口边界条件：4 个髂动脉出口各接一个三元 Windkessel（RCR）模型。",
        "walls": "壁面条件：血管壁刚性不动，贴壁流速为 0（无滑移）。",
        "solver": "求解器与时间设置（59 例全部相同）。",
        "identical_to_parent": "与母例设置一致性的逐项核验结果。",
    }
    fluid_notes = {
        "density_kg_m3": "血液密度，单位 kg/m³；按不可压流体处理。",
        "viscosity_model": "粘度模型：Carreau-Yasuda 非牛顿模型，写在 UDF 函数 cell_viscosity 里，Fluent 在每个网格单元按当地剪切率计算粘度。",
        "formula": "粘度公式：mu 为动力粘度（Pa·s），shear_rate 为剪切率 γ̇（1/s），其余符号见下面 5 个参数。剪切率越大粘度越低（剪切变稀）。",
        "mu_inf_Pa_s": "高剪切极限粘度 μ∞（Pa·s）：快速流动（剪切率很大）时趋近的粘度，0.0035 Pa·s = 3.5 mPa·s。",
        "mu_0_Pa_s": "零剪切粘度 μ0（Pa·s）：血液几乎静止时的粘度（红细胞聚集），0.16 Pa·s = 160 mPa·s。",
        "lambda_s": "松弛时间 λ（s）：决定粘度在哪个剪切率附近开始由高向低过渡（约从 1/λ ≈ 0.12 1/s 起明显下降）。",
        "a": "Yasuda 指数：控制过渡段曲线的弯曲形状。",
        "n": "幂律指数：n < 1 表示剪切变稀，数值越小变稀越明显。",
        "udf_defines_match": "核对标记：UDF 源码里 #define A1、B、D、E、n 五个常数与上面五个参数逐一相同时为 true。",
    }
    fluid_extra = {"udf_defines_match": {"viscosity_reference_mPa_s_说明": "按上面公式算出的几个剪切率下的粘度（mPa·s），仅供直观参考，不是求解输入。主流区剪切率多为几十到几百 1/s。",
                                         "viscosity_reference_mPa_s": visc}}
    fourier_notes = {
        "a0": f"常数项（mL/s，乘 scale 后为 m³/s）。因为基频 w 与周期不匹配（见 w 的说明），周期平均流量（本例 {qc.mean() * 1e6:.2f} mL/s）并不等于 a0。",
        "a": "余弦系数 a1…a8，依次对应 k = 1…8。",
        "b": "正弦系数 b1…b8，依次对应 k = 1…8。",
        "w": f"基频 w（rad/s）。注意 2π/w = {2 * np.pi / fr['w']:.3f} s，并不等于周期 {inl['period_s']} s：UDF 只取 0–{inl['period_s']} s 这一段循环使用，"
             f"所以每个周期起点流量从 {Qf(inl['period_s'] - 1e-6):.4e} 跳回 {Qf(0):.4e} m³/s（约 {abs(Qf(0) / Qf(inl['period_s'] - 1e-6) - 1) * 100:.1f}%，发生在低流量时刻，影响很小）。这是母例协议原样，59 例相同。",
        "scale": "单位换算系数：上面的系数按 mL/s 写，乘 1e-6 换成 m³/s。",
    }
    inlet_notes = {
        "bc_type": "边界类型：velocity-inlet（速度入口）。整个入口截面速度大小相同、方向垂直截面（平推流），数值由 UDF 函数 my_inlet 在每个时间步给出。",
        "flow_formula": "流量波形公式：Q(t) 单位 m³/s；tt 为一个周期内的时间（t 对周期取余），所以每个心动周期重复同一条波形；UDF 把流量除以 A_udf 得到入口速度（m/s）。",
        "fourier": "8 阶 Fourier 级数系数，59 例全部相同：Q(tt) = scale × [a0 + Σ_{k=1..8} (a_k·cos(k·w·tt) + b_k·sin(k·w·tt))]。",
        "period_s": "心动周期（s）。0.8 s 对应心率 75 次/分。",
        "A_udf_m2": f"UDF 公式里写死的入口面积（m²），用来把流量换成速度。本例 = {A * 1e4:.3f} cm²，等效直径 {2 * np.sqrt(A / np.pi) * 1000:.1f} mm。",
        "A_inlet_face_m2": "网格上实际的入口截面面积（m²）。",
        "inlet_area_ratio_face_over_udf": f"实际入口面积 ÷ A_udf。等于 1 时实际入流就是标称流量。本例 = {ratio:.4f}"
                                          + ("，实际入流等于标称流量。" if abs(ratio - 1) < 1e-3 else f"，实际入流只有标称值的 {ratio * 100:.1f}%。"),
        "actual_inflow_note": "实际入流的换算方法，以及到 h5 哪里去找逐帧实测入流。",
        "q_nominal_m3s_at_frames": f"81 个导出帧各自对应的标称入口流量（m³/s），与 h5 文件 time/step 的 81 帧一一对应（第 1 个与第 81 个是同一相位）。"
                                   f"本例周期平均 {fmt_ml(qc.mean())}，峰值 {fmt_ml(q.max())}（第 {kpk + 1} 个值，即峰值帧），最小 {fmt_ml(qc.min())}（相位 {phase[kmin]:.2f} s，舒张期几乎无流量）。",
    }
    outlets_notes = {
        "bc_type": "边界类型：pressure-outlet（压力出口）。出口压力不是常数，而是由 RCR 模型根据流过该出口的流量每个时间步算出。",
        "discretization": "UDF 每个时间步按此式更新出口压力：P_n 为本步压力（Pa），P_{n-1} 为上一步压力，Q_n、Q_{n-1} 为本步、上一步出口质量流量（kg/s），"
                          "dt 为时间步长 0.005 s，β = R2·C/dt；R1 近端阻力，R2 远端阻力，C 顺应性。",
        "units_note": "单位说明：UDF 按质量流量积分，所以阻力单位是 Pa·s/kg、顺应性单位是 kg/Pa。",
        "per_outlet": "4 个出口的逐项参数，顺序为 左髂外、左髂内、右髂外、右髂内。" + summary["出口分流与压力"] + "。",
        "protocol": f"RCR 协议组别：AG 队列一套，AAA 与 ILO 共用一套（两套只在总阻力上不同）。本例 {outl['protocol']}，四出口并联总阻力 {r_total:.4g} Pa·s/kg。",
        "rcr_rule": f"RCR 分配规则：①四出口并联总阻力为协议常数；②左右两侧各占 50%；③同侧髂外/髂内按开口面积的 1.5 次方分配（Murray r³ 规律）；"
                    f"④每个出口 (R1+R2)·C = 1.79 s。本例核对：{murray_txt}。",
    }
    per = []
    for k in ("out-le", "out-li", "out-re", "out-ri"):
        o = po[k]
        on = {
            "label": "出口代号：out-le 左髂外 / out-li 左髂内 / out-re 右髂外 / out-ri 右髂内（le、li、re、ri = left external、left internal、right external、right internal）。",
            "vessel": "对应血管（英文）。", "vessel_zh": "对应血管（中文）。",
            "fluent_bc_name": "Fluent 里真正施加压力边界条件的面（位于出口延伸段末端）的名字。",
            "fluent_bc_zone_id": "上面这个边界面在 Fluent 算例中的面区编号。",
            "fluent_interface_face_zone": "解剖区与出口延伸段交界截面的名字，即点云和 STL 在此敞开的开口。",
            "udf_profile": "该出口挂的 UDF 函数名，R1、R2、C 的数值就写在这个函数里（见 cfd_setup/udf-inlet*.c）。",
            "R1_Pa_s_per_kg": f"近端阻力 R1（Pa·s/kg，质量流量口径）：相当于大动脉的特征阻抗。本例占该出口总阻力 R1+R2 的 {o['R1_Pa_s_per_kg'] / rsum[k] * 100:.1f}%。",
            "R2_Pa_s_per_kg": f"远端阻力 R2（Pa·s/kg）：相当于外周小动脉与毛细血管床的阻力。本出口总阻力 R1+R2 = {rsum[k]:.4g} Pa·s/kg。",
            "C_kg_per_Pa": f"顺应性 C（kg/Pa）：相当于下游血管的弹性储血能力。本出口 (R1+R2)·C = {tau[k]:.3f} s（协议统一的时间常数 1.79 s）。",
            "R1_Pa_s_per_m3": "同一组 R1、R2、C 换算成体积流量口径（R × 密度 1060，C ÷ 密度），单位 Pa·s/m³ 与 m³/Pa，方便与文献对比。",
            "udf_matches_bundle": "核对标记：从 UDF 源码解析出的 R1、R2、C 与我们数据库记录逐位相同时为 true。",
            "opening_area_m2": f"该出口开口（解剖区交界截面）面积（m²），本例 = {o['opening_area_m2'] * 1e6:.2f} mm²。RCR 按这个面积以 Murray 规律分配；"
                               f"按本组 RCR 该出口应分到总流量的 {share[k] * 100:.1f}%，h5 实测周期平均 {fmt_ml(q_out[k])}（{q_out[k] / q_out_tot * 100:.1f}%），平均压力 {p_kpa[k]:.2f} kPa。",
        }
        per.append(with_notes(o, on))
    solver_notes = {
        "software": "求解软件与版本：ANSYS Fluent 2023 R1（集群模块 fluent/231），三维双精度求解器 3ddp，64 核并行。",
        "flow": "流动模型：层流，不开湍流模型。依据：求解日志里的残差只有连续性和 x/y/z 速度 4 项，没有 k、ω 等湍流方程。",
        "time_scheme": "时间格式：瞬态，dual-time（双时间步），每个物理时间步内部迭代到收敛再前进。",
        "time_step_s": "物理时间步长 0.005 s（5 ms），一个 0.8 s 周期 = 160 步。",
        "n_time_steps": "总时间步数：1280 步 = 6.4 s = 8 个心动周期。前 7 个周期让流场从初始状态进入周期稳定，只导出最后一个周期。",
        "max_iterations_per_step": "每个时间步最多内迭代次数。",
        "operating_pressure_Pa": "操作压力 101325 Pa（1 个标准大气压）。",
        "pressure_output": "本包所有压力都是表压 = 绝对压力 − 操作压力，所以数值在 1 万多 Pa（约 100 mmHg）量级。",
        "exported_frames": "导出帧：最后一个周期 step 1120–1280，每 2 步（0.01 s）一帧，共 81 帧；第 1 帧与第 81 帧是同一相位（周期首尾）。",
        "peak_frame": "峰值帧（入口流量最大的时刻）：step 为求解步号，index 为在 81 帧中的下标（从 0 数），phase_s 为周期内相位（s）。",
        "continuity_residual_tier": f"收敛质量分层（组内口径）：取 81 个导出步各自最后一次连续性残差的最大值，A ≤ 1e-3、B < 2e-3、C < 5e-3、D ≥ 5e-3。本例 {sol['continuity_residual_tier']} 档。",
        "continuity_last_max": "上述 81 步中最大的那个连续性残差。",
    }
    cas_notes = {
        "node_header_equal": "网格节点段结构（节点总数与编号范围）与母例相同，即没有增删节点、只移动了坐标。",
        "n_nodes": "整个网格的节点数（含入口/出口延伸段与内部节点）。",
        "n_nodes_moved": "被形变移动的节点数（都在解剖区内，延伸段与开口附近的节点不动）。",
        "max_node_displacement_mm": "节点最大位移（mm）。",
        "bytes_compared": "参与逐字节比对的字节数：.cas 解压后去掉节点坐标段的全部内容。",
        "identical_except_nodes_and_paths": "true = 去掉节点坐标与文件路径后，.cas 与母例逐字节相同：边界条件类型、材料、求解器设置全都一样。",
        "path_strings_parent": "母例 .cas 里绝对路径（结果导出位置）的个数；比对前已统一成文件名。",
        "path_strings_child": "子例 .cas 里绝对路径的个数（与母例相同）。",
    }
    ident_notes = {
        "parent": "对照的母例。",
        "udf_parent_file": "母例使用的 UDF 源文件名。", "udf_child_file": "本例使用的 UDF 源文件名（已放在 cfd_setup/ 里）。",
        "udf_identical": "true = 两个 UDF 文件逐字节相同，即入口波形、粘度模型、4 个出口的 RCR 完全一样。",
        "udf_sha256": "UDF 文件的 SHA-256 指纹，可用 sha256sum cfd_setup/udf-inlet*.c 自行核对。",
        "journal_identical_except_case_path": "true = 求解 journal（初始化、时间步长、步数、每步迭代数）除第一行算例文件路径外逐字节相同。",
        "slurm_identical": "集群提交脚本 fluent.slurm 是否与母例逐字节相同。",
        "slurm_identical_except_scheduling": "去掉作业调度差异（指定计算节点、并行核数）后是否相同；true 表示差别与任何物理设置无关。",
        "cas": "Fluent 算例文件（网格 + 全部设置）的逐字节比对结果。",
        "all_settings_identical": "综合结论：UDF、journal、.cas 三项都一致即为 true。",
    }
    ident = dict(idt); ident["cas"] = with_notes(idt["cas"], cas_notes)
    fourier = with_notes(fr, fourier_notes)
    ann = {
        "文件说明": "本文件记录这个合成病例 CFD 的全部血流动力学设置（血液、入口、出口、壁面、求解器）以及与母例的一致性核验。"
                  "每个参数前面紧跟一个以“_说明”结尾的键，是给人看的中文注释，程序读取时忽略即可；每个参数块（fluid、inlet 等）前面的“块名_说明”是该块的总体解释，块内的 per_outlet 等列表同样逐项带注释。"
                  "键名末尾是单位：_kg_m3 = kg/m³，_Pa_s = Pa·s，_m2 = m²，_s = 秒，_m3s = m³/s，_Pa_s_per_kg = Pa·s/kg，_kg_per_Pa = kg/Pa，_mm = 毫米。",
        "本例解读": summary,
    }
    ann.update(with_notes({"case_id": h["case_id"], "parent_id": h["parent_id"], "note": h["note"]}, top_notes))
    ann["fluid_说明"] = top_notes["fluid"]; ann["fluid"] = with_notes(fl, fluid_notes, extra_after=fluid_extra)
    inl2 = dict(inl); inl2["fourier"] = fourier
    ann["inlet_说明"] = top_notes["inlet"]; ann["inlet"] = with_notes(inl2, inlet_notes)
    out2 = dict(outl); out2["per_outlet"] = per
    ann["outlets_说明"] = top_notes["outlets"]; ann["outlets"] = with_notes(out2, outlets_notes)
    ann["walls_说明"] = top_notes["walls"]; ann["walls"] = h["walls"]
    ann["solver_说明"] = top_notes["solver"]; ann["solver"] = with_notes(sol, solver_notes)
    ann["identical_to_parent_说明"] = top_notes["identical_to_parent"]; ann["identical_to_parent"] = with_notes(ident, ident_notes)

    # 自检：去掉注释后与原始参数逐位相同；所有原始键都有注释
    assert strip(ann) == h, f"{cid}: annotation changed data"
    text = json.dumps(ann, ensure_ascii=False, indent=1)
    jf.write_text(text + "\n")

    # ---- h5：注释版参数 + 每个组/数据集的中文说明属性
    H5_NOTES = {
        "/": "合成病例 CFD 数据（全周期 81 帧）。wall = 壁面点云，volume = 解剖区体点，boundary = 5 个开口，time = 时间轴。根属性 hemodynamics 为带中文注释的血流动力学设置（与 *_hemodynamics.json 相同），morph 为形变参数，quality_flags 为质量标记。",
        "time": "时间轴：81 个导出帧（最后一个心动周期，每 0.01 s 一帧）。属性 peak_index 为峰值帧在 81 帧中的下标。",
        "time/step": "Fluent 求解步号（1120–1280，每 2 步一帧）。",
        "time/time_s": "求解物理时间（s），5.60–6.40 s。",
        "time/phase_s": "周期内相位（s），0–0.80 s；第 1 帧与第 81 帧同相位。",
        "wall": "壁面点云：解剖区壁面节点（与 STL 顶点逐点对应）。壁面无滑移，速度恒为 0，故速度只在 volume 组。",
        "wall/xyz_mm": "壁面点坐标 (N,3)，mm，Fluent 算例坐标系（与 STL 相同）。",
        "wall/normal_out": "壁面外法向单位向量 (N,3)，由网格面法向面积加权平均得到。",
        "wall/area_m2": "每个壁面点分到的面积 (N,)，m²；总和 = 解剖区壁面总面积，用于面积加权积分。",
        "wall/fluent_node_id": "该点在 Fluent 算例中的节点编号 (N,)，用于对照 cfd_setup 里的 .cas。",
        "wall/triangles": "壁面三角网 (M,3)，0 起始索引指向 wall 的行；法向朝外，与 STL 三角形逐个相同。",
        "wall/pressure_pa": "壁面压力 (81,N)，Pa，表压（相对 101325 Pa）。第一维是帧。",
        "wall/wss_vector_pa": "壁面切应力 WSS 三分量 (81,N,3)，Pa，x/y/z 方向。",
        "wall/wss_scalar_pa": "壁面切应力 WSS 标量 (81,N)，Pa，Fluent wall-shear 导出列；与三分量合成的模长有 0.1–0.6% 的插值差，做标量任务直接用本数据集。",
        "volume": "体点：解剖区（与壁面同一区域，不含入口/出口延伸段）网格单元中心，数值为单元中心值。",
        "volume/xyz_mm": "体点坐标 (M,3)，mm，网格单元中心。",
        "volume/volume_m3": "单元体积 (M,)，m³，用于体积加权积分。",
        "volume/fluent_cell_id": "该单元在 Fluent 算例中的编号 (M,)。",
        "volume/pressure_pa": "体点压力 (81,M)，Pa，表压。",
        "volume/velocity_m_s": "体点速度三分量 (81,M,3)，m/s。",
        "boundary": "5 个开口（解剖区与延伸段交界截面）：inlet 主动脉入口，out-le 左髂外，out-li 左髂内，out-re 右髂外，out-ri 右髂内。属性给面积、中心、外法向和 Fluent 出口边界名。",
    }
    for lab in ZH:
        H5_NOTES[f"boundary/{lab}"] = f"{ZH[lab]}开口。属性：area_m2 面积，center_mm 中心，normal_out 外法向，fluent_bc_name 对应的 Fluent 边界名。"
        H5_NOTES[f"boundary/{lab}/flux_outward_m3s"] = "逐帧通过该开口的体积流量 (81,)，m³/s；流出为正，入口为负。"
        H5_NOTES[f"boundary/{lab}/p_mean_pa"] = "逐帧该开口截面的面积平均压力 (81,)，Pa，表压。"
        H5_NOTES[f"boundary/{lab}/face_center_mm"] = "开口截面上各网格面的中心 (K,3)，mm。"
        H5_NOTES[f"boundary/{lab}/face_area_vec_m2"] = "开口截面上各网格面的面积向量 (K,3)，m²，方向朝外。"
    with h5py.File(hf, "a") as f:
        f.attrs["hemodynamics"] = text
        for path, note in H5_NOTES.items():
            (f if path == "/" else f[path]).attrs["说明"] = note
    return {"case_id": cid, "json": str(jf), "h5": hf, "n_notes": text.count("_说明\""), "bumps": [bump_text(b) for b in bumps]}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cases", nargs="*"); a = ap.parse_args()
    rows = list(csv.DictReader(open(PKG / "manifest.csv"))); fields = list(rows[0].keys())
    ids = a.cases or [r["case_id"] for r in rows]
    swaps = label_swaps()
    with Pool(8) as pool:
        res = pool.map(annotate_case, [(PKG / c, swaps.get(c)) for c in ids])
        shas = dict(zip(ids, pool.map(sha256, [r["h5"] for r in res])))
    bumps = {r["case_id"]: r["bumps"] for r in res}
    for r in rows:
        if r["case_id"] in shas:
            r["h5_sha256"] = shas[r["case_id"]]
            r["bump1"], r["bump2"] = (bumps[r["case_id"]] + [""])[:2]
    with open(PKG / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"annotated {len(res)} cases; notes per file: {min(r['n_notes'] for r in res)}–{max(r['n_notes'] for r in res)}")


if __name__ == "__main__":
    main()
