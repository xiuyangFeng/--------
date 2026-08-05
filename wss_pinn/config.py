"""峰值体域实验矩阵的配置合同与硬护栏。

学习要点
--------
1. **默认值 + 深合并**：``DEFAULTS`` 给出完整科学/路径/训练默认；用户 JSON
   只需覆盖差异字段，经 ``_deep_merge`` 得到完整 payload。
2. **冻结科学合同**：``validate()`` 拒绝任何偏离合同的配置（split 哈希、
   peak-only、四通道输出、SiLU、Carreau–Yasuda、禁止热启动等）。
3. **data_only vs pinn**：data_only 必须关闭全部物理项与权重；pinn 必须从
   第一步同时启用 continuity / momentum / no-slip。
4. **写路径护栏**：run_dir / output_root / sidecar 等必须落在允许写入的
   PINN 根目录内（见 ``wss_pinn.utils.guard_write_path``）。
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from wss_pinn.utils import ROOT, guard_write_path, sha256_file, sha256_json


ROUTE_V1 = "volume_uvwp_peak_v1"
ROUTE_V3 = "volume_uvwp_peak_qs_smooth_v3"
VALID_INPUTS = {"xyz", "xyz_geom"}
GEOMETRY_FEATURES = [
    "abscissa_norm",
    "local_radius",
    "curvature_signed_log1p",
]
RHEOLOGY = {
    "model": "carreau_yasuda",
    "mu_inf_pa_s": 0.0035,
    "mu_zero_pa_s": 0.16,
    "lambda_s": 8.2,
    "a": 0.64,
    "n": 0.2128,
}
ROUTE_CONTRACTS = {
    ROUTE_V1: {
        "modes": {"data_only", "pinn"},
        "architectures": {"pointnet", "pointnetpp"},
        "activation": "silu",
        "split": (
            "wss_pinn/configs/splits/"
            "split_WSS_PINN_AG_AAA_ILO_q2v_pool2025_"
            "train138_test35_exclude_SHI_YUN_XI_v1.json"
        ),
        "split_sha256": (
            "964d7021f2d12baadd630e7b936456a4e62294fa72c5ada4fc70abe4d9361f2b"
        ),
    },
    ROUTE_V3: {
        "modes": {"data", "data_bc", "data_bc_pde"},
        "architectures": {"smooth_pointnet"},
        "activation": "tanh",
        "split": (
            "wss_pinn/configs/splits/"
            "split_WSS_PINN_AG_AAA_ILO_qs_smooth_v3_"
            "train123_val15_test35_s1234.json"
        ),
        "split_sha256": None,
    },
}


# ---------------------------------------------------------------------------
# 完整默认配置：用户 JSON 与此深合并后形成 resolved config。
# 字段注释见各子节；validate() 对关键科学字段做硬校验。
# ---------------------------------------------------------------------------
DEFAULTS: dict[str, Any] = {
    "route": ROUTE_V1,
    "contracts": {
        "split_sha256": "",
        "boundary_gate_sha256": "",
    },
    "experiment": {
        "id": "",              # 实验唯一 ID，例如 VF-PNPP-XYZG-PINN-s1234-v1
        "mode": "",            # data_only | pinn
        "architecture": "",    # pointnet | pointnetpp
        "input_variant": "",   # xyz | xyz_geom
        "description": "",
    },
    "paths": {
        # 冻结 split：排除 SHI_YUN_XI 后的 train138/test35
        "split": (
            "wss_pinn/configs/splits/"
            "split_WSS_PINN_AG_AAA_ILO_q2v_pool2025_"
            "train138_test35_exclude_SHI_YUN_XI_v1.json"
        ),
        # schema-v2 sidecar 清单与 train138-only 严格体域场统计
        "sidecar_manifest": (
            "data_wss_pinn/volume_uvwp_peak_v1_train138_test35/manifest.json"
        ),
        "field_stats": (
            "data_wss_pinn/volume_uvwp_peak_v1_train138_test35/field_stats.json"
        ),
        "output_root": "outputs/wss_pinn/volume_uvwp_peak_v1",
        "run_dir": "",  # 必须由具体实验 JSON 给出，禁止空目录
    },
    "data": {
        "timestep": "peak",  # 只做峰值时刻，不伪造 du/dt
        "train_roles": ["train"],
        "validation_roles": [],
        "eval_roles": ["test"],
        # xyz_geom 时追加的三几何通道；合同冻结，顺序不可改
        "geometry_features": GEOMETRY_FEATURES,
        # 压力标签：每例严格体域均值中心化后的相对压力 Pa（固定 gauge）
        "pressure_target": "case_strict_volume_mean_centered_pa",
    },
    "sampling": {
        "support_points": 5000,   # 编码器支撑点
        "query_points": 5000,     # 监督/查询点
        "physics_points": 512,    # 从 query 中取的 PDE 配点（≤ query）
        "wall_points": 1024,      # 独立壁面云上的 no-slip 点
        "inlet_points": 512,
        "outlet_points": 512,
        "strategy": "uniform_volume_random",  # 严格体域均匀随机
        "query_mode": "independent",  # independent | same（SAME5K 用 same）
        "resample_each_epoch": True,  # 每 epoch 重采样，禁止固定一次
        "seed": 1234,
    },
    "model": {
        "out_dim": 4,          # 固定四通道 u,v,w,p
        "activation": "silu",  # 可微查询路径；禁止 ReLU 等不可导拐点
        # 历史 P2V 宽度；只继承架构，不加载旧权重
        "pointnet": {
            "support_channels": [256, 512],
            "query_channels": [256, 512],
            "decoder_channels": [512, 256],
        },
        # 纯 PointNet++ D2 c125-k128
        "pointnetpp": {
            "width": 32,
            "sa_ratios": [0.25, 0.25, 0.25],
            "sa_radius": [0.05, 0.10, 0.20],
            "sa_nsample": [128, 16, 16],
            "sa_center_counts": [125, 125, 32],
            "sa_grouping": ["knn_cover", "ball", "ball"],
            "sa_center_sampling": "fps",
            "fp_knn": 3,
            "head_hidden": 64,
        },
        "smooth_pointnet": {
            "branch_channels": [64, 128, 256],
            "decoder_channels": [256, 256, 128],
        },
    },
    "physics": {
        "enabled": False,  # data_only=False；pinn 配置须显式 True
        "density_kg_m3": 1060.0,
        "reference_velocity_m_s": 1.0,  # 无量纲化参考速度 U0
        "shear_rate_epsilon_s_inv": 1e-6,  # 剪切率下界，避免 0 处数值崩
        # Carreau–Yasuda；动量残差保留变黏度 div(2μD)，禁止 μ∇²u 近似
        "rheology": RHEOLOGY,
        "residual_epsilon": 1e-12,
    },
    "loss": {
        "velocity_component_weights": [1.0, 1.0, 1.0],
        "pressure_weight": 1.0,
        "continuity_weight": 1.0,  # data_only 必须为 0；pinn 必须 >0
        "momentum_weight": 1.0,
        "no_slip_weight": 1.0,
        "lambda_bc": 0.0,
        "lambda_pde": 0.0,
    },
    "train": {
        "seed": 1234,
        "epochs": 400,
        "batch_cases": 2,  # 每 step 病例数（病例级 batch）
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "warmup_epochs": 10,
        "min_learning_rate": 1e-5,
        "grad_clip_norm": 1.0,
        "device": "cuda",
        "precision": "float32",
        "num_workers": 0,
        "log_every_steps": 10,
        "checkpoint_every_epochs": 10,
        "validation_every_epochs": 1,
        "milestone_epochs": [],  # 例如 [400,1000,2500,5000,7500]
        "resume": None,  # 仅允许同 run 目录断点续训
        # 故意不支持：PINN 不得加载 data-only 权重（热启动禁止）
        "init_checkpoint": None,
    },
    "cluster": {
        "partition": "GPU",
        "time": "24:00:00",
        "memory": "64G",
        "cpus": 8,
        "gpus": 1,
        "python": "/public/newhome/cy/.conda/envs/GNN/bin/python",
    },
}


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """递归深合并：``update`` 覆盖 ``base`` 的同名字段，嵌套 dict 继续合并。"""
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


class ExperimentConfig:
    """解析后的实验配置，附带科学合同与路径硬护栏。

    构造流程：深合并 DEFAULTS → 解析相对路径为绝对路径 → ``validate()``。
    任何合同违规在构造时立即抛错，避免带着非法配置启动训练。
    """

    def __init__(self, payload: dict[str, Any], source: Path | None = None):
        self.payload = _deep_merge(DEFAULTS, payload)
        self.source = source  # 原始 JSON 路径，便于追溯
        self._resolve_paths()
        self.validate()

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        """从 JSON 文件加载并构造配置；相对路径相对仓库根解析。"""
        source = Path(path).expanduser()
        if not source.is_absolute():
            source = ROOT / source
        payload = json.loads(source.read_text(encoding="utf-8"))
        return cls(payload, source=source.resolve())

    def _resolve_paths(self) -> None:
        """把 paths / train 中的相对路径解析为仓库根下的绝对路径。"""
        for key in (
            "split",
            "sidecar_manifest",
            "field_stats",
            "output_root",
            "run_dir",
        ):
            value = self.payload["paths"].get(key)
            if value:
                path = Path(value)
                self.payload["paths"][key] = str(
                    (path if path.is_absolute() else ROOT / path).resolve()
                )
        for key in ("resume", "init_checkpoint"):
            value = self.payload["train"].get(key)
            if value:
                path = Path(value)
                self.payload["train"][key] = str(
                    (path if path.is_absolute() else ROOT / path).resolve()
                )

    def validate(self) -> None:
        """硬校验科学合同、采样协议、物理常数与训练安全约束。

        失败一律 ``ValueError``：宁可启动前拒绝，也不要跑出不可对照的结果。
        """
        exp = self.payload["experiment"]
        route = str(self.payload.get("route", ROUTE_V1))
        contract = ROUTE_CONTRACTS.get(route)
        if contract is None:
            raise ValueError(f"unsupported route={route!r}")
        mode = str(exp.get("mode", ""))
        architecture = str(exp.get("architecture", ""))
        input_variant = str(exp.get("input_variant", ""))
        if not exp.get("id"):
            raise ValueError("experiment.id is required")
        if mode not in contract["modes"]:
            raise ValueError(f"invalid experiment.mode={mode!r}")
        if architecture not in contract["architectures"]:
            raise ValueError(f"invalid architecture={architecture!r}")
        if input_variant not in VALID_INPUTS:
            raise ValueError(f"invalid input_variant={input_variant!r}")

        # ---- 冻结 split：路径必须精确匹配，且文件哈希不得漂移 ----
        paths = self.payload["paths"]
        split = Path(paths["split"])
        expected_split = (ROOT / str(contract["split"])).resolve()
        if split != expected_split:
            raise ValueError(f"route split must remain frozen: {expected_split}")
        if split.exists():
            expected_hash = contract["split_sha256"] or str(
                self.payload["contracts"].get("split_sha256", "")
            )
            if not expected_hash or sha256_file(split) != expected_hash:
                raise ValueError("frozen route split hash drift")
        # ---- 写路径必须落在 PINN 允许根内 ----
        if paths.get("run_dir"):
            guard_write_path(paths["run_dir"])
        guard_write_path(paths["output_root"])
        guard_write_path(paths["sidecar_manifest"])
        guard_write_path(paths["field_stats"])

        # ---- 数据合同：peak-only、三几何特征、固定压力 gauge ----
        if self.payload["data"]["timestep"] != "peak":
            raise ValueError("volume-field routes are peak-only")
        if self.payload["data"]["geometry_features"] != GEOMETRY_FEATURES:
            raise ValueError("geometry feature contract must remain the frozen three fields")
        if (
            self.payload["data"]["pressure_target"]
            != "case_strict_volume_mean_centered_pa"
        ):
            raise ValueError("pressure target must use the fixed strict-volume case gauge")

        # ---- 采样：正点数、physics≤query、均匀体域、每 epoch 重采样 ----
        sampling = self.payload["sampling"]
        sampling_keys = ["support_points", "query_points", "physics_points", "wall_points"]
        if route == ROUTE_V3:
            sampling_keys.extend(["inlet_points", "outlet_points"])
        for key in sampling_keys:
            if int(sampling[key]) <= 0:
                raise ValueError(f"sampling.{key} must be positive")
        if int(sampling["physics_points"]) > int(sampling["query_points"]):
            raise ValueError("physics_points cannot exceed query_points")
        if sampling["strategy"] != "uniform_volume_random":
            raise ValueError("v1 sampling must be uniform over the full interior volume")
        if sampling["query_mode"] not in {"independent", "same"}:
            raise ValueError("sampling.query_mode must be independent or same")
        if (
            sampling["query_mode"] == "same"
            and int(sampling["support_points"]) != int(sampling["query_points"])
        ):
            raise ValueError("same query_mode requires support_points == query_points")
        if not bool(sampling["resample_each_epoch"]):
            raise ValueError("volume-field v1 must resample support/query each epoch")

        # ---- 模型：四通道 + 光滑 SiLU（PINN 需要坐标可微）----
        if int(self.payload["model"]["out_dim"]) != 4:
            raise ValueError("volume-field output must be four channels (u,v,w,p)")
        expected_activation = str(contract["activation"])
        if self.payload["model"]["activation"] != expected_activation:
            raise ValueError(
                f"{route} query decoder activation is frozen to {expected_activation}"
            )
        if route == ROUTE_V3:
            if self.payload["data"].get("validation_roles") != ["val"]:
                raise ValueError("V3 requires a fixed val15 role")
            if sampling.get("query_mode") != "independent":
                raise ValueError("V3 data queries must be independent of support")

        # ---- 物理常数与流变学冻结 ----
        physics = self.payload["physics"]
        if float(physics["density_kg_m3"]) != 1060.0:
            raise ValueError("volume-field v1 density is frozen at 1060 kg/m^3")
        if float(physics["reference_velocity_m_s"]) != 1.0:
            raise ValueError("volume-field v1 reference velocity is frozen at 1 m/s")
        if any(
            physics["rheology"].get(key) != value for key, value in RHEOLOGY.items()
        ):
            raise ValueError(f"volume-field rheology is frozen: {RHEOLOGY}")

        # ---- 模式与物理权重一致性 ----
        enabled = bool(physics["enabled"])
        physics_weights = [
            float(self.payload["loss"][key])
            for key in ("continuity_weight", "momentum_weight", "no_slip_weight")
        ]
        if route == ROUTE_V1:
            if mode == "data_only":
                if enabled or any(weight != 0.0 for weight in physics_weights):
                    raise ValueError("data_only must disable all physics terms")
            else:
                if not enabled or any(weight <= 0.0 for weight in physics_weights):
                    raise ValueError(
                        "pinn must enable continuity, momentum and no-slip from step one"
                    )
        else:
            lambda_bc = float(self.payload["loss"]["lambda_bc"])
            lambda_pde = float(self.payload["loss"]["lambda_pde"])
            expected = {
                "data": (False, 0.0, 0.0),
                "data_bc": (False, 1.0, 0.0),
                "data_bc_pde": (True, 1.0, 1.0),
            }[mode]
            if (enabled, lambda_bc, lambda_pde) != expected:
                raise ValueError(
                    "V3 modes are frozen to DATA=(0,0), DATA+BC=(1,0), "
                    "DATA+BC+PDE=(1,1)"
                )
            if any(weight != 1.0 for weight in physics_weights):
                raise ValueError("V3 raw continuity/momentum/wall weights remain one")
        # 禁止任何跨 run / data-only 热启动
        if self.payload["train"].get("init_checkpoint"):
            raise ValueError(
                "warm-start is forbidden: use the same seed, never a data-only checkpoint"
            )
        if self.payload["train"]["precision"] not in {"float32", "float64"}:
            raise ValueError("train.precision must be float32 or float64")
        if int(self.payload["train"]["batch_cases"]) <= 0:
            raise ValueError("train.batch_cases must be positive")
        if int(self.payload["train"].get("validation_every_epochs", 1)) <= 0:
            raise ValueError("train.validation_every_epochs must be positive")
        total_epochs = int(self.payload["train"]["epochs"])
        if total_epochs <= 0:
            raise ValueError("train.epochs must be positive")
        milestones = [
            int(value) for value in self.payload["train"].get("milestone_epochs", [])
        ]
        if milestones != sorted(set(milestones)):
            raise ValueError("train.milestone_epochs must be sorted and unique")
        if any(value <= 0 or value > total_epochs for value in milestones):
            raise ValueError("train.milestone_epochs must lie within train.epochs")
        if any(
            float(value) <= 0
            for value in self.payload["loss"]["velocity_component_weights"]
        ) or float(self.payload["loss"]["pressure_weight"]) <= 0:
            raise ValueError("all four supervised output weights must be positive")
        if route == ROUTE_V3 and (
            self.payload["loss"]["velocity_component_weights"] != [1.0, 1.0, 1.0]
            or float(self.payload["loss"]["pressure_weight"]) != 1.0
        ):
            raise ValueError("V3 uses an equal mean over u/v/w/p data losses")
        if not paths.get("run_dir"):
            raise ValueError("paths.run_dir is required")

    @property
    def mode(self) -> str:
        """``data_only`` 或 ``pinn``。"""
        return str(self.payload["experiment"]["mode"])

    @property
    def route(self) -> str:
        return str(self.payload.get("route", ROUTE_V1))

    @property
    def architecture(self) -> str:
        """``pointnet`` 或 ``pointnetpp``。"""
        return str(self.payload["experiment"]["architecture"])

    @property
    def input_variant(self) -> str:
        """``xyz`` 或 ``xyz_geom``。"""
        return str(self.payload["experiment"]["input_variant"])

    @property
    def run_dir(self) -> Path:
        """本实验 run 输出目录（绝对路径）。"""
        return Path(self.payload["paths"]["run_dir"])

    @property
    def resolved_sha256(self) -> str:
        """解析后完整 payload 的稳定 SHA256，用于对照「配置是否真相同」。"""
        return sha256_json(self.payload)

    def as_dict(self) -> dict[str, Any]:
        """返回 payload 深拷贝，供写入 ``resolved_config.json`` / checkpoint。"""
        return copy.deepcopy(self.payload)

    def __getitem__(self, key: str) -> Any:
        """允许 ``config["train"]`` 风格访问顶层节。"""
        return self.payload[key]


# Backward-compatible name used by completed experiment scripts/checkpoints.
VolumeExperimentConfig = ExperimentConfig
