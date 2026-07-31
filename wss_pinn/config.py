"""实验配置：默认值合并、路径解析、按 Gate 阶段的硬校验。

学习要点
--------
``ExperimentConfig`` 是所有训练/评估脚本的统一入口。读配置时会：

1. 把用户 JSON **深合并**到 ``DEFAULTS``（缺省字段自动补齐）；
2. 把相对路径解析成仓库根下的绝对路径；
3. ``validate()``：按 ``experiment.stage``（p0*/p1/f0u/f0up/f1）检查
   模型开关与 loss 权重是否匹配「一次只开一类物理项」的阶梯约定。

阅读建议：先看 ``DEFAULTS`` 各段含义，再看 ``validate`` 里对 F0-U / F0-UP / F1
的约束——那就是实验矩阵的「合同」。
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .utils import ROOT, guard_write_path, sha256_json


# 当前代码实现已支持的阶段代号（小写）
VALID_STAGES = {"p0a", "p0b", "p0c", "p0d", "p1", "f0u", "f0up", "f1"}


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """递归合并字典：``update`` 覆盖 ``base``，子 dict 继续深合并而非整段替换。"""
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


# ---------------------------------------------------------------------------
# 默认配置：用户 JSON 只需写「与默认不同」的字段
# ---------------------------------------------------------------------------
DEFAULTS: dict[str, Any] = {
    # 实验元信息：id 必填；stage 决定 validate 规则；diagnostic_only 允许 F1 消融
    "experiment": {"id": "", "stage": "", "diagnostic_only": False},
    # 四条路径根：前两个必须冻结指向 data_new / data_wss_min
    "paths": {
        "raw_root": "data_new",
        "wss_bundle_root": "data_wss_min",
        "sidecar_root": "data_wss_pinn/pilot_v1",
        "output_root": "outputs/wss_pinn",
    },
    # 数据划分与角色过滤
    "data": {
        "split_path": "wss_pinn/configs/pilot_cases.json",
        # pilot 必须显式标成「复用开发筛选集」，禁止误写成独立测试确认
        "split_label": "reused_development_screen",
        "cohorts": ["AG", "AAA", "ILO"],
        "timesteps": ["peak"],  # 当前只做峰值时刻
        "train_roles": ["train", "unspecified"],
        "eval_roles": ["train", "test", "unspecified"],
    },
    # 构建 sidecar 时三槽采样点数；manifest_path 是汇总索引
    "sampling": {
        "wall_points": 5000,
        "near_wall_points": 8000,
        "core_points": 8000,
        "seed": 1234,
        "manifest_path": "data_wss_pinn/pilot_v1/sampling_manifest.json",
    },
    # 网络结构开关与宽度（见 models/field_decoder.py）
    "model": {
        "predict_direct_wss": True,
        "predict_velocity": True,
        "predict_pressure": False,
        "geometry_hidden": 64,
        "geometry_latent": 32,
        "field_hidden": 128,
        "field_layers": 4,
        "fourier_frequencies": 6,
        "architecture_version": "shared_v1",  # 或 split_v2：WSS/场分 trunk
    },
    # 损失权重：阶梯实验靠改这些数字开/关项（非改代码）
    "loss": {
        "direct_wss_weight": 1.0,
        "velocity_data_weight": 1.0,
        "pressure_data_weight": 0.0,
        "continuity_weight": 0.0,
        "no_slip_weight": 0.0,
        "wss_physics_weight": 0.0,  # F2 之前必须保持 0
        "momentum_weight": 0.0,  # F3 之前必须保持 0
    },
    # 物理常数：密度 + Carreau–Yasuda 参数（来自 UDF 审计）
    "physics": {
        "density_kg_m3": 1060.0,
        "rheology": {
            "model": "carreau_yasuda",
            "mu_inf_pa_s": 0.0035,
            "mu_zero_pa_s": 0.16,
            "lambda_s": 8.2,
            "a": 0.64,
            "n": 0.2128,
            "source": "data_new/AAA/ruputer/DING_JUN_FENG/udf-inlet4.c",
        },
        "characteristic_scales": {
            "length": "per_case_coord_scale_m",
            "velocity": "per_case_peak_p95_m_s",
            "pressure": "rho_u2",
        },
    },
    # 优化器与 batch；batch_* 是每个 epoch 每病例采样的点数
    "train": {
        "seed": 1234,
        "epochs": 2000,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "precision": "float32",
        "device": "cuda",
        "batch_wall": 512,
        "batch_near_wall": 512,
        "batch_core": 512,
        "checkpoint_every": 100,
        "log_every": 10,
        "batch_cases": 0,  # 0 = 每 epoch 用全部训练病例
        "init_checkpoint": None,  # 热启动权重（与 resume 互斥）
        "resume": None,  # 恢复优化器状态与 epoch
        "run_dir": "",
    },
    # Slurm 提交默认值（提交脚本读取）
    "cluster": {
        "partition": "GPU",
        "qos": "",
        "time": "24:00:00",
        "memory": "32G",
        "cpus": 8,
        "gpus": 1,
        "python": "/public/newhome/cy/.conda/envs/GNN/bin/python",
    },
}


class ExperimentConfig:
    """加载并校验一份实验 JSON，对外像 dict 一样用 ``config["train"]`` 访问。"""

    def __init__(self, payload: dict[str, Any], source: Path | None = None):
        self.payload = _deep_merge(DEFAULTS, payload)
        self.source = source
        self._resolve_paths()
        self.validate()

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        """从磁盘 JSON 构造；相对路径相对仓库根。"""
        source = Path(path).expanduser()
        if not source.is_absolute():
            source = ROOT / source
        with source.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(payload, source=source.resolve())

    def _resolve_paths(self) -> None:
        """把配置里常见相对路径统一成绝对路径字符串。"""
        for key in ("raw_root", "wss_bundle_root", "sidecar_root", "output_root"):
            value = Path(self.payload["paths"][key])
            self.payload["paths"][key] = str(
                (value if value.is_absolute() else ROOT / value).resolve()
            )
        for section, key in (
            ("data", "split_path"),
            ("sampling", "manifest_path"),
            ("train", "run_dir"),
            ("train", "init_checkpoint"),
            ("train", "resume"),
        ):
            value = self.payload[section].get(key)
            if value:
                path = Path(value)
                self.payload[section][key] = str(
                    (path if path.is_absolute() else ROOT / path).resolve()
                )

    def validate(self) -> None:
        """硬门禁：错误配置在启动训练前就失败，而不是训完才发现阶段不一致。"""
        exp = self.payload["experiment"]
        stage = str(exp.get("stage", "")).lower()
        if stage not in VALID_STAGES:
            raise ValueError(f"invalid experiment.stage: {stage!r}")
        if not exp.get("id"):
            raise ValueError("experiment.id is required")
        if self.payload["data"]["split_label"] != "reused_development_screen":
            raise ValueError("pilot split must be labelled reused_development_screen")

        # 上游根必须冻结，防止误指到别的数据树
        paths = self.payload["paths"]
        if Path(paths["raw_root"]) != (ROOT / "data_new").resolve():
            raise ValueError("paths.raw_root must remain the frozen data_new root")
        if Path(paths["wss_bundle_root"]) != (ROOT / "data_wss_min").resolve():
            raise ValueError("paths.wss_bundle_root must remain the frozen data_wss_min root")
        # 可写路径必须通过护栏
        guard_write_path(paths["sidecar_root"])
        guard_write_path(paths["output_root"])
        if self.payload["train"].get("run_dir"):
            guard_write_path(self.payload["train"]["run_dir"])
        guard_write_path(self.payload["sampling"]["manifest_path"])

        sampling = self.payload["sampling"]
        for key in ("wall_points", "near_wall_points", "core_points"):
            if int(sampling[key]) <= 0:
                raise ValueError(f"sampling.{key} must be positive")

        model = self.payload["model"]
        loss = self.payload["loss"]
        # 所有 F 阶段都保留直接 WSS 监督头（任务主指标）
        if not model["predict_direct_wss"] or float(loss["direct_wss_weight"]) <= 0:
            raise ValueError("all F stages retain a supervised WSS_direct output")
        # 下面两项超出当前 F1 实现范围，配置里必须为 0
        if float(loss["wss_physics_weight"]) != 0:
            raise ValueError("WSS physics is outside the F1 implementation scope")
        if float(loss["momentum_weight"]) != 0:
            raise ValueError("momentum must remain zero through F1")

        # —— 按阶段检查「合同」——
        if stage == "f0u":
            # 只开速度数据项；压力头与一阶物理项全关
            expected = (
                bool(model["predict_velocity"])
                and not bool(model["predict_pressure"])
                and float(loss["velocity_data_weight"]) > 0
                and float(loss["pressure_data_weight"]) == 0
                and float(loss["continuity_weight"]) == 0
                and float(loss["no_slip_weight"]) == 0
            )
            if not expected:
                raise ValueError("F0-U must be velocity data-only with pressure/physics off")
        elif stage == "f0up":
            # 在 F0-U 上增加压力监督；仍无 continuity / no-slip
            expected = (
                bool(model["predict_velocity"])
                and bool(model["predict_pressure"])
                and float(loss["velocity_data_weight"]) > 0
                and float(loss["pressure_data_weight"]) > 0
                and float(loss["continuity_weight"]) == 0
                and float(loss["no_slip_weight"]) == 0
            )
            if not expected:
                raise ValueError("F0-UP must add gauge pressure supervision only")
            # 科学门禁：必须能配对对照父 run（用于 Δ 指标）
            if not (
                exp.get("scientific_gate_control_run_dir")
                or exp.get("control_run_dir")
            ):
                raise ValueError(
                    "F0-UP requires scientific_gate_control_run_dir or control_run_dir"
                )
        elif stage == "f1":
            common = (
                bool(model["predict_velocity"])
                and bool(model["predict_pressure"])
                and float(loss["pressure_data_weight"]) > 0
            )
            continuity = float(loss["continuity_weight"])
            no_slip = float(loss["no_slip_weight"])
            if bool(exp.get("diagnostic_only")):
                # 诊断矩阵允许只开 continuity 或只开 no-slip
                expected = common and (continuity > 0 or no_slip > 0)
                message = "diagnostic F1 requires at least one first-order loss"
            else:
                expected = common and continuity > 0 and no_slip > 0
                message = "F1 requires pressure, continuity and no-slip"
            if not expected:
                raise ValueError(message)
            if not exp.get("control_run_dir"):
                raise ValueError("F1 requires a paired F0-UP control_run_dir")

        if stage.startswith("f") and not self.payload["train"].get("run_dir"):
            raise ValueError("train.run_dir is required for F stages")
        if self.payload["train"]["precision"] not in {"float32", "float64"}:
            raise ValueError("train.precision must be float32 or float64")
        if int(self.payload["train"].get("batch_cases") or 0) < 0:
            raise ValueError("train.batch_cases must be non-negative")
        if self.payload["train"].get("resume") and self.payload["train"].get(
            "init_checkpoint"
        ):
            raise ValueError("train.resume and train.init_checkpoint are mutually exclusive")
        for key in ("train_roles", "eval_roles"):
            roles = self.payload["data"].get(key)
            if not isinstance(roles, list) or not roles:
                raise ValueError(f"data.{key} must be a non-empty list")

    def as_dict(self) -> dict[str, Any]:
        """深拷贝整份已解析配置，用于落盘 resolved_config.json。"""
        return copy.deepcopy(self.payload)

    @property
    def stage(self) -> str:
        return self.payload["experiment"]["stage"].lower()

    @property
    def experiment_id(self) -> str:
        return str(self.payload["experiment"]["id"])

    @property
    def run_dir(self) -> Path:
        return Path(self.payload["train"]["run_dir"])

    @property
    def resolved_sha256(self) -> str:
        """解析后配置的内容哈希，写入 checkpoint 与 summary。"""
        return sha256_json(self.payload)

    def __getitem__(self, key: str) -> Any:
        """允许 ``config["loss"]`` 语法访问顶层段。"""
        return self.payload[key]
