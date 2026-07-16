"""AG v2/v3 配准阶段的历史坐标质量检查工具。"""

from dataclasses import replace

from pipeline_wss_min import config as C


def legacy_registration_for_case(
    cohort: str,
    case: str,
    base: C.RegistrationConfig | None = None,
) -> C.RegistrationConfig:
    """应用病例覆盖项，并把归档工具固定在历史中心线坐标架。"""
    configured = C.registration_for_case(cohort, case, base or C.DEFAULT.registration)
    return replace(configured, frame_mode="legacy_centerline")
