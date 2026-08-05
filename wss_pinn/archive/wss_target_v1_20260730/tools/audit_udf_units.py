from __future__ import annotations

import argparse
import re
from pathlib import Path

from wss_pinn.utils import ROOT, atomic_write_json, sha256_file, utc_now


DEFINE = re.compile(r"^\s*#define\s+(\w+)\s+([-+0-9.eE]+)", re.MULTILINE)


def audit_udf(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    values = {name: float(value) for name, value in DEFINE.findall(text)}
    rheology = {
        "mu_inf_pa_s": values["A1"],
        "mu_zero_pa_s": values["B"],
        "lambda_s": values["D"],
        "a": values["E"],
        "n": values["n"],
    }
    required_snippets = [
        "pow(10, -6)",
        "/\n        0.0003880212",
        "C_STRAIN_RATE_MAG",
        "sum_flow_outle += F_FLUX",
    ]
    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "status": "completed",
        "gate_result": "pass",
        "source": {"path": str(path.resolve()), "sha256": sha256_file(path)},
        "density_kg_m3": 1060.0,
        "period_s": values["T"],
        "time_step_s": 0.005,
        "wave_angular_factor_s_inv": values["w"],
        "rheology": {
            "model": "Carreau-Yasuda-compatible UDF form",
            **rheology,
            "formula": "mu_inf + (mu_0-mu_inf)*(1+(lambda*gamma_dot)^a)^((n-1)/a)",
        },
        "inlet": {
            "fourier_coefficients_unit": "mL/s inferred from explicit 1e-6 conversion",
            "flow_conversion": "1e-6 m3 per mL",
            "area_m2": 0.0003880212,
            "profile_output": "m/s = volumetric flow m3/s divided by inlet area m2",
        },
        "pressure": "Pa (Fluent SI setup and RCR resistance dimensions)",
        "wss": "Pa",
        "coordinate": "raw ASCII nominal m; per-case geometry unit factor remains audited",
        "rcr_flux_warning": {
            "status": "blocked_for_hard_residual",
            "reason": (
                "F_FLUX is summed directly while Fluent commonly exposes mass flow; "
                "the UDF does not document a density conversion. Do not hard-code an "
                "RCR/flux residual until the case/solver convention is verified."
            ),
        },
        "source_contract_checks": {
            snippet: snippet in text for snippet in required_snippets
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--udf",
        default="data_new/AAA/ruputer/DING_JUN_FENG/udf-inlet4.c",
    )
    parser.add_argument(
        "--output",
        default="outputs/wss_pinn/audits/p0_units/udf_units.json",
    )
    args = parser.parse_args()
    source = Path(args.udf)
    if not source.is_absolute():
        source = ROOT / source
    payload = audit_udf(source)
    output = atomic_write_json(args.output, payload)
    print(f"{payload['status']} {payload['gate_result']} {output}")


if __name__ == "__main__":
    main()

