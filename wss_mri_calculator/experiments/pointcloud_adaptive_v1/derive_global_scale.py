from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/stage_c_split173_s1200.json")
    parser.add_argument("--method", default="adaptive_cv_tol1p25")
    parser.add_argument("--role", default="train")
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    reports = [
        row["methods"][args.method]
        for row in payload["cases"]
        if row["truth_usable"] and row["role"] == args.role
    ]
    alpha = np.asarray([report["alpha"] for report in reports])
    raw_r2 = np.asarray([report["raw_r2"] for report in reports])
    scaled_r2 = np.asarray([report["scaled_r2"] for report in reports])

    # 对每病例有：R2(a) = scaled_R2 - w*(a-alpha_i)^2，且
    # w = (scaled_R2-raw_R2)/(1-alpha_i)^2。最小化病例平均 SSE/variance
    # 的全局尺度是 sum(w*alpha_i)/sum(w)。只使用 train 病例摘要。
    denominator = np.square(1.0 - alpha)
    weight = np.divide(
        scaled_r2 - raw_r2,
        denominator,
        out=np.zeros_like(alpha),
        where=denominator > 1e-12,
    )
    scale = float(np.sum(weight * alpha) / np.sum(weight))
    print(
        json.dumps(
            {
                "source": args.input,
                "method": args.method,
                "role": args.role,
                "n_cases": len(reports),
                "objective": "maximize mean case-level raw R2",
                "global_scale": scale,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
