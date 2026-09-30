"""wss_min_phys1d_v1 (2026-09-26): one-dimensional reduced-order haemodynamics prior per wall point.

Deployment-legal inputs only (centerline atlas radius, branch flow shares from the virtual-cap Murray rule, and the
CFD protocol constants that are identical for every case):
  * Q(t): the protocol inlet volume-flow waveform (vf-in monitor of a library case; the same waveform drives every
    case; the peak export step 1162 is phase 0.21 s of the 0.8 s cycle);
  * blood: rho = 1060 kg/m3, Carreau viscosity mu(gamma) = mu_inf + (mu_0 - mu_inf) (1 + (lambda gamma)^a)^((n-1)/a)
    with mu_inf 0.0035, mu_0 0.16, lambda 8.2 s, a 0.64, n 0.2128 (the case UDF);
  * branch flow share q (log_q_branch_murray_cap of wss_min_flowref_v1 = the outlet-area Murray rule of the protocol);
  * R(s): atlas radius at the point's station.

Features (float32 per valid wall node, same rows as the frozen view):
  log_tau_1d_pois  ln Pa  Poiseuille wall shear at the peak step with the Carreau viscosity evaluated at the
                          Poiseuille wall shear rate 4 Q / (pi R^3) (one fixed-point iteration; captures the
                          low-shear viscosity rise in aneurysm sacs)
  log_tau_1d_wom   ln Pa  Womersley (rigid straight tube, harmonics of Q(t)) wall shear at the peak step, times the
                          same Carreau ratio mu(gamma)/mu_inf: the radius-dependent phase/amplitude correction of
                          pulsatile flow that Poiseuille misses (large trunk: high alpha; iliac leaves: low alpha)
  log_wom_alpha    ln     Womersley number of the fundamental harmonic, R sqrt(omega_1 / nu_inf)
  log_re_1d        ln     Reynolds number 2 rho Q / (pi R mu_inf) at the peak step
The pack also reports, per case, the zero-parameter fit of each prior against the CFD peak WSS (R2 in ln space and
in Pa) so the physics baseline itself can be quoted.

    python -m wss_v5.views.wall_phys1d_v1 --view-root <wss_min_view_v1> --flow-root <wss_min_flowref_v1> \
        --out <wss_min_phys1d_v1> --cases ... [--workers 12]
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
from scipy.special import jve

from .. import contract as C

PACK_NAME = "wss_min_phys1d_v1"
PACK_VERSION = "v1.0"
ROOT = C.ROOT
WAVEFORM_CASE = ROOT / "data_new/AG/slow/QIN_SI_FU/Global_conditions/vf-in-rfile.out"
RHO = 1060.0
MU_INF, MU_0, LAMBDA, A_CARREAU, N_CARREAU = 0.0035, 0.16, 8.2, 0.64, 0.2128
PERIOD = 0.8
DT = 0.005
PEAK_STEP = 1162
N_HARMONICS = 20
FLOOR_PA = 1e-3


def carreau_mu(gamma: np.ndarray) -> np.ndarray:
    return MU_INF + (MU_0 - MU_INF) * (1.0 + (LAMBDA * np.abs(gamma)) ** A_CARREAU) ** ((N_CARREAU - 1.0) / A_CARREAU)


def waveform_harmonics() -> tuple[np.ndarray, float, float]:
    """Complex one-sided Fourier coefficients Q_hat[n] (m3/s) of the last cycle, the peak-step flow, and the peak phase (s)."""
    data = np.loadtxt(WAVEFORM_CASE, skiprows=3)
    steps = data[:, 0].astype(int); q = data[:, 1]
    cyc = (steps >= 1120) & (steps < 1280)
    qc = q[cyc]; assert len(qc) == 160, len(qc)
    spec = np.fft.rfft(qc) / len(qc)  # spec[0] = mean; one-sided
    t_peak = (PEAK_STEP - 1120) * DT
    q_peak = float(q[steps == PEAK_STEP][0])
    return spec, q_peak, t_peak


def womersley_ratio(R: np.ndarray, n: int) -> np.ndarray:
    """tau_n / Q_n (Pa per m3/s) for harmonic n in a rigid tube of radius R (m), Newtonian mu_inf."""
    omega = 2.0 * np.pi * n / PERIOD
    nu = MU_INF / RHO
    alpha = R * np.sqrt(omega / nu)
    lam = alpha * np.exp(1j * 3.0 * np.pi / 4.0)  # i^(3/2) alpha
    j1_over_j0 = jve(1, lam) / jve(0, lam)
    num = -MU_INF * (lam / R) * j1_over_j0  # tau_w = -mu du/dr|_R with u = A (1 - J0(lam r/R)/J0(lam)); J0' = -J1
    den = np.pi * R ** 2 * (1.0 - 2.0 * j1_over_j0 / lam)
    return num / den


def one_case(args) -> dict:
    cid, view_root, flow_root, out_root = args
    started = time.time()
    with np.load(Path(view_root) / cid / "bundle.npz", allow_pickle=True) as b:
        node_id = b["wall_node_id_cas"].astype(np.int64); steps = b["steps"].tolist(); wss_peak = b["wall_wss"][steps.index(int(b["peak_step"]))].astype(np.float64)
        radius_mm = b["wall_local_radius"].astype(np.float64)
    with np.load(Path(flow_root) / cid / "features.npz") as f:
        assert np.array_equal(f["wall_node_id_cas"], node_id), f"{cid}: flowref rows differ"
        log_q = f["wall_log_q_branch_murray_cap"].astype(np.float64)
    spec, q_peak, t_peak = waveform_harmonics()
    R = np.clip(radius_mm, 0.3, None) / 1000.0
    share = np.exp(log_q)
    q_pt = q_peak * share
    # Poiseuille + Carreau fixed point
    gamma = 4.0 * q_pt / (np.pi * R ** 3)
    mu = carreau_mu(gamma)
    tau_pois = mu * gamma
    # Womersley at the peak step (Newtonian mu_inf), then the same Carreau ratio
    tau_w = np.full_like(R, 4.0 * MU_INF * spec[0].real * share / np.pi) / R ** 3
    for n in range(1, N_HARMONICS + 1):
        ratio = womersley_ratio(R, n)
        tau_w = tau_w + 2.0 * np.real(ratio * spec[n] * share * np.exp(1j * 2.0 * np.pi * n * t_peak / PERIOD))
    tau_wom = np.clip(tau_w, FLOOR_PA, None) * (mu / MU_INF)
    alpha1 = R * np.sqrt(2.0 * np.pi / PERIOD / (MU_INF / RHO))
    re = 2.0 * RHO * q_pt / (np.pi * R * MU_INF)
    payload = {"wall_node_id_cas": node_id, "wall_log_tau_1d_pois": np.log(np.clip(tau_pois, FLOOR_PA, None)).astype(np.float32),
               "wall_log_tau_1d_wom": np.log(tau_wom).astype(np.float32), "wall_log_wom_alpha": np.log(alpha1).astype(np.float32),
               "wall_log_re_1d": np.log(np.clip(re, 1e-3, None)).astype(np.float32)}
    for k, v in payload.items():
        if not np.isfinite(v).all():
            raise ValueError(f"{cid}: non-finite {k}")
    out_dir = Path(out_root) / cid; out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "features.tmp.npz"; np.savez(tmp, **payload); tmp.replace(out_dir / "features.npz")
    # zero-parameter fit report
    y = np.log(np.clip(wss_peak, FLOOR_PA, None)); valid = np.isfinite(y)
    def r2(a, b):
        return float(1.0 - np.sum((a - b) ** 2) / np.sum((a - a.mean()) ** 2))
    rep = {"canonical_id": cid, "pack": PACK_NAME, "version": PACK_VERSION, "n_points": int(len(node_id)), "seconds": round(time.time() - started, 2),
           "q_peak_m3s": q_peak, "t_peak_s": t_peak, "n_harmonics": N_HARMONICS,
           "fit_ln": {"pois": r2(y[valid], payload["wall_log_tau_1d_pois"][valid]), "wom": r2(y[valid], payload["wall_log_tau_1d_wom"][valid])},
           "fit_pa": {"pois": r2(wss_peak[valid], np.exp(payload["wall_log_tau_1d_pois"][valid])), "wom": r2(wss_peak[valid], np.exp(payload["wall_log_tau_1d_wom"][valid]))},
           "median_ratio_cfd_over_prior": {"pois": float(np.exp(np.median(y[valid] - payload["wall_log_tau_1d_pois"][valid]))), "wom": float(np.exp(np.median(y[valid] - payload["wall_log_tau_1d_wom"][valid])))},
           "wom_over_pois_median": float(np.exp(np.median(payload["wall_log_tau_1d_wom"] - payload["wall_log_tau_1d_pois"]))),
           "carreau_mu_median": float(np.median(mu)), "alpha1_range": [float(alpha1.min()), float(alpha1.max())]}
    (out_dir / "phys1d_report.json").write_text(json.dumps(rep, indent=1))
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--view-root", required=True); ap.add_argument("--flow-root", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--cases", nargs="*", required=True); ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args(argv)
    with ProcessPoolExecutor(a.workers) as ex:
        reps = list(ex.map(one_case, [(c, a.view_root, a.flow_root, a.out) for c in a.cases]))
    man = {"pack": PACK_NAME, "version": PACK_VERSION, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "view_root": a.view_root, "flow_root": a.flow_root,
           "constants": {"rho": RHO, "mu_inf": MU_INF, "mu_0": MU_0, "lambda_s": LAMBDA, "a": A_CARREAU, "n": N_CARREAU, "period_s": PERIOD, "peak_step": PEAK_STEP, "waveform_case": str(WAVEFORM_CASE)},
           "cases": len(reps), "fit_ln_pois_median": float(np.median([r["fit_ln"]["pois"] for r in reps])), "fit_ln_wom_median": float(np.median([r["fit_ln"]["wom"] for r in reps])),
           "fit_pa_pois_median": float(np.median([r["fit_pa"]["pois"] for r in reps])), "fit_pa_wom_median": float(np.median([r["fit_pa"]["wom"] for r in reps])), "reports": reps}
    Path(a.out).mkdir(parents=True, exist_ok=True); (Path(a.out) / "phys1d_manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps({k: v for k, v in man.items() if k != "reports"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
