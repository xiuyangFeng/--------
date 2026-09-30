"""Capture the exact model-input case dict of real stage-B runs (sandbox copies), then stop before inference."""
import json, pickle, shutil, sys
from pathlib import Path
ROOT = Path("/public/newhome/cy/Digital_twin/GNN"); sys.path.insert(0, str(ROOT))
import wss_deploy.pipeline as P
from wss_deploy.infer import Release

class Captured(Exception):
    pass

def capture(src: Path, release_id: str, out: Path, sandbox: Path):
    dst = sandbox / f"{src.name}__{release_id}"
    if dst.exists(): shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("*.vtp", "*.csv", "report*.html", "field.npz"))
    job = json.loads((dst / "job.json").read_text())
    rel = Release(ROOT / "outputs/wss_deploy_release" / release_id, device="cpu")
    def grab(release, case):
        with open(out, "wb") as fh: pickle.dump(case, fh, protocol=5)
        raise Captured
    P.run_inference = grab
    try:
        if rel.is_volume:
            from wss_deploy.volume_pipeline import stage_b_volume
            stage_b_volume(dst, job["mapping"], rel, confirmed=True)
        else:
            P.stage_b_wall(dst, job["mapping"], rel, confirmed=True)
    except Captured:
        c = pickle.load(open(out, "rb"))
        print(out.name, release_id, "n_pos", len(c["pos"]), "keys", len(c))

if __name__ == "__main__":
    sandbox = Path(sys.argv[1]); sandbox.mkdir(exist_ok=True)
    G = ROOT / "outputs/wss_deploy_golden/20260920_baseline"; J = ROOT / "outputs/wss_deploy_jobs"
    plan = [(G / "20260920_144135_673ccd0e36b1", "X5D_v51_5seed_20260916", "LV_x5d"),
            (G / "20260922_173705_27065942b465", "M1_3head_3seed_20260922", "LV_m1"),
            (G / "20260920_113510_0aa3ba155bfa", "PF6_VF6_peak_3seed_20260920", "LV_vol"),
            (J / "20260927_173546_79aaf41ce5f9", "M1_3head_3seed_20260922", "SHI_m1"),
            (J / "20260927_173546_79aaf41ce5f9", "X5D_v51_5seed_20260916", "SHI_x5d"),
            (J / "20260927_173635_38d9c5f3ec64", "PF6_VF6_peak_3seed_20260920", "SHI_vol")]
    for src, rid, name in plan:
        capture(src, rid, sandbox.parent / f"case_{name}.pkl", sandbox)
