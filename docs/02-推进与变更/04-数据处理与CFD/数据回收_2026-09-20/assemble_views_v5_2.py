"""拼装 v5.2 训练视图根（2026-09-23，只建符号链接，不复制数据）。

来源优先级：`views_v5_2_20260922`（91 个新单元 + 端点半径修正后重建的 11 例在库病例）> `views_v5_1`（其余在库病例）。
三个视图（wss_min_view_v1 / wss_min_geom_v2 / wss_min_flowref_v1）逐病例目录链接；view_manifest.json 按来源合并。
队列文件（split、wss_global_stats）不在这里写——等正式 v5.2 split 定稿后用 wss_min_view 的 write_split / write_wss_stats 生成。
用法：python assemble_views_v5_2.py --out data_wss_v5/views_v5_2_full_20260923 [--dry-run]
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
ROOT = Path("/public/newhome/cy/Digital_twin/GNN"); HERE = Path(__file__).resolve().parent
V51 = ROOT / "data_wss_v5/views_v5_1"; V52 = ROOT / "data_wss_v5/views_v5_2_20260922"
VIEWS = ("wss_min_view_v1", "wss_min_geom_v2", "wss_min_flowref_v1")
def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--dry-run", action="store_true"); a = ap.parse_args()
    split = json.loads((V51 / "wss_min_view_v1/split_V5_train136_test34.json").read_text()); library = split["train_cases"] + split["test_cases"]
    manifest = json.loads((HERE / "new_units_manifest.json").read_text()); new = manifest["new_units"]
    hold = {h["canonical_id"] for h in manifest.get("hold", [])}
    excluded = {e["canonical_id"] for e in manifest.get("excluded", [])}
    plan = {}
    for cid in library:
        plan[cid] = "v5.2-rebuilt" if (V52 / "wss_min_view_v1" / cid / "bundle.npz").is_file() else "v5.1"
    for cid in new:
        if cid in excluded: continue
        plan[cid] = "v5.2-new" if cid not in hold else "hold"
    counts = {k: sum(1 for v in plan.values() if v == k) for k in ("v5.1", "v5.2-rebuilt", "v5.2-new", "hold")}
    print("plan:", counts)
    if a.dry_run: return
    for view in VIEWS:
        for cid, src in plan.items():
            if src == "hold": continue
            root = V52 if src.startswith("v5.2") else V51
            s = root / view / cid; d = a.out / view / cid
            assert s.is_dir(), (view, cid, src)
            d.parent.mkdir(parents=True, exist_ok=True)
            if d.is_symlink() or d.exists(): os.remove(d) if d.is_symlink() else None
            os.symlink(s, d)
        # merged manifest (wss_min_view only has a manifest with reports)
        if view == "wss_min_view_v1":
            reps = {}
            for root in (V51, V52):
                mp = root / view / "view_manifest.json"
                if mp.is_file():
                    for r in json.loads(mp.read_text()).get("reports", []): reps[r["canonical_id"]] = r
            keep = [reps[c] for c in sorted(plan) if plan[c] != "hold" and c in reps]
            missing = [c for c in plan if plan[c] != "hold" and c not in reps]
            (a.out / view / "view_manifest.json").write_text(json.dumps({"view": view, "assembled": "2026-09-23", "sources": {"v5.1": str(V51), "v5.2": str(V52)},
                "cases": len(keep), "plan_counts": counts, "reports": keep, "missing_reports": missing}, indent=1, ensure_ascii=False))
            print(view, "manifest cases", len(keep), "missing reports", missing)
    (a.out / "assembly_plan.json").write_text(json.dumps({"counts": counts, "plan": plan}, indent=1, ensure_ascii=False)); print("→", a.out)
if __name__ == "__main__":
    main()
