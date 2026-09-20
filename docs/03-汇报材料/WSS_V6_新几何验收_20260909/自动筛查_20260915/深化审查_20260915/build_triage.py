"""Compatibility entry point for the final reconstruction-based review desk.

The first draft's 141/29 classification is retired. This entry point requires
the reconstructed 170-case candidate and its independent audit.
"""
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[5]
if __name__=='__main__':
    candidate=ROOT/'outputs/wss_v6_geometry_candidate_20260915_opt_final'
    if not (candidate/'audit_refined_geometry.json').is_file():
        raise SystemExit('Run the geometry refinement and independent audit before building the review desk.')
    subprocess.run([sys.executable,'-m','wss_v5.render_refined_review',
                    '--root',str(candidate),'--resolution',str(Path(__file__).with_name('case_resolution_audit.json')),
                    '--out',str(ROOT/'docs/03-汇报材料/WSS_V6_新几何验收_20260909/算法优化_20260915')],cwd=ROOT,check=True)
