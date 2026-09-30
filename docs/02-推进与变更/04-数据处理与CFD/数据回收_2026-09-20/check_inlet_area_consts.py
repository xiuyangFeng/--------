"""只读：UDF 入口面积常量 vs 网格入口面积（SU_KAI_LI 型错误）。用法：python check_inlet_area_consts.py <病例目录 ...>"""
import glob, re, sys, numpy as np
sys.path.insert(0, "/public/newhome/cy/Digital_twin/GNN")
from wss_pinn.v4 import fluent_topology as FT
for d in sys.argv[1:]:
    try:
        cas=[p for p in glob.glob(d+"/*.cas.gz") if "orig" not in p][0]
        u=(glob.glob(d+"/udf-inlet4.c")+glob.glob(d+"/udf-inlet.c"))[0]; txt=open(u,errors="ignore").read()
        m=re.search(r"b8 \* sin\(8 \* tt \* w\)\)\s*/\s*([0-9.eE+-]+)", txt); c=float(m.group(1))
        mesh=FT.read_fluent_mesh(cas, keep_interior=False); a=None
        for s in mesh.face_sections:
            if s.bc_type==10:
                _,av=mesh.face_geometry(s); a=float(np.linalg.norm(av,axis=1).sum()); break
        print(f"{d:34s} mesh {a*1e6:9.2f} mm²  udf {c*1e6:9.2f} mm²  ratio {a/c:.4f}", flush=True)
    except Exception as e:
        print(f"{d:34s} ERR {type(e).__name__}: {e}", flush=True)
