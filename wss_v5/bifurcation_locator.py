"""Geometry-only bifurcation transition locator.

The locator is intentionally a sidecar diagnostic.  It reads a candidate
``geometry.npz`` and ``report.json`` and never rewrites either input.  A valid
transition requires the two daughter centerline points on the *same* slice to
be enclosed by two different closed wall contours, preceded by a slice where
one contour encloses both points.  Other contours therefore cannot create a
false positive by merely increasing a contour count.
"""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
from .section_features import (SurfaceSlicer, contains_origin, frame, unit,
                               polyline_plane_intersection_groups)

SCHEMA = "wss_v6_bifurcation_locator_v1"

def _plain(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, (np.integer, np.floating)): return x.item()
    if isinstance(x, dict): return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [_plain(v) for v in x]
    return x

def _sample(xyz, sid, ss, seg, s):
    m = sid == seg
    order = np.argsort(ss[m]); p = xyz[m][order]; a = ss[m][order]
    return np.array([np.interp(s, a, p[:, i]) for i in range(3)])

def _tangent(xyz, sid, ss, seg, s):
    m = sid == seg; hi = float(ss[m].max())
    return _sample(xyz, sid, ss, seg, min(hi, s + .5)) - _sample(xyz, sid, ss, seg, max(0., s - .5))

def _inside(poly, point, center, normal):
    e1, e2 = frame(normal)
    p = np.column_stack(((poly - center) @ e1, (poly - center) @ e2))
    q = np.asarray(point) - center
    return contains_origin(p - np.array([(q @ e1), (q @ e2)]))

def _record(slicer, center, normal, daughter_points, detail=False):
    daughter_points = np.asarray(daughter_points, float).reshape(-1, 3)
    normal = np.asarray(normal, float)
    if (np.linalg.norm(normal) < 1e-9 or not np.isfinite(daughter_points).all()
            or (len(daughter_points) and np.max(np.abs((daughter_points-center) @ unit(normal))) > 1e-4)):
        return {'state': 'ambiguous', 'reason': 'invalid_or_non_coplanar_probes', 'center_mm': np.asarray(center),
                'normal': unit(normal), 'daughter_points_mm': np.empty((0,3)), 'daughter_contour_ids': [],
                'selected_contours_xyz_mm': []} if detail else {
                    'state':'ambiguous','reason':'invalid_or_non_coplanar_probes','center_mm':np.asarray(center),
                    'normal':unit(normal),'daughter_points_mm':np.empty((0,3)),'daughter_contour_ids':[]}
    loops = slicer.contours(center, normal)
    owners = [[i for i, loop in enumerate(loops) if _inside(loop["polygon_xyz_mm"], p, center, normal)]
              for p in daughter_points]
    distinct = len(owners) == 2 and len(owners[0]) == 1 and len(owners[1]) == 1 and owners[0][0] != owners[1][0]
    shared = len(owners) == 2 and len(owners[0]) == 1 and owners[0] == owners[1]
    state = "two" if distinct else "one" if shared else "ambiguous"
    out = {"state": state, "center_mm": np.asarray(center), "normal": unit(normal),
           "daughter_points_mm": np.asarray(daughter_points), "contour_count": len(loops),
           "daughter_contour_ids": owners}
    if detail:
        out["selected_contours_xyz_mm"] = [loops[i]["polygon_xyz_mm"] for i in sorted(set(sum(owners, [])))]
    return out

def _scan(xyz, sid, ss, parent, children, slicer, step=.25, persistence=None, detail_radius=2.):
    """Daughter scan using exact centerline/plane intersections."""
    end = min(float(ss[sid == c].max()) for c in children) - 1.
    vals = np.arange(0., end + step*.5, step)
    planes = []; probes = []; valid = []
    for s in vals:
        pp0 = np.array([_sample(xyz,sid,ss,c,s) for c in children])
        n = sum(_tangent(xyz,sid,ss,c,s) for c in children)
        center = pp0.mean(axis=0)
        # The daughter points must be actual intersections with this same
        # plane. A near-s window prevents a remote loop of a folded branch
        # being silently selected; coplanar intervals are ambiguous.
        pp=[]; ok=True
        for c in children:
            m = sid == c; order=np.argsort(ss[m]); path=xyz[m][order]; aa=ss[m][order]
            groups=polyline_plane_intersection_groups(path,aa,center,n)
            local=[g for g in groups if g["s_min_mm"] <= s+10 and g["s_max_mm"] >= s-10]
            if len(local)!=1 or local[0]["coplanar_interval"] or local[0]['s_max_mm']-local[0]['s_min_mm'] > 1e-4:
                ok=False; break
            pp.append(np.asarray(local[0]["points_xyz_mm"][0]))
        planes.append((s,center,n)); probes.append(np.asarray(pp) if ok else np.empty((0,3))); valid.append(ok)
    if persistence is None: persistence=max(2,int(np.ceil(1.0/step))+1)
    records=[]
    for (s, center, normal), pp, ok in zip(planes, probes, valid):
        r = _record(slicer, center, normal, pp, False) if ok else {"state":"ambiguous","center_mm":center,"normal":unit(normal),"daughter_points_mm":pp,"contour_count":None,"daughter_contour_ids":[]}
        r["s_mm"] = float(s); records.append(r)
    found = next((i for i in range(len(records)-persistence+1)
                  if records[i]["state"] == "two" and all(r["state"] == "two" for r in records[i:i+persistence])), None)
    if found is None:
        for i in np.unique(np.linspace(0, len(records)-1, min(17,len(records))).astype(int)):
            s,c,n = planes[i]
            if valid[i]: records[i] = {'s_mm':float(s), **_record(slicer,c,n,probes[i],True)}
        return {"mode":"daughter","step_mm":step,"persistence_samples":persistence,"records":records,
                "candidate":None,"status":"no_persistent_two_contour_transition"}
    # Require a preceding one-contour observation; otherwise retain as
    # unconfirmed (the scan may begin after the anatomical transition).
    prior = next((i for i in range(found-1,-1,-1) if records[i]["state"] == "one"), None)
    status = "candidate" if prior is not None else "unconfirmed_no_preceding_one"
    idx = found
    for i in range(max(0,found- int(detail_radius/step)), min(len(records),found+int(detail_radius/step)+1)):
        s, c, n = planes[i]
        records[i] = {"s_mm":float(s), **(_record(slicer,c,n,probes[i],True) if valid[i] else {"state":"ambiguous","center_mm":c,"normal":unit(n),"daughter_points_mm":probes[i],"contour_count":None,"daughter_contour_ids":[]})}
    return {"mode":"daughter","step_mm":step,"persistence_samples":persistence,"records":records,
            "candidate":None if prior is None else {"s_mm":records[idx]["s_mm"],"center_mm":records[idx]["center_mm"],
                         "bracket_mm":[records[prior]["s_mm"],records[idx]["s_mm"]],
                         "status":status,"confidence":"medium" if records[idx]['s_mm']-records[prior]['s_mm'] <= 1 else "low"}, "status":status}

def locate_case(case_dir: Path, output: Path|None=None, step=.25):
    started = time.monotonic()
    if not np.isfinite(step) or step <= 0: raise ValueError('step must be finite and positive')
    report=json.loads((case_dir/"report.json").read_text()); z=np.load(case_dir/"geometry.npz",allow_pickle=False)
    xyz=z["centerline_xyz_mm"].astype(float); sid=z["centerline_segment_id"].astype(int); ss=z["centerline_s_local_mm"].astype(float)
    wall=z["wall_xyz_mm"].astype(float); tri=z["wall_triangles"].astype(np.int64); slicer=SurfaceSlicer(wall,tri)
    children={int(s["segment_id"]):[int(c) for c in s.get("children",[])] for s in report["segments"]}
    junctions=[]
    for parent, ch in children.items():
        if len(ch)!=2: continue
        rows=np.flatnonzero(sid==parent); row=rows[np.argmax(ss[rows])]
        node=xyz[row]; scans=[_scan(xyz,sid,ss,parent,ch,slicer,step=step)]
        junctions.append({"parent_segment":parent,"daughter_segments":ch,"graph_node_mm":node,"scans":scans})
    out={"schema":SCHEMA,"diagnostic_only":True,"training_allowed":False,"awaiting_user_review":True,
         "canonical_id":report.get("canonical_id"),"source_case":str(case_dir.resolve()),
         "source_report_sha256":hashlib.sha256((case_dir/"report.json").read_bytes()).hexdigest(),
         "source_geometry_sha256":hashlib.sha256((case_dir/"geometry.npz").read_bytes()).hexdigest(),"junctions":junctions,
         "algorithm_code":{p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('section_features.py')]},
         "seconds":time.monotonic()-started, "parameters":{"step_mm":step,"minimum_two_state_extent_mm":1.0,"pairing_window_mm":10.0},
         "definition":"local_slice_shared_to_separate_daughter_contours; plane center is not a wall carina or replacement graph node"}
    target=output or case_dir/"bifurcation_locator.json"; target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(_plain(out),ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    temporary.replace(target)
    z.close(); return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("case",type=Path); ap.add_argument("--output",type=Path)
    ap.add_argument("--step",type=float,default=.25,choices=[.25,1.0], help="scan spacing in mm")
    a=ap.parse_args(); locate_case(a.case.resolve(),a.output.resolve() if a.output else None,step=a.step)

if __name__ == "__main__": main()
