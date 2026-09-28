"""Solver-settings comparison between two Fluent case files (text sections only).

Parses the Scheme text of sections 37 (rp variables), 38 (domain variables) and 39/45 (zone settings, keyed by zone
name, zone id dropped) and reports every differing entry, split into *expected* (mesh/partition/meshing bookkeeping,
paths inside the work directory) and *unexpected* differences. The unexpected list must be empty before a case runs.
"""
from __future__ import annotations

import gzip
import re
from pathlib import Path

BINARY_RE = re.compile(rb"(?m)^\((?:20|30)\d\d \(")  # every 20xx/30xx section (at line start) is binary; values inside rp text can look like '(2040 ('
TOKEN_RE = re.compile(r'"(?:\\.|[^"\\])*"|[()]|[^\s()"]+')
# rp-variable prefixes that are mesh / partition / meshing-session bookkeeping, not solver settings
EXPECTED_PREFIXES = ("partition/", "parallel/", "mesh/", "prism/", "hexcore/", "rapid-octree/", "cartesian/", "size-func/", "scoped-", "objects",
                     "backup-info", "cfd-post-mesh-info", "tgrid/", "domain-", "poly/", "wb/", "fluent-case-version", "geometry/", "file", "extract-feature",
                     "normalized-sphere", "sliver-", "node-tolerance", "hole/", "max-cell-size", "skewness-method", "solution-zones", "size-controls",
                     "prism-controls", "materials", "export/automatic", "gpuapp/", "report/", "adapt/", "dynamesh/", "nonconformal/", "case-", "cx-", "graphics/",
                     "geom-thread-id/")          # CAD-import face-zone naming bookkeeping (e.g. 'wall:--.5')


# zone fields that replace-mesh resets without effect on this protocol (radiation is off: every rp variable equal)
ZONE_FIELDS_EXPECTED = {"parallel-collimated-beam?": "radiation beam flag; radiation model off in the library protocol",
                        # inactive-model BC lists that replace-mesh writes with v231 defaults (library QIN_SI_FU itself carries these values)
                        "pb-disc-bc": "population-balance BC list; model off", "pb-dqmom-bc": "population-balance BC list; model off",
                        "pb-qmom-bc": "population-balance BC list; model off", "pb-smm-bc": "population-balance BC list; model off",
                        "tss-scalar": "pollutant-model scalar (listed among pollut_*); model off"}
SOLUTION_ZONES_RE = re.compile(r"\(solution-zones ([0-9 ]+)\)")


def _norm(v: str | None) -> str | None:
    """Order-free zone-id lists (replace-mesh keeps the ids but lists them in the new mesh's order)."""
    return None if v is None else SOLUTION_ZONES_RE.sub(lambda m: "(solution-zones " + " ".join(sorted(m.group(1).split())) + ")", v)


def _fields(settings: str) -> dict:
    x = parse(settings)[0] if parse(settings) else []
    x = x[0] if x and isinstance(x[0], list) and x[0] and isinstance(x[0][0], list) else x
    return {f[0]: dump(f[1:]) for f in x if isinstance(f, list) and f and isinstance(f[0], str)}


def case_text(path: str | Path) -> str:
    p = Path(path)
    data = gzip.open(p, "rb").read() if p.suffix == ".gz" else p.read_bytes()
    out, pos = [], 0
    for m in BINARY_RE.finditer(data):
        if m.start() < pos:
            continue
        out.append(data[pos:m.start()])
        end = data.find(b"End of Binary Section", m.start())
        pos = data.find(b")", end) + 1 if end > 0 else len(data)  # marker line is 'End of Binary Section   3012)'
    out.append(data[pos:])
    return b"".join(out).decode("latin1")


def parse(text: str) -> list:
    """Top-level s-expressions (lists of tokens / nested lists)."""
    stack, top = [], []
    for tok in TOKEN_RE.findall(text):
        if tok == "(":
            stack.append([])
        elif tok == ")":
            if not stack:
                continue
            done = stack.pop()
            (stack[-1] if stack else top).append(done)
        elif stack:
            stack[-1].append(tok)
    return top


def dump(x) -> str:
    return "(" + " ".join(dump(y) for y in x) + ")" if isinstance(x, list) else x


def sections(text: str) -> dict:
    rp, dom, zones = {}, {}, {}
    for sx in parse(text):
        if not sx or not isinstance(sx[0], str):
            continue
        tag = sx[0]
        if tag in ("37", "38") and len(sx) > 1 and isinstance(sx[1], list):
            target = rp if tag == "37" else dom
            for item in sx[1]:
                if isinstance(item, list) and item and isinstance(item[0], str):
                    target[item[0]] = dump(item[1:])
        elif tag in ("39", "45") and len(sx) > 1 and isinstance(sx[1], list) and len(sx[1]) >= 3:
            _, ztype, name = sx[1][:3]
            zones[name] = {"type": ztype, "settings": dump(sx[2:])}
    return {"rp": rp, "domain": dom, "zones": zones}


def _materials(rp: dict) -> dict:
    raw = rp.get("materials")
    if raw is None:
        return {}
    mats = parse(raw)[0] if parse(raw) else []
    mats = mats[0] if mats and isinstance(mats[0], list) and mats[0] and isinstance(mats[0][0], list) else mats
    return {m[0]: dump(m) for m in mats if isinstance(m, list) and m and isinstance(m[0], str)}


def _exports(rp: dict) -> dict:
    raw = rp.get("export/automatic")
    if raw is None:
        return {}
    out = {}
    for e in (parse(raw)[0] if parse(raw) else []):
        for ex in (e if isinstance(e, list) and e and isinstance(e[0], list) else [e]):
            if isinstance(ex, list) and ex and isinstance(ex[0], str):
                fields = {f[0]: dump(f[1:]) for f in ex[1:] if isinstance(f, list) and f and isinstance(f[0], str) and not f[0].endswith("last-iter-no")}
                out[ex[0]] = fields
    return out


def compare(ref_case: str | Path, new_case: str | Path, workdir: str | Path, library_dir: str | Path) -> dict:
    a, b = sections(case_text(ref_case)), sections(case_text(new_case))
    unexpected, expected = [], []
    for key in sorted(set(a["rp"]) | set(b["rp"])):
        va, vb = _norm(a["rp"].get(key)), _norm(b["rp"].get(key))
        if va == vb:
            continue
        (expected if key.startswith(EXPECTED_PREFIXES) else unexpected).append({"section": "rp", "key": key, "ref": (va or "<absent>")[:300], "new": (vb or "<absent>")[:300]})
    for key in sorted(set(a["domain"]) | set(b["domain"])):
        va, vb = a["domain"].get(key), b["domain"].get(key)
        if va != vb:
            unexpected.append({"section": "domain", "key": key, "ref": (va or "<absent>")[:300], "new": (vb or "<absent>")[:300]})
    za, zb = a["zones"], b["zones"]
    named = lambda z: {k: v for k, v in z.items() if not re.match(r"interior-\d+$", k)}
    for name in sorted(set(named(za)) | set(named(zb))):
        va, vb = za.get(name), zb.get(name)
        if va is None or vb is None or va["type"] != vb["type"]:
            unexpected.append({"section": "zone", "key": name, "ref": str(va)[:300], "new": str(vb)[:300]})
            continue
        fa, fb = _fields(va["settings"]), _fields(vb["settings"])
        for f in sorted(set(fa) | set(fb)):
            if fa.get(f) != fb.get(f):
                row = {"section": f"zone {name}", "key": f, "ref": str(fa.get(f))[:300], "new": str(fb.get(f))[:300]}
                if f in ZONE_FIELDS_EXPECTED:
                    expected.append({**row, "reason": ZONE_FIELDS_EXPECTED[f]})
                else:
                    unexpected.append(row)
    ma, mb = _materials(a["rp"]), _materials(b["rp"])
    for name in sorted(set(ma) | set(mb)):
        if ma.get(name) != mb.get(name):
            unexpected.append({"section": "material", "key": name, "ref": (ma.get(name) or "<absent>")[:300], "new": (mb.get(name) or "<absent>")[:300]})
    ea, eb = _exports(a["rp"]), _exports(b["rp"])
    for name in sorted(set(ea) | set(eb)):
        fa, fb = ea.get(name, {}), eb.get(name, {})
        for f in sorted(set(fa) | set(fb)):
            if fa.get(f) != fb.get(f):
                unexpected.append({"section": f"export {name}", "key": f, "ref": fa.get(f, "<absent>")[:300], "new": fb.get(f, "<absent>")[:300]})
    return {"unexpected": unexpected, "expected": expected, "n_rp": [len(a["rp"]), len(b["rp"])], "n_zones": [len(za), len(zb)],
            "materials": sorted(mb), "exports": {k: {f: v for f, v in e.items() if f in ("file-name", "surfaces", "cellzones", "quantities", "frequency", "type")} for k, e in eb.items()}}
