"""2026-09-21 数据回收：把 6 个"需重跑 CFD"病例的 Fluent 输入修到能直接 sbatch 的状态（只改路径/导出变量表/入口挂接/网格缩放，不改物理设置）。

每例做的事（全部可逆，原件备份 *.orig_20260921）：
  - 2.jou：read-case 指向当前病例目录里的 .cas.gz；PENG_JI_MING 额外在 read-case 后插一行 /mesh/scale 1000 1000 1000（网格比 STL 小 1000 倍）
  - .cas.gz（scheme 文本段按字节替换，二进制网格段不含这些串）：
      * 旧目录前缀 GNN/data/... → 当前病例目录（导出前缀 export-1/-2、UDF 源文件）
      * AG 三例 export-2 只导 "Static Pressure" → 补齐 wall-shear 四列（与正常 AG 例逐字相同）
      * WANG_CAI 入口 vmag 常量 0 → 挂 my_inlet::libudf（与其他 ILO 例逐字相同）
  - 回读校验：替换后 gzip 解压逐字节等于替换结果；旧前缀串数为 0；目标串出现次数为 1

用法： python fix_cfd_cases.py            （在仓库根目录任意位置运行）
输出： 本目录 fix_cfd_cases_log.md
"""
from __future__ import annotations

import gzip
import os
import re
import shutil
from pathlib import Path

ROOT = Path("/public/newhome/cy/Digital_twin/GNN")
HERE = Path(__file__).resolve().parent
STAMP = "orig_20260921"
PAT = re.compile(rb"/?/public/newhome/cy/Digital_twin/GNN/data/[^\"\s)]*")

OLD_EXPORT2_AG = b'(surfaces wall) (cellzones blood) (quantities "Static Pressure") (vector-quantities)'
NEW_EXPORT2_AG = (b'(surfaces wall) (cellzones blood) (quantities "Static Pressure" "Wall Shear Stress" '
                  b'"X-Wall Shear Stress" "Y-Wall Shear Stress" "Z-Wall Shear Stress") (vector-quantities)')
OLD_VMAG = b'(vmag (constant . 0) (profile "" ""))'
NEW_VMAG = b'(vmag (profile "udf" "my_inlet::libudf") (constant . 0))'

CASES = {
    "AAA/ruputer/LV_GUO_YOU":   dict(extra=[], scale=False),
    "AG/slow/LIU_XI_QUAN":      dict(extra=[(OLD_EXPORT2_AG, NEW_EXPORT2_AG)], scale=False),
    "AG/slow/WEI_JUN_WEN":      dict(extra=[(OLD_EXPORT2_AG, NEW_EXPORT2_AG)], scale=False),
    "AG/slow/TE_JIN_WANG":      dict(extra=[(OLD_EXPORT2_AG, NEW_EXPORT2_AG)], scale=False),
    "ILO/WANG_CAI-0/before":    dict(extra=[(OLD_VMAG, NEW_VMAG)], scale=False),
    "AG/fast/PENG_JI_MING":     dict(extra=[], scale=True),
}


def backup(path: Path) -> None:
    dst = path.with_name(path.name + "." + STAMP)
    if not dst.exists():
        shutil.copy2(path, dst)


def fix_jou(d: Path, scale: bool) -> str:
    jou = d / "2.jou"
    raw = jou.read_bytes()
    m = re.search(rb"(/file/read-case\s+)(\S+)", raw)
    assert m, d
    base = os.path.basename(m.group(2).decode())
    cas = d / base
    assert cas.exists(), (d, base)
    new_path = f"{d}/{base}".encode()
    new = raw.replace(m.group(2), new_path, 1)
    notes = ["read-case 已一致" if m.group(2) == new_path else f"read-case {m.group(2).decode()} → 当前目录/{base}"]
    if scale and b"/mesh/scale" not in new:
        new = new.replace(b"/solve/initialize/initialize-flow", b"/mesh/scale 1000 1000 1000\n/solve/initialize/initialize-flow", 1)
        assert b"/mesh/scale 1000 1000 1000" in new
        notes.append("插入 /mesh/scale 1000 1000 1000")
    if new != raw:
        backup(jou)
        jou.write_bytes(new)
    return base, "；".join(notes)


def fix_cas(d: Path, base: str, extra: list[tuple[bytes, bytes]]) -> str:
    cas = d / base
    data = gzip.open(cas, "rb").read()
    new = data
    notes = []
    found = sorted(set(PAT.findall(data)))
    if found:
        norm = [s.decode().replace("//", "/", 1) if s.startswith(b"//") else s.decode() for s in found]
        prefix = os.path.commonprefix(norm)
        old_dir = prefix[: prefix.rfind("/") + 1]
        rems = [n[len(old_dir):] for n in norm]
        assert all(r and not r.startswith("/") for r in rems), (d, old_dir, rems)
        total = 0
        for s, n in zip(found, norm):
            t = (str(d) + "/" + n[len(old_dir):]).encode()
            total += new.count(s)
            new = new.replace(s, t)
        notes.append(f"旧前缀 `{old_dir.rstrip('/')}` → 当前目录（{', '.join(sorted(set(rems)))}，{total} 串）")
    else:
        notes.append("无旧路径串")
    for old, rep in extra:
        n_old = new.count(old)
        assert n_old == 1, (d, old[:40], n_old)
        assert new.count(rep) == 0, (d, rep[:40])
        new = new.replace(old, rep)
        notes.append(f"`{old[:38].decode()}…` → `{rep[:38].decode()}…`")
    left = [x for x in set(PAT.findall(new))]
    assert not left, (d, left)
    for old, rep in extra:
        assert new.count(rep) == 1 and new.count(old) == 0
    if new != data:
        backup(cas)
        tmp = d / (base + ".tmp_20260921")
        with gzip.open(tmp, "wb", compresslevel=6) as f:
            f.write(new)
        assert gzip.open(tmp, "rb").read() == new
        os.replace(tmp, cas)
    else:
        notes.append("（.cas 无改动）")
    return "；".join(notes)


def main() -> None:
    rows = ["| 病例 | 2.jou | .cas.gz |", "|---|---|---|"]
    for cid, spec in CASES.items():
        d = ROOT / "data_new" / cid
        base, jnote = fix_jou(d, spec["scale"])
        cnote = fix_cas(d, base, spec["extra"])
        rows.append(f"| `{cid}` | {jnote} | {cnote} |")
        print(cid, "|", jnote, "|", cnote, flush=True)
    (HERE / "fix_cfd_cases_log.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(HERE / "fix_cfd_cases_log.md")


if __name__ == "__main__":
    main()
