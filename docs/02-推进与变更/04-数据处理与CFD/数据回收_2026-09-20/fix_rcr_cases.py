"""2026-09-21 运行态 RCR 审计后的修复（A 层 11 例；只改 UDF 线程号 / cas 挂接 / execute-at-end 注册 / 路径，不改物理设置）。
每例原件备份 *.orig_20260921c。修复类型：
  swap   : UDF execute_at_end 里左侧两出口线程号互换（outle 写到 leftnei、outli 写到 leftwai），cas 挂接正确 → 把 UDF 的 t1/t2 换成 cas 上
           pressure_outle / pressure_outli 各自挂接的区号
  hook   : cas 某出口区没挂 pressure_out*（(p (constant . 0) (profile "" ""))）→ 在该区段内改成 (p (profile "udf" "pressure_outXX::libudf") (constant . 0))
  eae    : cas 的 (udf/execute-at-end-fcns ()) 为空 → 注册 execute_at_end::libudf
  outre  : UDF 的 outre 线程号写成了非出口区（YANG_QING_REN-1/before 5502）→ 改为 cas 上 pressure_outre 挂接的区号
  area   : UDF 入口面积常量与网格入口面积不一致（LU_FU_SHAN-0/after 364.92 vs 354.92 mm²）→ 改为网格实测
路径：2.jou read-case 与 cas 内 GNN/data/... 旧前缀 → 当前病例目录（同 fix_cfd_cases.py）。
用法：python fix_rcr_cases.py        输出 fix_rcr_cases_log.md
"""
from __future__ import annotations

import gzip
import os
import re
import shutil
from pathlib import Path

ROOT = Path("/public/newhome/cy/Digital_twin/GNN/data_new")
HERE = Path(__file__).resolve().parent
STAMP = "orig_20260921c"
PAT = re.compile(rb"/?/public/newhome/cy/Digital_twin/GNN/data/[^\"\s)]*")
ZONE = re.compile(rb"\((\d+) (pressure-outlet|velocity-inlet) ([^\s)]+) \d\)\(")
HOOK = re.compile(rb"pressure_out(le|li|re|ri)::libudf")
CONST0 = b'(p (constant . 0) (profile "" ""))'

CASES = {
    "ILO/WANG_JIN_MING-0/after":   ["swap"],
    "ILO/ZHANG_JIN_CHUN-1/after":  ["swap"],
    "ILO/ZHANG_YONG_SHENG-0/after": ["swap"],
    "ILO/ZHAO_JIAN_PING-0/after":  ["swap"],
    "ILO/ZHAO_JIAN_PING-0/before": ["swap"],
    "ILO/SHEN_CHUN_WANG-0/before": ["swap"],
    "ILO/ZHANG_YAN_SHAN-0/after":  [("hook", "li")],
    "ILO/YANG_QING_REN-1/before":  [("outre",)],
    "ILO/LU_FU_SHAN-0/after":      ["eae", ("area", "0.00035492")],   # 网格实测 354.92 mm²；UDF 里的旧常量按正则取
    "ILO/SUN_XU_XIA-1/after":      ["eae"],
    "ILO/SUN_YU_SHENG-0/after":    ["eae"],
    # 导出 1140 步起全零而运行态正常 → 挂接无误，只修路径后整例重跑
    "ILO/GUO_AI_JUN-0/after":      [],
    "ILO/GUO_AI_JUN-0/before":     [],
    "ILO/LIU_BAO_JUN-0/after":     [],   # 2.jou 用 LIU_BAO_JUN.cas.gz（2026-01，挂接齐全）；旧 LIU_BAO_JUN_1.cas.gz 不用；导出 1140 起全零
    "ILO/XUE_YOU_TANG-0/before":   [],   # ascii/ 壁面 81 帧全零（根目录 81 帧是体场格式，已归入 volume_frames_root_recalc/）
}


def backup(path: Path) -> None:
    dst = path.with_name(path.name + "." + STAMP)
    if not dst.exists():
        shutil.copy2(path, dst)


def zones_and_hooks(data: bytes) -> tuple[list, dict]:
    zones = [(m.start(), int(m.group(1)), m.group(2).decode(), m.group(3).decode()) for m in ZONE.finditer(data)]
    hooks = {}
    for i, (pos, zid, zt, name) in enumerate(zones):
        end = zones[i + 1][0] if i + 1 < len(zones) else pos + 20000
        h = HOOK.search(data[pos:end])
        hooks[zid] = (name, h.group(1).decode() if h else None, pos, end)
    return zones, hooks


def fix_jou(d: Path) -> tuple[str, str]:
    jou = d / "2.jou"; raw = jou.read_bytes()
    m = re.search(rb"(/file/read-case\s+)(\S+)", raw); assert m, d
    base = os.path.basename(m.group(2).decode()); cas = d / base
    assert cas.exists(), (d, base)
    new_path = f"{d}/{base}".encode(); new = raw.replace(m.group(2), new_path, 1)
    note = "read-case 已一致" if m.group(2) == new_path else f"read-case → 当前目录/{base}"
    if new != raw:
        backup(jou); jou.write_bytes(new)
    return base, note


def main() -> None:
    rows = ["| 病例 | 2.jou | cas | UDF |", "|---|---|---|---|"]
    for cid, ops in CASES.items():
        d = ROOT / cid
        base, jnote = fix_jou(d)
        cas = d / base; data = gzip.open(cas, "rb").read(); new = data; cnotes = []
        # ---- 旧路径
        found = sorted(set(PAT.findall(data)))
        if found:
            norm = [s.decode().replace("//", "/", 1) if s.startswith(b"//") else s.decode() for s in found]
            prefix = os.path.commonprefix(norm); old_dir = prefix[: prefix.rfind("/") + 1]
            rems = [n[len(old_dir):] for n in norm]; assert all(r and not r.startswith("/") for r in rems), (cid, old_dir, rems)
            for s, n in zip(found, norm):
                new = new.replace(s, (str(d) + "/" + n[len(old_dir):]).encode())
            assert not PAT.findall(new)
            cnotes.append(f"旧前缀 `{old_dir.rstrip('/')}` → 当前目录（{', '.join(sorted(set(rems)))}）")
        zones, hooks = zones_and_hooks(new)
        udf = next(p for p in (d / "udf-inlet4.c", d / "udf-inlet.c") if p.exists()); utxt = udf.read_bytes(); unew = utxt; unotes = []
        for op in ops:
            kind = op if isinstance(op, str) else op[0]
            if kind == "swap":
                zle = next(z for z, (nm, fn, *_) in hooks.items() if fn == "le"); zli = next(z for z, (nm, fn, *_) in hooks.items() if fn == "li")
                cur = {m.group(2).decode(): int(m.group(1)) for m in re.finditer(rb"Lookup_Thread\(d,\s*(\d+)\);\s*/\*\s*(\w+)", unew)}
                if cur["outle"] == zle and cur["outli"] == zli:
                    unotes.append("左侧线程号已一致（此前已修）"); continue
                assert cur["outle"] == zli and cur["outli"] == zle, (cid, cur, zle, zli)
                unew = re.sub(rb"(t1 = Lookup_Thread\(d, )\d+(\); /\*\s*outle)", lambda m: m.group(1) + str(zle).encode() + m.group(2), unew, 1)
                unew = re.sub(rb"(t2 = Lookup_Thread\(d, )\d+(\); /\*\s*outli)", lambda m: m.group(1) + str(zli).encode() + m.group(2), unew, 1)
                unotes.append(f"outle {cur['outle']}→{zle}（{hooks[zle][0]}），outli {cur['outli']}→{zli}（{hooks[zli][0]}）")
            elif kind == "hook":
                fn = op[1]
                if any(f_ == fn for _, f_, *_ in hooks.values()):
                    cnotes.append(f"pressure_out{fn} 已挂（此前已修）"); continue
                zid = next(z for z, (nm, f_, *_) in hooks.items() if f_ is None and (("nei" in nm) if fn in ("li", "ri") else ("wai" in nm)) and (("left" in nm) if fn in ("le", "li") else ("right" in nm)))
                nm, _, pos, end = hooks[zid]; seg = new[pos:end]
                assert seg.count(CONST0) == 1, (cid, zid, seg.count(CONST0))
                seg2 = seg.replace(CONST0, b'(p (profile "udf" "pressure_out' + fn.encode() + b'::libudf") (constant . 0))')
                new = new[:pos] + seg2 + new[end:]
                cnotes.append(f"区 {zid}（{nm}）挂 pressure_out{fn}::libudf")
            elif kind == "eae":
                old = b"(udf/execute-at-end-fcns ())"
                if b'(udf/execute-at-end-fcns ("execute_at_end::libudf"))' in new:
                    cnotes.append("execute_at_end 已注册（此前已修）"); continue
                assert new.count(old) == 1, (cid, new.count(old))
                new = new.replace(old, b'(udf/execute-at-end-fcns ("execute_at_end::libudf"))'); cnotes.append("注册 execute_at_end::libudf")
            elif kind == "outre":
                zre = next(z for z, (nm, fn, *_) in hooks.items() if fn == "re")
                m = re.search(rb"(t4 = Lookup_Thread\(d, )(\d+)(\); /\*\s*outre)", unew); assert m, cid
                if int(m.group(2)) == zre:
                    unotes.append("outre 线程号已一致（此前已修）"); continue
                assert int(m.group(2)) not in hooks, (cid, m)
                unew = unew[: m.start()] + m.group(1) + str(zre).encode() + m.group(3) + unew[m.end():]; unotes.append(f"outre {m.group(2).decode()}→{zre}（{hooks[zre][0]}）")
            elif kind == "area":
                m = re.search(rb"(b8 \* sin\(8 \* tt \* w\)\)\s*/\s*)([0-9.eE+-]+)", unew); assert m, cid
                old, rep = m.group(2), op[1].encode()
                if old == rep:
                    unotes.append("入口面积常量已一致")
                else:
                    assert unew.count(old) == 1, (cid, unew.count(old)); unew = unew.replace(old, rep); unotes.append(f"入口面积常量 {old.decode()}→{op[1]}")
        # ---- 终检：UDF 四个线程号都对应 cas 上同名函数挂接的区
        zones, hooks = zones_and_hooks(new)
        ids = {m.group(2).decode(): int(m.group(1)) for m in re.finditer(rb"Lookup_Thread\(d,\s*(\d+)\);\s*/\*\s*(\w+)", unew)}
        for k, z in ids.items():
            assert z in hooks and hooks[z][1] == k[3:], (cid, k, z, hooks.get(z))
        assert b'(udf/execute-at-end-fcns ("execute_at_end::libudf"))' in new, cid
        if unew != utxt:
            backup(udf); udf.write_bytes(unew)
        if new != data:
            backup(cas); tmp = d / (base + ".tmp_20260921c")
            with gzip.open(tmp, "wb", compresslevel=6) as f:
                f.write(new)
            assert gzip.open(tmp, "rb").read() == new; os.replace(tmp, cas)
        rows.append(f"| `{cid}` | {jnote} | {'；'.join(cnotes) or '无改动'} | {'；'.join(unotes) or '无改动'} |")
        print(rows[-1], flush=True)
    (HERE / "fix_rcr_cases_log.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
