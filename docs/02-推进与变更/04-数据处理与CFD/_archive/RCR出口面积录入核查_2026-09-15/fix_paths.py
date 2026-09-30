"""把 26 个重跑病例的 2.jou（read-case 路径）和 .cas.gz 里的 Linux 绝对路径统一改成当前病例目录。
.cas 里的路径是 scheme 文本字符串（ascii 导出前缀、autosave 根名、UDF 源文件），按字节替换旧目录前缀，二进制网格段不含这些字符串。
备份：2.jou.orig_20260915、<case>.cas.gz.orig_20260915（只在有改动时生成）。"""
import gzip, os, re, shutil, sys
from pathlib import Path
ROOT = Path('/public/newhome/cy/Digital_twin/GNN'); HERE = Path(__file__).resolve().parent
PAT = re.compile(rb'/?/public/newhome/cy/Digital_twin/GNN/[^"\s)]*')
cases = [l.strip() for l in open(HERE / 'rerun_cases.txt') if l.strip()]
report = ['| 病例目录 | 2.jou | .cas.gz 旧目录前缀 → 新 | 替换串数 |', '|---|---|---|---|']
for c in cases:
    d = ROOT / c; new_dir = str(d.resolve())
    # ---- 2.jou
    jou = d / '2.jou'; raw = jou.read_bytes(); m = re.search(rb'(/file/read-case\s+)(\S+)', raw); assert m, c
    old_path = m.group(2).decode(); base = os.path.basename(old_path); assert (d / base).exists(), (c, base)
    new_path = f'{new_dir}/{base}'; jou_note = '已一致'
    if old_path != new_path:
        if not (d / '2.jou.orig_20260915').exists(): shutil.copy2(jou, d / '2.jou.orig_20260915')
        jou.write_bytes(raw.replace(m.group(2), new_path.encode(), 1)); jou_note = f'{old_path} → 当前目录/{base}'
    # ---- .cas.gz
    cas = d / base; data = gzip.open(cas, 'rb').read(); found = sorted(set(PAT.findall(data)))
    norm = [s.decode().replace('//', '/', 1) if s.startswith(b'//') else s.decode() for s in found]
    if not found: report.append(f'| `{c}` | {jou_note} | .cas 无 Linux 绝对路径 | 0 |'); continue
    prefix = os.path.commonprefix(norm); old_dir = prefix[:prefix.rfind('/') + 1]
    rems = [n[len(old_dir):] for n in norm]
    if old_dir.rstrip('/') == new_dir: report.append(f'| `{c}` | {jou_note} | 已一致（{", ".join(rems)}） | 0 |'); continue
    assert all(r and not r.startswith('/') for r in rems), (c, old_dir, rems)
    mapping = {s: (new_dir + '/' + n[len(old_dir):]).encode() for s, n in zip(found, norm)}
    total = 0; new = data
    for s, t in mapping.items(): total += new.count(s); new = new.replace(s, t)
    left = [x for x in set(PAT.findall(new)) if not x.decode().startswith(new_dir)]; assert not left, (c, left)
    tmp = d / (base + '.tmp_20260915')
    with gzip.open(tmp, 'wb', compresslevel=6) as f: f.write(new)
    assert gzip.open(tmp, 'rb').read() == new
    if not (d / (base + '.orig_20260915')).exists(): os.rename(cas, d / (base + '.orig_20260915'))
    os.rename(tmp, cas)
    report.append(f'| `{c}` | {jou_note} | `{old_dir.rstrip("/")}` → 当前目录（{", ".join(sorted(set(rems)))}） | {total} |')
(HERE / 'fix_paths_log.md').write_text('\n'.join(report) + '\n')
for l in report: print(l)
