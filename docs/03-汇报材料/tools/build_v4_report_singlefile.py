#!/usr/bin/env python3
"""把 V4 汇报网页（图件相对引用 V4汇报/）打包成单文件 HTML（图片 base64 内嵌）。

用法（在仓库任意位置）：
    python3 docs/03-汇报材料/tools/build_v4_report_singlefile.py
    python3 docs/03-汇报材料/tools/build_v4_report_singlefile.py --src <本地版.html> --dst <单文件版.html>

约定：
- **只改本地版**（`WSS_PINN_V4实验分析与汇报提纲_2026-09-02.html`，可直接用编辑器改中文），
  改完重新运行本脚本生成单文件版；不要手改单文件版——它后半段是图片的 base64 编码，
  看起来像乱码是正常的，不是编码错误。
- `<img src="...">` 的相对路径按本地版所在目录解析；`<a href="同一图片">原图</a>` 会改成
  `download="文件名"` 的 data URI，拷到别的电脑仍可另存原图。
- 页眉里的“本地版 / 图件引用 V4汇报/”提示会被替换成“单文件版”。
"""
from __future__ import annotations

import argparse
import base64
import mimetypes
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_SRC = HERE.parent / "WSS_PINN_V4实验分析与汇报提纲_2026-09-02.html"
DEFAULT_DST = HERE.parent / "WSS_PINN_V4实验分析与汇报提纲_2026-09-02_单文件版.html"

IMG_RE = re.compile(r'<img\s+src="([^"]+)"')
A_RE = re.compile(r'<a\s+href="([^"]+\.(?:png|jpg|jpeg|svg|pdf))">([^<]*)</a>')


def data_uri(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or "application/octet-stream"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def build(src: Path, dst: Path) -> dict[str, int]:
    html = src.read_text(encoding="utf-8")
    base = src.parent
    cache: dict[str, str] = {}
    missing: list[str] = []

    def uri_for(rel: str) -> str | None:
        if rel.startswith("data:") or rel.startswith("http"):
            return None
        if rel not in cache:
            p = (base / rel).resolve()
            if not p.is_file():
                missing.append(rel)
                return None
            cache[rel] = data_uri(p)
        return cache[rel]

    def img_sub(m: re.Match) -> str:
        uri = uri_for(m.group(1))
        return m.group(0) if uri is None else f'<img src="{uri}"'

    def a_sub(m: re.Match) -> str:
        rel, text = m.group(1), m.group(2)
        uri = uri_for(rel)
        if uri is None:
            return m.group(0)
        name = Path(rel).name
        label = text if text.startswith("下载") else f"下载{text}"
        return f'<a href="{uri}" download="{name}">{label}</a>'

    n_img = len(IMG_RE.findall(html))
    html = IMG_RE.sub(img_sub, html)
    html = A_RE.sub(a_sub, html)
    html = html.replace("· 实验分析</div>", "· 实验分析（单文件版，图件已内嵌）</div>", 1)
    html = html.replace(
        "本页为本地版，图件相对引用 <code>V4汇报/</code>。",
        "本页为单文件版，图件已 base64 内嵌；文字修改请改本地版后用 <code>tools/build_v4_report_singlefile.py</code> 重建。",
    )
    if missing:
        raise FileNotFoundError("missing figure files: " + ", ".join(sorted(set(missing))))
    dst.write_text(html, encoding="utf-8")
    return {"images": n_img, "unique_files": len(cache), "bytes": dst.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST)
    args = parser.parse_args()
    info = build(args.src.resolve(), args.dst.resolve())
    print(f"wrote {args.dst}  images={info['images']} unique_files={info['unique_files']} size={info['bytes']/1e6:.1f} MB")


if __name__ == "__main__":
    main()
