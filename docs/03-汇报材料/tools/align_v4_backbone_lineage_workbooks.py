#!/usr/bin/env python3
"""Patch workbook wording in-place without round-tripping drawings/charts.

The two reporting workbooks contain embedded drawings that openpyxl drops on save.
This utility therefore replaces only selected worksheet cell XML nodes inside the
OOXML zip and copies every other archive member byte-for-byte.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import os
import re
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Tuple, Union


ROOT = Path(__file__).resolve().parents[3]

PINN_WORKBOOK = ROOT / "docs/03-汇报材料/WSS_PINN_V1_V2_V3_field-v4实验矩阵与指标汇总_2026-08-06.xlsx"
POINT_WORKBOOK = ROOT / "docs/03-汇报材料/WSS_PointNet实验矩阵与结果汇总last.xlsx"


PINN_NOTE = (
    "行5–12为体域PINN V2：PointNet继承P2V骨干锚点，PointNet++继承纯D2 "
    "c125-k128骨干锚点；只继承结构/support概念，不加载旧WSS权重。行19–34为"
    "pre-Centerline-V2 BC/RCR V4 seed1234 official test35历史screen：场指标来自全部"
    "评估宇宙（瞬态=eligible×81帧），WSS为冻结Profile-Secant V3的full-wall下游"
    "审计。该旧V4使用约250k容量配平简化PN/PNPP，不是P2V/D2。2026-09-04用户选择"
    "formal Centerline-V2 V4恢复P2V/D2 c125-k128，但新模型尚未训练，本表无其结果。"
    "禁止把旧V4、field-v4、V2 SAME5K和V3 val-selected数值裸比。"
)

POINT_NOTE = (
    "同一实验分组内：绿色表示相对更优，红色表示相对较弱；不同test16/test15/test27"
    "不能按颜色直接横比。★仅表示各自时间点的开发候选，不是全表统一的‘当前最优’："
    "P2V/Q1V是2026-07-18 Phase-V候选；体域PINN V1/V2后来冻结P2V与纯D2 c125-k128"
    "作为架构provenance。2026-09-04用户决定formal Centerline-V2 V4继续采用P2V/D2 "
    "c125-k128；后续LSA2/H2/log-radius星标仍只属于direct-WSS开发线。"
)


ExpectedValue = Union[str, Tuple[str, ...]]
CellPatch = Tuple[ExpectedValue, str]


PATCHES: dict[Path, dict[str, dict[str, CellPatch]]] = {
    PINN_WORKBOOK: {
        "实验矩阵汇总": {
            "A1": (
                "WSS_PINN V2 / V3 / V4 实验矩阵与核心指标汇总（2026-09-01 full-wall WSS 更新）",
                "WSS_PINN V2 / V3 / pre-Centerline-V2 BC/RCR V4 指标汇总（2026-09-04 骨干口径对齐）",
            ),
            "A3": (
                "V4 行是 seed1234 official test35 screen：场指标来自全部评估宇宙（瞬态=eligible×81 帧）；E_rel_l2 为设计 primary。WSS 为冻结 Profile-Secant V3、峰值全场速度、test35×每例全部冻结壁面节点（full-wall）。index 0–15（seed1234 16 臂）已填。单 seed，禁止与 V2 SAME5K / V3 val-selected 裸比。",
                PINN_NOTE,
            ),
            **{
                f"E{row}": (
                    ("PointNet", "PointNet（P2V anchor + continuous query）"),
                    "PN · P2V anchor",
                )
                for row in range(5, 9)
            },
            **{
                f"E{row}": (
                    ("PointNet++", "PointNet++（D2 c125-k128 anchor + continuous query）"),
                    "PN++ · D2 c125-k128",
                )
                for row in range(9, 13)
            },
            **{
                f"E{row}": (
                    ("PointNet", "PointNet（matched v1.2: 64-128-256 global）"),
                    "PN · matched 64-128-256",
                )
                for row in (*range(19, 23), *range(27, 31))
            },
            **{
                f"E{row}": (
                    ("PointNet++", "PointNet++（matched v1.2: FPS128-k32 single-stage）"),
                    "PN++ · matched FPS128-k32",
                )
                for row in (*range(23, 27), *range(31, 35))
            },
        },
        "V4 Checkpoint敏感性": {
            "A1": (
                "V4 checkpoint 敏感性（seed1234 index 0–15；主读数 last_converged）",
                "pre-Centerline-V2 BC/RCR V4 checkpoint敏感性（matched v1.2；seed1234 index 0–15；主读数last_converged）",
            ),
        },
        "V4 Full-wall WSS": {
            "A1": (
                "V4 full-wall WSS 下游审计（seed1234；仅用于替换原 1200 点 WSS 指标）",
                "pre-Centerline-V2 BC/RCR V4 full-wall WSS下游审计（matched v1.2；seed1234；非formal P2V/D2结果）",
            ),
        },
    },
    POINT_WORKBOOK: {
        "教师汇报视图": {
            "A29": (
                "同一实验分组内：绿色表示相对更优，红色表示相对较弱；不同 test16 / test15 / test27 不按颜色直接横比。★ 表示当前单次优先复核候选（P2V、Q1V），不代表发布结论。",
                POINT_NOTE,
            ),
            "B22": (
                (
                    "★ P2V：PointNet顶点采样(SEP)",
                    "★ P2V：PointNet顶点采样(SEP)｜PINN PN provenance",
                ),
                "★ P2V｜PINN PN anchor",
            ),
            "B24": (
                (
                    "★ Q1V：Point++SA3顶点采样(SAME)",
                    "★ Q1V：Point++SA3顶点采样(SAME)｜07-18阶段候选",
                ),
                "★ Q1V｜07-18 阶段候选",
            ),
            "B101": (
                ("D2 c125 × k128", "D2 c125 × k128｜PINN PNPP provenance"),
                "D2 c125×k128｜PINN PNPP anchor",
            ),
        },
        "实验矩阵总览": {
            "A17": (
                (
                    "P2V：PointNet顶点采样(SEP)",
                    "P2V：PointNet顶点采样(SEP)｜PINN PN provenance",
                ),
                "P2V｜PINN PN anchor",
            ),
            "A19": (
                (
                    "Q1V：Point++SA3顶点采样(SAME)",
                    "Q1V：Point++SA3顶点采样(SAME)｜07-18阶段候选",
                ),
                "Q1V｜07-18 阶段候选",
            ),
            "A101": (
                ("D2 c125 × k128", "D2 c125 × k128｜PINN PNPP provenance"),
                "D2 c125×k128｜PINN PNPP anchor",
            ),
        },
    },
}

ROW_HEIGHT_PATCHES: dict[Path, dict[str, dict[int, tuple[str, str]]]] = {
    POINT_WORKBOOK: {"教师汇报视图": {29: ("16.8", "44")}},
}

# These fragments repair the workbook's print contract without loading/saving it
# through openpyxl, which would discard embedded drawings in the PointNet file.
XML_PATCHES: dict[Path, dict[str, list[CellPatch]]] = {
    PINN_WORKBOOK: {
        "xl/workbook.xml": [
            (
                "'实验矩阵汇总'!$A$4:$AN$18",
                "'实验矩阵汇总'!$A$4:$AO$34",
            ),
            (
                "'实验矩阵汇总'!$A$1:$AN$18",
                "'实验矩阵汇总'!$A$1:$AO$34",
            ),
            (
                "</definedNames>",
                "<definedName name=\"_xlnm.Print_Titles\" localSheetId=\"5\">"
                "'V4 Checkpoint敏感性'!$1:$3</definedName>"
                "<definedName name=\"_xlnm.Print_Area\" localSheetId=\"5\">"
                "'V4 Checkpoint敏感性'!$A$1:$N$51</definedName>"
                "<definedName name=\"_xlnm.Print_Titles\" localSheetId=\"6\">"
                "'V4 Full-wall WSS'!$1:$3</definedName>"
                "<definedName name=\"_xlnm.Print_Area\" localSheetId=\"6\">"
                "'V4 Full-wall WSS'!$A$1:$O$11</definedName>"
                "</definedNames>",
            ),
        ],
        "xl/worksheets/sheet1.xml": [
            ('<autoFilter ref="A4:AN18"/>', '<autoFilter ref="A4:AO34"/>'),
            ('<mergeCell ref="A3:AN3"/>', '<mergeCell ref="A3:AO3"/>'),
            ('<mergeCell ref="A1:AN2"/>', '<mergeCell ref="A1:AO2"/>'),
        ],
        "xl/worksheets/sheet6.xml": [
            ('<pageSetUpPr/>', '<pageSetUpPr fitToPage="1"/>'),
            (
                '<pageMargins left="0.75" right="0.75" top="1" bottom="1" '
                'header="0.5" footer="0.5"/></worksheet>',
                '<pageMargins left="0.3" right="0.3" top="0.5" bottom="0.5" '
                'header="0.2" footer="0.2"/>'
                '<pageSetup orientation="landscape" paperSize="9" '
                'fitToHeight="0" fitToWidth="1"/></worksheet>',
            ),
        ],
        "xl/worksheets/sheet7.xml": [
            ('<pageSetUpPr/>', '<pageSetUpPr fitToPage="1"/>'),
            (
                '<pageMargins left="0.75" right="0.75" top="1" bottom="1" '
                'header="0.5" footer="0.5"/></worksheet>',
                '<pageMargins left="0.3" right="0.3" top="0.5" bottom="0.5" '
                'header="0.2" footer="0.2"/>'
                '<pageSetup orientation="landscape" paperSize="9" '
                'fitToHeight="0" fitToWidth="1"/></worksheet>',
            ),
        ],
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sheet_targets(archive: zipfile.ZipFile) -> dict[str, str]:
    main_ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    rel_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    pkg_ns = "http://schemas.openxmlformats.org/package/2006/relationships"
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    rel_targets = {
        rel.attrib["Id"]: rel.attrib["Target"]
        for rel in rels.findall(f"{{{pkg_ns}}}Relationship")
    }
    targets: dict[str, str] = {}
    for sheet in workbook.findall(f".//{{{main_ns}}}sheet"):
        target = rel_targets[sheet.attrib[f"{{{rel_ns}}}id"]].lstrip("/")
        if not target.startswith("xl/"):
            target = f"xl/{target}"
        targets[sheet.attrib["name"]] = target
    return targets


def _cell_value(xml: str, ref: str, shared_strings: list[str]) -> str:
    pattern = re.compile(
        rf'<c\b(?=[^>]*\br="{re.escape(ref)}")(?P<attrs>[^>]*?)>'
        r"(?P<body>.*?)</c>",
        re.DOTALL,
    )
    match = pattern.search(xml)
    if match is None:
        raise RuntimeError(f"cell {ref} not found")
    attrs = match.group("attrs")
    body = match.group("body")
    cell_type_match = re.search(r'\bt="([^"]+)"', attrs)
    cell_type = cell_type_match.group(1) if cell_type_match else None
    if cell_type == "inlineStr":
        texts = re.findall(r"<t(?:\s[^>]*)?>(.*?)</t>", body, re.DOTALL)
        return "".join(html.unescape(value) for value in texts)
    value_match = re.search(r"<v>(.*?)</v>", body, re.DOTALL)
    if value_match is None:
        return ""
    value = html.unescape(value_match.group(1))
    if cell_type == "s":
        return shared_strings[int(value)]
    return value


def _replace_cell(
    xml: str,
    ref: str,
    expected: ExpectedValue,
    replacement: str,
    shared_strings: list[str],
) -> str:
    actual = _cell_value(xml, ref, shared_strings)
    if actual == replacement:
        return xml
    allowed = (expected,) if isinstance(expected, str) else expected
    if actual not in allowed:
        raise RuntimeError(f"{ref}: expected one of {allowed!r}, found {actual!r}")
    pattern = re.compile(
        rf'<c\b(?=[^>]*\br="{re.escape(ref)}")(?P<attrs>[^>]*?)>'
        r"(?P<body>.*?)</c>",
        re.DOTALL,
    )
    match = pattern.search(xml)
    if match is None:
        raise RuntimeError(f"cell {ref} not found")
    attrs = re.sub(r'\s+t="[^"]*"', "", match.group("attrs"))
    escaped = html.escape(replacement, quote=False)
    node = f'<c{attrs} t="inlineStr"><is><t>{escaped}</t></is></c>'
    return xml[: match.start()] + node + xml[match.end() :]


def _replace_row_height(xml: str, row: int, expected: str, replacement: str) -> str:
    pattern = re.compile(rf'<row\b(?=[^>]*\br="{row}")(?P<attrs>[^>]*)>')
    match = pattern.search(xml)
    if match is None:
        raise RuntimeError(f"row {row} not found")
    attrs = match.group("attrs")
    height_match = re.search(r'\bht="([^"]+)"', attrs)
    actual = height_match.group(1) if height_match else ""
    if actual == replacement:
        return xml
    if actual != expected:
        raise RuntimeError(f"row {row}: expected height {expected!r}, found {actual!r}")
    attrs = re.sub(r'\bht="[^"]+"', f'ht="{replacement}"', attrs)
    node = f"<row{attrs}>"
    return xml[: match.start()] + node + xml[match.end() :]


def _replace_xml_fragment(
    xml: str,
    member: str,
    expected: ExpectedValue,
    replacement: str,
) -> str:
    if replacement in xml:
        return xml
    allowed = (expected,) if isinstance(expected, str) else expected
    matches = [(candidate, xml.count(candidate)) for candidate in allowed]
    candidate = next((value for value, count in matches if count == 1), None)
    if candidate is None:
        raise RuntimeError(
            f"{member}: expected exactly one replaceable fragment; counts={matches!r}"
        )
    return xml.replace(candidate, replacement, 1)


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    return ["".join(node.text or "" for node in item.iter(f"{{{ns}}}t")) for item in root]


def patch_workbook(path: Path, patches: dict[str, dict[str, CellPatch]], *, check: bool) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    before_sha = _sha256(path)
    with zipfile.ZipFile(path, "r") as source:
        source.testzip()
        targets = _sheet_targets(source)
        shared = _shared_strings(source)
        changed_members: dict[str, bytes] = {}
        for sheet_name, cells in patches.items():
            member = targets[sheet_name]
            xml = source.read(member).decode("utf-8")
            updated = xml
            for ref, (expected, replacement) in cells.items():
                updated = _replace_cell(updated, ref, expected, replacement, shared)
            for row, (expected, replacement) in ROW_HEIGHT_PATCHES.get(path, {}).get(
                sheet_name, {}
            ).items():
                updated = _replace_row_height(updated, row, expected, replacement)
            if updated != xml:
                changed_members[member] = updated.encode("utf-8")
        for member, replacements in XML_PATCHES.get(path, {}).items():
            original = source.read(member)
            xml = changed_members.get(member, original).decode("utf-8")
            updated = xml
            for expected, replacement in replacements:
                updated = _replace_xml_fragment(
                    updated,
                    member,
                    expected,
                    replacement,
                )
            if updated.encode("utf-8") != original:
                changed_members[member] = updated.encode("utf-8")
        if check:
            print(f"CHECK {path}: {len(changed_members)} OOXML member(s) would change")
            return
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        os.close(fd)
        temp = Path(temp_name)
        try:
            with zipfile.ZipFile(temp, "w") as target:
                for info in source.infolist():
                    target.writestr(info, changed_members.get(info.filename, source.read(info.filename)))
            with zipfile.ZipFile(temp, "r") as validation:
                if validation.testzip() is not None:
                    raise RuntimeError(f"zip CRC validation failed: {temp}")
                for member, payload in changed_members.items():
                    if validation.read(member) != payload:
                        raise RuntimeError(f"written XML mismatch: {member}")
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)
    print(f"UPDATED {path}: {before_sha} -> {_sha256(path)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="validate expected cells without writing")
    args = parser.parse_args()
    for workbook, patches in PATCHES.items():
        patch_workbook(workbook, patches, check=args.check)


if __name__ == "__main__":
    main()
