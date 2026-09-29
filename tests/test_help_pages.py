"""WORKSPACE_V2_CONTRACT §5.4 C8: the three help pages, their stylesheet and the example STL.

The error table quotes the service's messages verbatim; this test keeps the quoted texts and the source in step.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from wss_deploy.paths import STATIC_DIR

V2 = STATIC_DIR / "v2"
PAGES = ("help_input.html", "help_quickstart.html", "help_errors.html")
PACKAGE = Path(STATIC_DIR).parent
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿️]")


def _text(html: str) -> str:
    return re.sub(r"<[^>]+>", " ", html)


@pytest.mark.parametrize("name", PAGES)
def test_help_pages_are_static_and_follow_the_visual_rules(name):
    html = (V2 / name).read_text(encoding="utf-8")
    assert html.startswith("<!doctype html>") and '<meta charset="utf-8">' in html
    assert '<link rel="stylesheet" href="help.css">' in html
    assert "<script" not in html.lower()
    assert not re.search(r"\son[a-z]+=", html)                       # no inline handlers (CSP: no inline scripts)
    text = _text(html)
    assert not EMOJI.search(text) and "！" not in text and "!" not in text
    for word in ("智能", "一键", "轻松", "您好", "欢迎"):
        assert word not in text, word
    for other in PAGES:                                                # every page links the other two
        assert f'href="{other}"' in html


def test_help_css_carries_the_contract_colours():
    css = (V2 / "help.css").read_text(encoding="utf-8")
    for token in ("--bg:#f3f4f5", "--panel:#ffffff", "--ink:#1b2430", "--ink-2:#465361", "--ink-3:#77828e", "--line:#dfe3e7",
                  "--line-2:#c6ccd2", "--accent:#1d5d95", "--accent-weak:#e9f0f7", "--warn:#8a5a00", "--warn-bg:#fbf3e2",
                  "--error:#9c3326", "--error-bg:#faece9", "--ok:#2c6a4b", "--viewport:#eceef0", "--missing:#a7adb3"):
        assert token in css, token
    assert "-gradient(" not in css and "@page{size:A4" in css


def test_input_page_states_the_real_criteria_and_has_the_diagram():
    html = (V2 / "help_input.html").read_text(encoding="utf-8")
    assert "<svg" in html and "</svg>" in html and 'role="img"' in html
    from wss_deploy import ingest
    assert ingest.MIN_BBOX_DIAG_MM == 50.0 and ingest.MAX_BBOX_DIAG_MM == 1500.0 and "50–1500 mm" in html
    assert ingest.MAX_FRAGMENT_FRACTION == 0.01 and "1%" in html
    assert ingest.MAX_FACES == 2_000_000 and "2,000,000" in html and "128 MiB" in html
    assert "D &lt; 2 按米，2 ≤ D &lt; 100 按厘米，D ≥ 100 按毫米" in html
    assert "恰好 5 个开口" in html and "三出口几何目前不支持" in html and "面积最大的开口" in html


# Messages quoted in help_errors.html, each checked against the module that raises it.
QUOTED = {
    "ingest.py": ["；需要主动脉入口与四个髂动脉出口，且切口保持开放。", "，可能含断裂分支，不能自动删除。",
                  "换算后的整体尺寸超出工具预检范围（包围盒对角 50–1500 mm）；请重新核对单位或血管范围。",
                  "表面含退化面片或无效面积，请修复后重新导出。", "表面含重复三角面，请修复后重新导出。",
                  " 条非流形边，请修复表面。", " 个非环状连接点，不能视为有效开口。",
                  "STL 文件为空、不可读或超过 128 MiB 上限。", "STL 面片数超过 2,000,000 上限或没有面片。",
                  "STL 二进制长度不匹配，或 ASCII 格式不正确。", "无法读取 STL，请导出有效的 ASCII 或二进制 STL。",
                  "STL 坐标含 NaN/Inf 或维数不正确。", "自动单位有多个合理候选，请确认 STL 原始单位。",
                  "自动单位没有唯一的合理候选，请明确选择 mm、cm 或 m。", "自动单位接近尺寸判定边界，请确认 STL 原始单位。",
                  " 的面积，请核对后确认。", " 条内部边的相邻面片绕序不一致，建议复核法向。", " 个面片极度狭长，建议复核网格质量。"],
    "errors.py": ["GPU 显存不足，已改用 CPU 重试。", "内存不足，请稍后重试。", "磁盘空间不足，请联系维护者清理后重试。",
                  "中心线提取超时。请检查网格是否异常大或含大量碎片。", "计算环境缺少必要模块，请联系维护者。"],
    "centerline.py": ["秒）。请检查网格是否异常大或含大量碎片。", "中心线提取失败（VMTK / vessel_geom 出错）。请检查表面质量后重试，或联系维护者。",
                      "中心线提取没有生成完整结果，请重试或联系维护者。", "），无法自动命名", "不在同一侧；同一侧必须是 左内+左外 或 右内+右外"],
    "jobs.py": ["计算未完成，请重试；如仍失败，请向维护者提供诊断编号。", "输入或中心线未通过检查，请检查 STL 和开口。",
                "自动命名未满足已校准的 95% 自动门控，请结合原始影像核对开口名称。", "四个出口必须分别且唯一对应左外、左内、右外、右内。",
                "入口必须是当前几何中有效的开口编号。", "请确认出口命名及其对计算的影响。", "历史任务缺少发布包指纹，不能安全重试。",
                "绑定的发布包指纹已变化，不能继续重试。", "仅失败、取消或中断的任务可以重试。", "请上传非空 STL 文件。", "同一几何已有任务。",
                "同一几何的已有任务没有可复用的中心线与出口确认，请选择强制重算。", "文件未能保存，请重试或提供诊断编号给维护者。",
                "扫描日期必须是有效的 YYYY-MM-DD 日期。", "；请先「重新打开」并说明原因。", "任务状态已更新，请刷新后再操作。",
                "任务阶段已改变，请刷新后再操作。", "该任务正在后台按新版分析重建（升级后自动执行），请稍后再操作。",
                "任务不存在或不属于当前会话。", "任务正在计算，请先取消并等待其停止后再删除。",
                "只有已完成且确认出口的任务才能换模型重跑。", "同名任务已存在，不能恢复。"],
    "pipeline.py": ["输入检查尚未通过，不能开始预测。", "必须确认出口命名后才能开始预测。"],
}


def test_error_table_quotes_the_service_messages_verbatim():
    page = _text((V2 / "help_errors.html").read_text(encoding="utf-8"))
    page = re.sub(r"\s+", " ", page)
    for module, messages in QUOTED.items():
        source = (PACKAGE / module).read_text(encoding="utf-8")
        for message in messages:
            assert message in source, (module, message)
            assert message.strip() in page, (module, message)


def test_example_stl_is_anonymous_and_passes_the_input_check(tmp_path):
    path = V2 / "example_aaa.stl"
    if not path.is_file():   # patient-derived: not in git, written on the server by `python -m wss_deploy.v2_examples`
        pytest.skip("example_aaa.stl has not been generated on this machine")
    data = path.read_bytes()
    assert b"LV_GUO_YOU" not in data[:80] and not data[:5].lower().startswith(b"solid")
    from wss_deploy.ingest import ingest
    result = ingest(path, tmp_path, units="auto")
    assert result["status"] == "pass" and result["openings"] == 5 and result["resolved_units"] == "mm"
    assert result["faces"] == 23184


def test_workbench_links_resolve_to_files():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    for name in re.findall(r'href="/static/v2/([^"]+)"', html):
        if name.startswith("example_") and not (V2 / name).is_file():
            continue   # generated on the server (wss_deploy.v2_examples), never committed
        assert (V2 / name).is_file(), name
