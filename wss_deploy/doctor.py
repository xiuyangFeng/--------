"""Environment self-check: ``python -m wss_deploy.cli doctor [--json] [--jobs-root …] [--host …]`` (contract §19.6).

Every check yields ``ok`` (✓), ``info`` (ℹ, 信息), ``warn`` (⚠, 警告) or ``fail`` (✗, 硬错) with one sentence of
advice; the exit code is 1 when any check fails, and ``service upgrade`` aborts on any ✗ (its preflight runs
``doctor --skip service``).  Nothing here writes anywhere (writability is tested with ``os.access``; sizes and ages
are read with ``lstat``), so it is safe on the production jobs root.

v0.15 (上线前可运维加固): exposure (shared binding needs password login with an enabled admin; direct HTTP without
a TLS reverse proxy is reported), secret-file permissions, disk thresholds (``WSS_DEPLOY_MIN_FREE_GB``), log
rotation, derived caches (``.tmp`` / ``geometry_cache``), release pins (``env/releases.sha256``), VMTK executable
bit and the resolved time zone (``clock``).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

SYMBOLS = {"ok": "✓", "info": "ℹ", "warn": "⚠", "fail": "✗"}
SEVERITY = {"ok": "通过", "info": "信息", "warn": "警告", "fail": "硬错"}
DISK_WARN_GB = 20.0
DISK_FAIL_GB = 5.0
MIN_FREE_ENV = "WSS_DEPLOY_MIN_FREE_GB"
TRUST_PROXY_ENV = "WSS_DEPLOY_TRUST_PROXY"
SECRET_FILES = ("users.json", ".sessions.json", ".service_token")
LOG_SLACK = 1.10                 # a log this much above the rotation size means rotation is not happening
TRUTHY = {"1", "true", "yes", "on"}


def disk_thresholds() -> tuple[float, float]:
    """``(fail_below_gb, warn_below_gb)``: 5 / 20 GB, or ``WSS_DEPLOY_MIN_FREE_GB`` = ``<fail>`` or ``<fail>,<warn>``
    (one number keeps the warning at max(20, fail)).  Also used by ``JobManager.health()`` (``/api/ready``)."""
    raw = str(os.environ.get(MIN_FREE_ENV) or "").strip()
    if not raw:
        return DISK_FAIL_GB, DISK_WARN_GB
    try:
        parts = [float(p) for p in raw.replace("，", ",").split(",") if p.strip()]
    except ValueError:
        return DISK_FAIL_GB, DISK_WARN_GB
    if not parts or parts[0] < 0:
        return DISK_FAIL_GB, DISK_WARN_GB
    fail = parts[0]
    warn = parts[1] if len(parts) > 1 and parts[1] >= fail else max(DISK_WARN_GB, fail)
    return fail, warn


def _check(key: str, label: str, status: str, detail: str, advice: str = "") -> dict:
    return {"key": key, "label": label, "status": status, "detail": detail, "advice": advice}


def check_torch() -> list[dict]:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 — any import failure is the finding
        return [_check("torch", "PyTorch", "fail", f"无法导入 torch：{exc}", "请在 GNN 环境（~/.conda/envs/GNN）中运行。")]
    rows = [_check("torch", "PyTorch", "ok", f"torch {torch.__version__}，Python {sys.version.split()[0]}（{sys.executable}）")]
    try:
        # device_count() does not create a CUDA context (get_device_name would, ~300 MB on GPU 0); names via nvidia-smi.
        from .service import gpu_summary
        available = torch.cuda.is_available()
        count = torch.cuda.device_count() if available else 0
        names = (gpu_summary().get("names") or [])[:count] if available else []
        names = names or [f"GPU {i}" for i in range(count)]
    except Exception as exc:  # noqa: BLE001
        available, names = False, []
        rows.append(_check("cuda", "CUDA", "warn", f"CUDA 检查出错：{exc}", "推理会退回 CPU（单例约 100 s）。"))
        return rows
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    scope = "CUDA_VISIBLE_DEVICES 未设置" if visible is None else f"CUDA_VISIBLE_DEVICES={visible!r}"
    if available:
        label = names[0] if len(set(names)) == 1 else "、".join(names)
        rows.append(_check("cuda", "CUDA", "ok", f"{len(names)} × {label}（{scope}）"))
    else:
        rows.append(_check("cuda", "CUDA", "warn", f"当前进程看不到 GPU（{scope}）", "推理会用 CPU（单例约 100 s）；需要 GPU 时检查 CUDA_VISIBLE_DEVICES。"))
    return rows


def check_vmtk() -> list[dict]:
    from .paths import VESSEL_GEOM_DIR, VMTK_PYTHON
    rows = []
    geom_ok = (Path(VESSEL_GEOM_DIR) / "vessel_geom" / "cli.py").is_file()
    rows.append(_check("vessel_geom", "vessel_geom 路径", "ok" if geom_ok else "fail", str(VESSEL_GEOM_DIR),
                       "" if geom_ok else "设置 WSS_DEPLOY_VESSEL_GEOM 指向含 vessel_geom/cli.py 的目录。"))
    if not Path(VMTK_PYTHON).is_file():
        rows.append(_check("vmtk", "VMTK 解释器", "fail", f"不存在：{VMTK_PYTHON}", "设置 WSS_DEPLOY_VMTK_PYTHON 指向 GNN_vmtk 环境的 python。"))
        return rows
    if not os.access(VMTK_PYTHON, os.X_OK):
        rows.append(_check("vmtk", "VMTK 解释器", "fail", f"不可执行：{VMTK_PYTHON}", f"chmod u+x {VMTK_PYTHON}，或设置 WSS_DEPLOY_VMTK_PYTHON。"))
        return rows
    try:
        proc = subprocess.run([str(VMTK_PYTHON), "-c", "import vmtk; from vmtk import vmtkscripts; import vessel_geom.cli; print(vmtk.__file__)"],
                              cwd=str(VESSEL_GEOM_DIR) if geom_ok else None, capture_output=True, text=True, timeout=120)
        if proc.returncode == 0:
            rows.append(_check("vmtk", "VMTK 解释器", "ok", f"{VMTK_PYTHON}：vmtk 与 vessel_geom 可导入"))
        else:
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or ["未知错误"]
            rows.append(_check("vmtk", "VMTK 解释器", "fail", f"{VMTK_PYTHON}：{tail[0]}", "检查 GNN_vmtk 环境是否完整（vmtk、vtk、vessel_geom 依赖）。"))
    except (OSError, subprocess.SubprocessError) as exc:
        rows.append(_check("vmtk", "VMTK 解释器", "fail", f"{VMTK_PYTHON}：{exc}", "检查解释器是否可执行。"))
    return rows


def check_releases() -> list[dict]:
    from .registry import ReleaseError, ReleaseRegistry, _contract, _verify_package
    try:
        registry = ReleaseRegistry()
    except Exception as exc:  # noqa: BLE001
        return [_check("releases", "发布包", "fail", f"没有可用发布包：{exc}", "检查 outputs/wss_deploy_release/ 与 WSS_DEPLOY_RELEASE_ROOT。")]
    rows = []
    known = set()
    for public in registry.list():
        rid = public["id"]
        record = registry._records[rid]
        known.add(record.path.resolve())
        default = "（默认）" if public.get("default") else ""
        try:
            registry.describe(rid)
            _verify_package(record)
            rows.append(_check(f"release:{rid}", f"发布包 {rid}{default}", "ok",
                               f"合同 {record.contract.get('protocol')}，{public.get('models_count', '?')} 个模型，MANIFEST 逐文件校验通过，指纹 {record.fingerprint[:12]}"))
        except (ReleaseError, OSError, ValueError) as exc:
            rows.append(_check(f"release:{rid}", f"发布包 {rid}{default}", "fail", str(exc), "不要手改发布包；需要更新请重建并换新的 release 标识。"))
        rows.append(_reference_check(rid, record.path))
    for path in sorted(registry.root.iterdir()) if registry.root.is_dir() else []:
        if (path / "release.json").is_file() and path.resolve() not in known:
            try:
                _contract(json.loads((path / "release.json").read_text(encoding="utf-8")))
                reason = "标识重复或不可读"
            except Exception as exc:  # noqa: BLE001
                reason = str(exc)
            rows.append(_check(f"release:{path.name}", f"发布包 {path.name}", "fail", f"未被识别：{reason}", "修正 release.json 或移出发布根目录。"))
    return rows


def _reference_check(rid: str, path: Path) -> dict:
    sidecar = Path(path) / "reference.json"
    label = f"参考侧车 {rid}"
    if not sidecar.is_file():
        return _check(f"reference:{rid}", label, "warn", "没有 reference.json：几何越界提示与人群分位对该发布包不生效",
                      "用 python -m wss_deploy.build_reference_profiles 生成（不改发布包指纹）。")
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("不是 JSON 对象")
        parts = []
        for key, name in (("geometry_reference", "几何范围"), ("population_reference", "人群分位")):
            block = data.get(key)
            if block is None:
                continue
            if not isinstance(block, dict) or block.get("release") != rid:
                raise ValueError(f"{key} 未绑定本发布包")
            parts.append(name)
        if not parts:
            return _check(f"reference:{rid}", label, "warn", "reference.json 没有几何范围或人群分位", "重新生成参考侧车。")
        return _check(f"reference:{rid}", label, "ok", "含" + "、".join(parts))
    except (OSError, ValueError) as exc:
        return _check(f"reference:{rid}", label, "fail", f"reference.json 无效：{exc}", "重新生成参考侧车；无效侧车会让该发布包的任务失败。")


def check_feature_contract() -> list[dict]:
    try:
        from wss_features import contract
        info = contract()
    except Exception as exc:  # noqa: BLE001
        return [_check("features", "特征合同", "fail", f"无法读取 wss_features 合同：{exc}", "确认 PYTHONPATH 指向项目根目录。")]
    if info.get("matches_record"):
        return [_check("features", "特征合同", "ok", f"{info['version']}，源码哈希与 CONTRACT.json 一致（{str(info['source_hash'])[:12]}）")]
    return [_check("features", "特征合同", "fail",
                   f"源码哈希 {str(info.get('source_hash'))[:12]} ≠ CONTRACT.json {str(info.get('recorded_hash'))[:12]}",
                   "特征程序被改动：先与训练实现做等价验证，再 python -m wss_features record。")]


def check_glossary() -> list[dict]:
    try:
        from .glossary import glossary_json
        from .paths import STATIC_DIR
        stored = (STATIC_DIR / "glossary.json").read_text(encoding="utf-8")
        same = stored == glossary_json()
    except Exception as exc:  # noqa: BLE001
        return [_check("glossary", "术语表", "fail", f"无法比较术语表：{exc}", "运行 python -m wss_deploy.glossary。")]
    if same:
        return [_check("glossary", "术语表", "ok", "static/glossary.json 与 glossary.py 同步")]
    return [_check("glossary", "术语表", "fail", "static/glossary.json 与 glossary.py 不同步", "运行 PYTHONPATH=. python -m wss_deploy.glossary。")]


def check_jobs_root(root: Path) -> list[dict]:
    from .service import disk_free_gb
    rows = []
    if not root.is_dir():
        rows.append(_check("jobs_root", "任务目录", "warn", f"{root} 不存在", "首次启动服务时会创建。"))
        parent = root.parent
    else:
        writable = os.access(root, os.W_OK | os.X_OK)
        rows.append(_check("jobs_root", "任务目录", "ok" if writable else "fail", f"{root}" + ("（可写）" if writable else "（不可写）"),
                           "" if writable else "检查目录权限；服务需要写任务记录与报告。"))
        parent = root
    free = disk_free_gb(parent)
    fail_gb, warn_gb = disk_thresholds()
    limits = f"阈值：< {warn_gb:g} GB 警告、< {fail_gb:g} GB 硬错" + (f"（{MIN_FREE_ENV}）" if os.environ.get(MIN_FREE_ENV) else "")
    if free is None:
        rows.append(_check("disk", "磁盘空间", "warn", "无法读取剩余空间"))
    elif free < fail_gb:
        rows.append(_check("disk", "磁盘空间", "fail", f"剩余 {free} GB（{limits}）",
                           "立即清理：python -m wss_deploy.cli jobs du 找大户，jobs prune-cache --older-than 7 --yes 清可重算缓存；不足时任务会失败。"))
    elif free < warn_gb:
        rows.append(_check("disk", "磁盘空间", "warn", f"剩余 {free} GB（{limits}）",
                           "清理旧任务或回收站（30 天自动清空）；python -m wss_deploy.cli jobs du 查看占用。"))
    else:
        rows.append(_check("disk", "磁盘空间", "ok", f"剩余 {free} GB（{limits}）"))
    return rows


def _modules_newer_than_process(pid: int) -> list[str]:
    """``wss_deploy`` / ``wss_features`` modules modified after the service process started (it still runs the old code)."""
    from .service import _proc_started
    started = _proc_started(pid)
    if not started:
        return []
    root = Path(__file__).resolve().parent
    return sorted(str(path.relative_to(root.parent)) for folder in (root, root.parent / "wss_features")
                  for path in folder.glob("*.py") if path.stat().st_mtime > started + 1)


def check_service(root: Path, port: int | None = None) -> list[dict]:
    from .service import status
    info = status(root, port)
    rows = []
    if not info["running"]:
        rows.append(_check("service", "服务", "warn", f"端口 {info['port']} 上没有运行中的服务",
                           "启动：python -m wss_deploy.cli service start（共享给他人加 --host 0.0.0.0）。"))
        shared = False
    else:
        how = "service 管理" if info["managed"] else "手工启动（未托管）"
        if not info.get("healthy"):
            rows.append(_check("service", "服务", "fail", f"PID {info['pid']}（{how}）健康检查未响应", "查看日志：python -m wss_deploy.cli service logs。"))
        elif info.get("legacy"):
            rows.append(_check("service", "服务", "warn", f"PID {info['pid']}（{how}）运行旧版代码（无 /api/health）",
                               "Python 模块已更新，需要 python -m wss_deploy.cli service upgrade 才生效（停服务、重建需要新分析的任务、刷新报告、再启动）。"))
        elif not info["managed"]:
            rows.append(_check("service", "服务", "warn", f"PID {info['pid']} 版本 {info.get('version')}，手工启动（未托管）",
                               "执行一次 python -m wss_deploy.cli service upgrade 接管并升级（沿用参数、环境与令牌）。"))
        else:
            newer = _modules_newer_than_process(info["pid"])
            if newer:
                rows.append(_check("service", "服务", "warn", f"PID {info['pid']} 版本 {info.get('version')}：服务启动后有 {len(newer)} 个 Python 模块更新（{'、'.join(newer[:4])}{' 等' if len(newer) > 4 else ''}）",
                                   "运行中的进程仍是旧代码：python -m wss_deploy.cli service upgrade。"))
            else:
                rows.append(_check("service", "服务", "ok", f"PID {info['pid']} 版本 {info.get('version')}，{info['address']}"))
        shared = bool(info.get("shared"))
    users = (root / "users.json").is_file()
    if shared and not users:
        rows.append(_check("login", "登录方式", "warn", "共享模式只用令牌登录：换浏览器后任务不在原会话下",
                           "建议启用用户名登录：python -m wss_deploy.cli user add <名字> [--admin]。"))
    elif users:
        rows.append(_check("login", "登录方式", "ok", "已启用用户名登录（users.json）"))
    else:
        rows.append(_check("login", "登录方式", "ok", "本机回环模式免登录" if info["running"] else "服务未运行；共享模式建议启用用户名登录"))
    return rows


def check_lock(root: Path) -> list[dict]:
    """J3/J10: the single-writer lock of the jobs root (``.service.lock``)."""
    from .service import LOCK_FILE, lock_holder, scan_processes
    holder = lock_holder(root)
    running = [info for info in scan_processes() if str(info.get("jobs_root")) == str(Path(root).resolve())]
    if holder is None:
        if running:
            return [_check("lock", "写锁", "warn", f"服务（PID {running[0]['pid']}）在运行但任务目录没有 {LOCK_FILE}：旧版进程，不防并发写入",
                           "执行 python -m wss_deploy.cli service upgrade 后由新服务持有写锁。")]
        return [_check("lock", "写锁", "ok", "没有进程持有任务目录写锁（服务未运行）")]
    pid, purpose = holder.get("pid"), holder.get("purpose") or "?"
    if holder.get("held"):
        if purpose == "serve":
            return [_check("lock", "写锁", "ok", f"由服务 PID {pid} 持有（自 {holder.get('started_at') or '?'}）")]
        return [_check("lock", "写锁", "warn", f"由维护命令 PID {pid}（{purpose}）持有：服务此时无法启动",
                       "等待该命令结束；卡住时确认进程后再结束它。")]
    if running:
        return [_check("lock", "写锁", "warn", f"服务（PID {running[0]['pid']}）在运行但没有持有写锁（锁文件上次由 PID {pid}（{purpose}）写入）",
                       "旧版进程：执行 python -m wss_deploy.cli service upgrade。")]
    return [_check("lock", "写锁", "ok", f"未被持有（锁文件是 PID {pid}（{purpose}）留下的，flock 已随进程释放，无需清理）")]


def check_git() -> list[dict]:
    """J4: results record ``git_dirty``; a dirty tree means the manifest cannot name the exact code by commit alone."""
    try:
        from .schema import code_provenance
        code = code_provenance(refresh=True)
    except Exception as exc:  # noqa: BLE001
        return [_check("git", "代码版本", "warn", f"无法读取代码版本：{exc}")]
    describe, dirty = code.get("git_describe"), code.get("git_dirty")
    label = f"wss_deploy {code.get('deploy_version')}，{describe or '无 git 信息'}，源码哈希 {str(code.get('source_hash'))[:12]}"
    if dirty is None:
        return [_check("git", "代码版本", "warn", label, "不在 git 工作树中：结果清单只能用 source_hash 标识代码。")]
    if dirty:
        return [_check("git", "代码版本", "warn", label + "（工作树有未提交改动）",
                       "结果清单会记录 git_dirty=true 与 source_hash；正式结果建议在提交后的干净工作树上计算。")]
    return [_check("git", "代码版本", "ok", label)]


def check_access_log(root: Path) -> list[dict]:
    """J8: request lines go to ``<jobs_root>/access.log`` (rotating), application lines to server.log."""
    from .service import ACCESS_LOG
    path = Path(root) / ACCESS_LOG
    if path.is_file():
        writable = os.access(path, os.W_OK)
        size = path.stat().st_size / 1e6
        return [_check("access_log", "访问日志", "ok" if writable else "fail", f"{path}（{size:.1f} MB，轮转 20 MB × 5）",
                       "" if writable else "检查文件权限：服务需要追加访问日志。")]
    parent_ok = Path(root).is_dir() and os.access(root, os.W_OK | os.X_OK)
    if not Path(root).is_dir():
        return [_check("access_log", "访问日志", "ok", f"{path}（任务目录尚未创建）")]
    return [_check("access_log", "访问日志", "ok" if parent_ok else "fail", f"{path}（尚未生成：v0.14 的 serve 启动后写入）",
                   "" if parent_ok else "任务目录不可写。")]


def check_reports(root: Path) -> list[dict]:
    from .report_freshness import auto_enabled, stale_jobs
    if not root.is_dir():
        return []
    stale = stale_jobs(root)
    if not stale:
        return [_check("reports", "报告模板", "ok", "所有已完成任务的报告都是当前模板")]
    mode = "打开报告时会自动刷新" if auto_enabled() else "自动刷新已关闭（WSS_DEPLOY_AUTO_REFRESH_REPORTS=0）"
    return [_check("reports", "报告模板", "warn", f"{len(stale)} 个报告不是当前模板（{mode}）",
                   "也可一次刷新：python -m wss_deploy.cli reports refresh --all。")]


# ---------------------------------------------------------------------------------------------- v0.15 上线预检
def _truthy(value) -> bool:
    return str(value or "").strip().lower() in TRUTHY


def service_host(root: Path, host: str | None = None) -> tuple[str, str, dict]:
    """The bind address the service of ``root`` uses or will use: ``host`` (``doctor --host``, passed by the upgrade
    preflight with its ``--host`` override) → ``service.json`` → the running (hand-started) process → the default.
    Returns ``(host, where, config env)``; read only."""
    from .service import DEFAULT_HOST, load_config, locate
    cfg = load_config(root)
    env = dict(cfg.get("env") or {})
    if host:
        return host, "--host 参数", env
    if cfg.get("exists"):
        return str(cfg.get("host") or DEFAULT_HOST), "service.json", env
    try:
        info = locate(root)
    except Exception:  # noqa: BLE001 — a /proc scan problem must not break doctor
        info = None
    if info and info.get("host") and info.get("root_matches"):
        return str(info["host"]), f"运行中进程 PID {info['pid']}", env
    return DEFAULT_HOST, "默认值", env


def _trust_proxy(env: dict) -> bool:
    return _truthy(os.environ.get(TRUST_PROXY_ENV)) or _truthy((env or {}).get(TRUST_PROXY_ENV))


def check_exposure(root: Path, host: str | None = None) -> list[dict]:
    """Shared binding (non-loopback, or ``WSS_DEPLOY_TRUST_PROXY=1``) needs password login with an enabled admin
    (硬错 otherwise: a token-only or open service would be reachable by the whole network); direct HTTP on a
    non-loopback address without a TLS reverse proxy is reported as 信息."""
    from .service import is_loopback
    from .users import UserStore
    bound, where, env = service_host(root, host)
    proxy = _trust_proxy(env)
    shared = not is_loopback(bound) or proxy
    rows = []
    if not shared:
        rows.append(_check("bind_login", "共享绑定与登录", "ok", f"绑定 {bound}（{where}）：本机回环，免登录"))
    else:
        path = Path(root) / "users.json"
        try:
            users = UserStore(path).load() if path.is_file() else {}
            admins = sorted(name for name, row in users.items()
                            if isinstance(row, dict) and not row.get("disabled") and (row.get("role") or "user") == "admin")
            enabled = sum(1 for row in users.values() if isinstance(row, dict) and not row.get("disabled"))
            problem = None if admins else ("没有 users.json（只能用令牌登录）" if not path.is_file()
                                           else f"users.json 有 {enabled} 个启用用户但没有启用的管理员")
        except Exception as exc:  # noqa: BLE001 — unreadable users.json is itself the finding
            admins, problem = [], f"users.json 无法读取：{exc}"
        scope = f"绑定 {bound}（{where}）" + ("，WSS_DEPLOY_TRUST_PROXY=1（反代后同样需要登录）" if proxy else "")
        if problem:
            rows.append(_check("bind_login", "共享绑定与登录", "fail", f"{scope}：{problem}",
                               "共享模式必须用用户名登录：python -m wss_deploy.cli user add <名字> --admin --jobs-root "
                               f"{Path(root)}（或把服务改回 --host 127.0.0.1）。"))
        else:
            rows.append(_check("bind_login", "共享绑定与登录", "ok", f"{scope}：用户名登录，启用的管理员 {len(admins)} 个（{'、'.join(admins[:3])}）"))
    if not is_loopback(bound):
        if proxy:
            rows.append(_check("tls", "传输加密", "info", f"绑定 {bound} 且 WSS_DEPLOY_TRUST_PROXY=1：只信任回环对端的 X-Forwarded-*",
                               "反代应监听 443 并把服务改绑 127.0.0.1，避免绕过反代直连 HTTP。"))
        else:
            rows.append(_check("tls", "传输加密", "info", f"绑定 {bound}，WSS_DEPLOY_TRUST_PROXY 未开：直接暴露 HTTP（口令与会话 Cookie 明文传输）",
                               "建议反代 TLS：nginx/caddy 终止 HTTPS → 服务 --host 127.0.0.1 + --env WSS_DEPLOY_TRUST_PROXY=1。"))
    return rows


def check_secret_perms(root: Path) -> list[dict]:
    """users.json / .sessions.json / .service_token must be 0600 (password hashes, session ids, the token)."""
    import stat as _stat
    loose, present = [], []
    for name in SECRET_FILES:
        path = Path(root) / name
        try:
            st = path.lstat()
        except OSError:
            continue
        present.append(name)
        mode = _stat.S_IMODE(st.st_mode)
        if mode != 0o600 or _stat.S_ISLNK(st.st_mode):
            loose.append((path, mode))
    if not present:
        return [_check("secret_perms", "凭据文件权限", "ok", "没有 users.json / .sessions.json / .service_token")]
    if loose:
        detail = "；".join(f"{path.name} 为 {mode:04o}" for path, mode in loose)
        return [_check("secret_perms", "凭据文件权限", "warn", detail + "（应为 0600）",
                       "chmod 600 " + " ".join(str(path) for path, _ in loose))]
    return [_check("secret_perms", "凭据文件权限", "ok", "、".join(present) + " 均为 0600")]


def check_log_rotation(root: Path) -> list[dict]:
    """server.log is rotated by ``serve --log-file`` (``service start`` always passes it) and access.log by the
    ``wss_deploy.access`` handler; a file well past the rotation size means a second writer or an old process."""
    from .service import ACCESS_LOG, LOG_FILE, load_config, locate
    try:
        from .server import LOG_BACKUPS, LOG_MAX_BYTES
    except Exception:  # noqa: BLE001
        LOG_MAX_BYTES, LOG_BACKUPS = 20 * 1024 * 1024, 5
    cfg = load_config(root)
    problems, parts = [], []
    try:
        info = locate(root)
    except Exception:  # noqa: BLE001
        info = None
    if info and not info.get("root_matches"):
        info = None                  # a service of another jobs root on the same port
    server_log = Path(cfg.get("log_file") or Path(root) / LOG_FILE)
    if info and not info.get("log_file") and not (info.get("managed") and (info.get("pid_record") or {}).get("log_file")):
        problems.append(f"运行中的服务（PID {info['pid']}）没有 --log-file：日志只到标准输出、不轮转")
    elif not cfg.get("exists") and not info:
        parts.append("尚未由 service 启动（service start 会带 --log-file）")
    if LOG_MAX_BYTES <= 0 or LOG_BACKUPS <= 0:
        problems.append(f"轮转参数无效（{LOG_MAX_BYTES} B × {LOG_BACKUPS}）")
    for path in (server_log, Path(root) / ACCESS_LOG):
        try:
            size = path.stat().st_size
        except OSError:
            parts.append(f"{path.name} 尚未生成")
            continue
        backups = sum(1 for i in range(1, LOG_BACKUPS + 1) if Path(f"{path}.{i}").is_file())
        parts.append(f"{path.name} {size / 1e6:.1f} MB（备份 {backups} 个）")
        if size > LOG_MAX_BYTES * LOG_SLACK:
            problems.append(f"{path.name} {size / 1e6:.1f} MB 超过轮转上限 {LOG_MAX_BYTES / 1e6:.0f} MB：轮转没有生效")
    limit = f"轮转 {LOG_MAX_BYTES // (1024 * 1024)} MB × {LOG_BACKUPS}"
    if problems:
        return [_check("log_rotation", "日志轮转", "warn", "；".join(problems) + f"（{limit}）",
                       "用 python -m wss_deploy.cli service upgrade 重启为托管服务（带 --log-file 与访问日志轮转）。")]
    return [_check("log_rotation", "日志轮转", "ok", f"{limit}；" + "，".join(parts))]


def check_derived_cache(root: Path) -> list[dict]:
    """Size and oldest file of ``.tmp/`` and of every job's ``geometry_cache/`` (信息; nothing is deleted here)."""
    from . import housekeeping as H
    from .clock import iso
    if not Path(root).is_dir():
        return []
    tmp = H._tree(Path(root) / H.TMP_DIR)
    caches = [H._tree(job / H.GEOMETRY_CACHE) for job in H.job_dirs(root) if (job / H.GEOMETRY_CACHE).is_dir()]
    cache_bytes = sum(c["bytes"] for c in caches)
    oldest = min((c["oldest"] for c in caches if c["oldest"] is not None), default=None)
    detail = (f".tmp {tmp['files']} 个文件 {H.human(tmp['bytes'])}" + (f"（最老 {iso(tmp['oldest'])}）" if tmp["oldest"] else "")
              + f"；geometry_cache {len(caches)} 个任务 {sum(c['files'] for c in caches)} 个文件 {H.human(cache_bytes)}"
              + (f"（最老 {iso(oldest)}）" if oldest else ""))
    return [_check("derived_cache", "可重算缓存", "info", detail,
                   "按需清理：python -m wss_deploy.cli jobs prune-cache --older-than 30（默认只列清单，加 --yes 才删除）。")]


def _pinned_release_hashes() -> dict[str, str]:
    path = Path(__file__).resolve().parent / "env" / "releases.sha256"
    out = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 2 and len(parts[0]) == 64:
                out[parts[1].lstrip("*")] = parts[0].lower()
    except OSError:
        pass
    return out


def check_release_pins() -> list[dict]:
    """``release.json`` + ``MANIFEST.sha256`` of every release against ``wss_deploy/env/releases.sha256`` (the
    checklist in git): a mismatch means the package on disk is not the one the code was validated with."""
    import hashlib
    from .registry import ReleaseRegistry
    pins = _pinned_release_hashes()
    if not pins:
        return [_check("release_pins", "发布包清单核对", "info", "没有 wss_deploy/env/releases.sha256，跳过")]
    try:
        root = ReleaseRegistry().root
    except Exception as exc:  # noqa: BLE001 — check_releases reports the registry failure itself
        return [_check("release_pins", "发布包清单核对", "info", f"发布包目录不可用：{exc}")]
    matched, mismatched, unlisted = [], [], []
    for folder in sorted(p for p in Path(root).iterdir() if (p / "release.json").is_file()):
        for name in ("release.json", "MANIFEST.sha256"):
            rel = f"{folder.name}/{name}"
            if rel not in pins:
                unlisted.append(rel)
                continue
            try:
                digest = hashlib.sha256((folder / name).read_bytes()).hexdigest()
            except OSError:
                digest = None
            (matched if digest == pins[rel] else mismatched).append(rel)
    if mismatched:
        return [_check("release_pins", "发布包清单核对", "warn", f"{len(mismatched)} 个文件与 env/releases.sha256 不一致：{'、'.join(mismatched[:4])}",
                       "发布包被替换或清单过期：核对来源后再更新 wss_deploy/env/releases.sha256（见 env/README.md）。")]
    status = "info" if unlisted else "ok"
    detail = f"{len(matched)} 个清单文件与 env/releases.sha256 一致" + (f"；未登记 {'、'.join(unlisted[:4])}" if unlisted else "")
    return [_check("release_pins", "发布包清单核对", status, detail,
                   "新发布包请把 release.json 与 MANIFEST.sha256 的 sha256 追加到 wss_deploy/env/releases.sha256。" if unlisted else "")]


def check_tz(root: Path) -> list[dict]:
    """The zone every timestamp is written in (``clock``: WSS_DEPLOY_TZ → service.json env.TZ → system)."""
    from . import clock
    info = clock.describe(None if Path(root).resolve() == clock.jobs_root().resolve() else root)
    source = {"WSS_DEPLOY_TZ": "环境变量 WSS_DEPLOY_TZ", "service.json": "service.json env.TZ", "pending": "本次 --env TZ",
              "system": "系统 /etc/localtime"}.get(info["source"], info["source"])
    detail = f"{info['name']}（UTC{info['offset']}，来自{source}）；系统 {info['system']}（UTC{info['system_offset']}）"
    if info.get("process_tz") and info["process_tz"] != info["name"]:
        detail += f"；当前 shell TZ={info['process_tz']} 不参与（服务与命令行一律按上面的时区记时间）"
    if info["errors"]:
        return [_check("tz", "时区", "warn", detail + "；无效设置：" + "；".join(info["errors"]),
                       "修正时区名（如 Asia/Shanghai）：service upgrade --env TZ=Asia/Shanghai。")]
    advice = "" if info["consistent"] and info["source"] != "system" else \
        "建议固定：service upgrade --env TZ=Asia/Shanghai（写入 service.json，服务与离线命令都读它）。"
    return [_check("tz", "时区", "info" if not info["consistent"] or info["source"] == "system" else "ok", detail, advice)]


CHECKS: dict[str, Callable] = {"torch": check_torch, "vmtk": check_vmtk, "releases": check_releases,
                               "features": check_feature_contract, "glossary": check_glossary}


def run_checks(jobs_root: Path, *, port: int | None = None, skip: set[str] | None = None, host: str | None = None) -> list[dict]:
    """All checks in display order; ``skip`` names groups to leave out (``torch``, ``vmtk``, ``releases``, …).

    ``host`` (``doctor --host``): the bind address about to be used (the upgrade preflight passes its override)."""
    root = Path(jobs_root).resolve()
    skip = set(skip or ())
    rows: list[dict] = []
    for name, fn in CHECKS.items():
        if name not in skip:
            rows += fn()
    if "releases" not in skip:
        rows += check_release_pins()
    if "jobs_root" not in skip:
        rows += check_jobs_root(root)
    if "exposure" not in skip:
        rows += check_exposure(root, host)
        rows += check_secret_perms(root)
    if "lock" not in skip:
        rows += check_lock(root)
    if "logs" not in skip:
        rows += check_access_log(root)
        rows += check_log_rotation(root)
    if "cache" not in skip:
        rows += check_derived_cache(root)
    if "tz" not in skip:
        rows += check_tz(root)
    if "git" not in skip:
        rows += check_git()
    if "service" not in skip:
        rows += check_service(root, port)
    if "reports" not in skip:
        rows += check_reports(root)
    return rows


def format_rows(rows: list[dict]) -> str:
    lines = []
    for row in rows:
        lines.append(f"{SYMBOLS.get(row['status'], '?')} {row['label']}：{row['detail']}")
        if row.get("advice") and row["status"] != "ok":  # ℹ rows carry their suggestion too
            lines.append(f"    → {row['advice']}")
    counts = {status: sum(1 for row in rows if row["status"] == status) for status in SYMBOLS}
    lines.append(f"\n共 {len(rows)} 项：✓ {counts['ok']}  ℹ {counts['info']}  ⚠ {counts['warn']}  ✗ {counts['fail']}（✗ 硬错会阻止 service upgrade）")
    return "\n".join(lines)


def exit_code(rows: list[dict]) -> int:
    return 1 if any(row["status"] == "fail" for row in rows) else 0


__all__ = ["CHECKS", "SEVERITY", "check_derived_cache", "check_exposure", "check_log_rotation", "check_release_pins",
           "check_secret_perms", "check_tz", "disk_thresholds", "exit_code", "format_rows", "run_checks", "service_host"]
