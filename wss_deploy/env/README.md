# wss_deploy 运行环境与 git 之外的文件（2026-09-24 快照）

仓库里只有代码。服务要跑起来还需要两个 conda 环境，以及 `outputs/` 下两个被 git 忽略的目录（`.gitignore` 忽略整个 `outputs/`）。
本目录记录这些依赖的版本，并给出新机器的获取与核对步骤。不提供 `setup.py` / `pyproject`：一直用 `PYTHONPATH=<仓库根>` 运行。

| 文件 | 内容 |
|---|---|
| `GNN.yml` | `conda env export -n GNN --no-builds` 原样导出：服务、测试、推理和几何特征都在这个环境里跑 |
| `GNN_vmtk.yml` | `conda env export -n GNN_vmtk --no-builds` 原样导出：只用来跑 vessel_geom / VMTK 中心线子进程 |
| `releases.sha256` | 三个发布包的 `release.json` 与 `MANIFEST.sha256` 的 sha256（路径相对 `outputs/wss_deploy_release/`） |
| `vessel_geom_toolkit.sha256` | vessel_geom 工具包 `vessel_geom/*.py` 的 sha256（路径相对工具包根目录） |

## 1. 两个环境

| 环境 | 用途 | 关键版本 | 代码怎么找到它 |
|---|---|---|---|
| **GNN** | `python -m wss_deploy.cli serve / service / run / doctor`，全部测试，模型加载（`training_wss_min`），冻结特征库 `wss_features/` | Python 3.10.19，torch 2.5.1+cu118（CUDA 11.8），torch-geometric 2.6.1（torch-scatter / torch-sparse / torch-cluster / pyg-lib 为 pt25cu118 构建），numpy 1.26.4，scipy 1.15.2，vtk 9.3.1，pyvista 0.48.4，pytest 9.1.1 | 用哪个解释器启动就是哪个；`service start` 默认沿用当前解释器（可 `--python` 指定，写入 `service.json`） |
| **GNN_vmtk** | `centerline.py` 以子进程调用 vessel_geom（`--preset frozen-aortoiliac`），工作目录是工具包根目录 | Python 3.10.14，vmtk 1.5.0，vtk 9.2.6（conda-forge 构建），numpy 1.26.4 | `paths.py`：`VMTK_PYTHON = Path.home() / ".conda/envs/GNN_vmtk/bin/python"`，可用 `WSS_DEPLOY_VMTK_PYTHON` 覆盖 |

两个 yml 都是原样导出，最后一行 `prefix:` 是本机路径；用 `-n` 建环境时会被忽略。

**重建 GNN**：conda 部分走清华镜像，pip 部分里带 `+cu118` / `+pt25cu118` 的轮子不在 PyPI 上，要额外给出两个索引（pip 会读这两个环境变量）：

```bash
PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu118 \
PIP_FIND_LINKS=https://data.pyg.org/whl/torch-2.5.1+cu118.html \
conda env create -n GNN -f wss_deploy/env/GNN.yml
```

显卡驱动需支持 CUDA 11.8（驱动 ≥ 520）；没有 GPU 也能跑（CPU 一例约 100 s）。

**重建 GNN_vmtk**：这个环境当初是用离线显式清单建的（`conda create --offline --file explicit-linux-64-abs.txt`），所以导出的 channel 是 `<unknown>`。
- 首选离线包：`env_packages/GNN_vmtk_env_bundle/`（或 `env_packages/GNN_vmtk_env_bundle.tar.gz`，303 MB，sha256 `6d1fb7fa0a31608f7db107c637847fc6851be2aecfcd06d4ec167aeb8cfe5b3c`；同样被 git 忽略），解压后执行 `bash install_gnn_vmtk_env.sh`，不走依赖求解。
- 联网替代：把 `GNN_vmtk.yml` 里的 `- <unknown>` 一行删掉（这些包实际都来自 conda-forge），再 `conda env create -n GNN_vmtk -f wss_deploy/env/GNN_vmtk.yml`；vmtk 与 vtk 的依赖求解较慢，可能需要放宽个别小版本。

## 2. git 之外必须拷贝的目录

| 目录（相对仓库根） | 内容 | 大小 | 覆盖用的环境变量 |
|---|---|---|---|
| `outputs/wss_deploy_release/X5D_v51_5seed_20260916/` | 默认发布包，模型族 `wall_wss_v1`（峰值 WSS，五 seed） | 15 MB | `WSS_DEPLOY_RELEASE`（默认发布包目录） |
| `outputs/wss_deploy_release/M1_3head_3seed_20260922/` | 模型族 `wall_cycle_multi_v1`（峰值 WSS + TAWSS + OSI，三 seed） | 7.8 MB | `WSS_DEPLOY_RELEASE_ROOT`（发布包所在的父目录，默认即上面这个目录的父目录） |
| `outputs/wss_deploy_release/PF6_VF6_peak_3seed_20260920/` | 模型族 `pf6_vf6_volume_v1`（压力 + 速度体场，三 seed） | 28 MB | 同上 |
| `outputs/vessel_geom_toolkit_2026-09-17/` | vessel_geom 工具包（至少要 `vessel_geom/` 包；`tests/`、`examples/` 可选） | 1.1 MB | `WSS_DEPLOY_VESSEL_GEOM` |
| `outputs/wss_deploy_golden/20260920_baseline/`（可选） | 黄金回归参照，6 个自包含任务（含 1 例 M1 三头） | 69 MB | 由测试的 `WSS_DEPLOY_GOLDEN_REFERENCE` / `WSS_DEPLOY_GOLDEN_JOBS_ROOT` 指定 |

每个发布包目录里：`release.json`（合同）、`MANIFEST.sha256`（每行 `sha256  相对路径  字节数`）、`models/`、`rules/`，以及不进清单、不影响指纹的参考侧车 `reference.json`（缺失时几何越界提示与人群分位不生效，doctor 给 ⚠，可用 `python -m wss_deploy.build_reference_profiles` 重新生成）。
任务目录（默认 `outputs/wss_deploy_jobs/`，`--jobs-root` 可改）运行时自动创建，不用拷贝。

## 3. 获取与核对清单

在新机器的仓库根目录依次执行：

1. **拷贝目录**：`rsync -a <源机器>:<仓库根>/outputs/wss_deploy_release/ outputs/wss_deploy_release/`，vessel_geom 工具包同理（保持目录名，或用上表的环境变量指过去）。
2. **核对发布包清单本身**（确认拿到的是同一份 `release.json` 与 `MANIFEST.sha256`）：
   ```bash
   (cd outputs/wss_deploy_release && sha256sum -c ../../wss_deploy/env/releases.sha256)
   ```
3. **按清单逐文件核对权重**（三个包都应输出 all OK）：
   ```bash
   for d in outputs/wss_deploy_release/*/; do (cd "$d" && awk '{print $1"  "$2}' MANIFEST.sha256 | sha256sum -c --quiet && echo "$d all OK"); done
   ```
   发布包指纹（`release.json` + `MANIFEST.sha256` 的联合哈希，写进每个任务的 `model_release.fingerprint`）应为：
   X5D_v51_5seed_20260916 `bf875ba6ccd8…`、M1_3head_3seed_20260922 `c40326c1c0a0…`、PF6_VF6_peak_3seed_20260920 `e210ca93e4f3…`。
4. **核对 vessel_geom 代码**：
   ```bash
   (cd outputs/vessel_geom_toolkit_2026-09-17 && sha256sum -c ../../wss_deploy/env/vessel_geom_toolkit.sha256)
   ```
   注意：工具包自带的 `MANIFEST.txt` 停在 2026-09-17，之后 09-21 改过 `vessel_geom/cli.py`、`vessel_geom/extract.py`、`README.md` 并新增 `vessel_geom/recenter.py`，
   所以用它核对会有 3 个不一致；以本目录的 `vessel_geom_toolkit.sha256`（2026-09-24 快照，即当前服务实际使用的版本）为准。
5. **环境自检**：`PYTHONPATH=. ~/.conda/envs/GNN/bin/python -m wss_deploy.cli doctor`，应全部 ✓（本机 2026-09-24：12 项全过，未含 service / reports / jobs_root 三组）。
   它会再验一遍每个发布包的 `MANIFEST.sha256`、`GNN_vmtk` 能否导入 vmtk 与 vessel_geom、`wss_features/CONTRACT.json` 源码哈希。
6. **数值核对**（有 GPU 时）：黄金回归应 6/6 通过，约 90 s：
   ```bash
   WSS_DEPLOY_GOLDEN_REFERENCE=$PWD/outputs/wss_deploy_golden/20260920_baseline \
   WSS_DEPLOY_GOLDEN_JOBS_ROOT=$PWD/outputs/wss_deploy_golden/20260920_baseline \
   CUDA_VISIBLE_DEVICES=<空闲卡> PYTHONPATH=. ~/.conda/envs/GNN/bin/python -m pytest -q tests/test_golden_regression.py
   ```

## 4. 上线前加固开关：反向代理 / TLS、umask、登录限速（v0.15，默认不改行为）

网络层开关全部默认关闭：不设置下列变量时，服务行为与 v0.14 相同（登录限速默认值除外，见表末两行）。
变量写进 `service.json` 才会随服务启动：`python -m wss_deploy.cli service restart --env 名=值`（`--unset-env 名` 删除）。

| 变量 | 默认 | 作用 |
|---|---|---|
| `WSS_DEPLOY_TRUST_PROXY` | `0` | `1` 时，**仅当 TCP 对端是 127.0.0.1 / ::1**（本机反向代理）才信任 `X-Forwarded-For`（取最右一个非回环地址）、`X-Forwarded-Proto`（最后一个值，只认 http / https）、`X-Forwarded-Host`；客户端 IP 用于登录限速、审计行与 access.log，协议用于 Origin 校验（只接受代理报告的那个协议）。同时**强制共享模式**：即使监听 127.0.0.1 也必须登录（users.json 要有启用用户，或设了令牌），否则经代理的请求都来自回环，会落进免登录的本机模式。默认下伪造的 `X-Forwarded-*` 一律忽略 |
| `WSS_DEPLOY_COOKIE_SECURE` | `auto` | `auto`：仅当请求经可信代理且 `X-Forwarded-Proto: https` 时会话 Cookie 加 `Secure`；`1`：总是加（纯 http 访问时浏览器将不保存 Cookie，无法登录）；`0`：从不加。判定为 https 的请求另带 `Strict-Transport-Security: max-age=31536000`（不含 includeSubDomains / preload）；非 https 从不发 HSTS |
| `WSS_DEPLOY_UMASK` | 未设置（不改） | 八进制，如 `077`：`serve` 启动时先 `os.umask`，之后该进程创建的任务目录、报告、导出文件都不给组 / 其他人权限。与之无关，`users.json`、`.sessions.json`、`.service_token`、`service.json`、所有 JSON 原子写入、`server.log` / `access.log`（含轮转出的新文件）、`server.console.log`、`deleted_jobs.jsonl` 在任何 umask 下都以 0600 创建（已有的日志文件在服务打开时收窄为 0600） |
| `WSS_DEPLOY_LOGIN_RATE_PER_MIN` | `30`（v0.14 为 10） | 每个客户端地址每分钟登录 / 改口令尝试次数（成功的尝试退还）。课题组共用一个出口 IP 时，一个人输错不再挡住其他人 |
| `WSS_DEPLOY_LOGIN_USER_RATE_PER_MIN` | `10` | 每个用户名每分钟尝试次数（成功退还）。其内还有 users.py 的失败锁定：同一用户名 60 s 内错 5 次锁定到最早一次失败满 60 s。三者超限都返回 429，响应体带 `retry_after`（秒）并有 `Retry-After` 头；每个地址 / 用户名每分钟最多记一条 `login_throttled` 审计行 |

**接 nginx 的步骤**（用户拍板后再做；示例见同目录 `nginx.example.conf`）：

1. 确认已有启用的用户名账号（`python -m wss_deploy.cli user list`），否则第 2 步后服务起不来（共享模式需要账号或令牌）。
2. `python -m wss_deploy.cli service restart --host 127.0.0.1 --env WSS_DEPLOY_TRUST_PROXY=1`（只听回环，外部必须走 nginx）。
3. 按 `nginx.example.conf` 配置证书、`client_max_body_size 257m`、SSE 路径 `proxy_buffering off` 与长 `proxy_read_timeout`；`Host` / `X-Forwarded-Host` 用 `$http_host`（带端口，否则 Origin 校验失败）。
4. 验收：浏览器经 https 登录后，开发者工具里 `wss_session` 带 `Secure`、响应有 `Strict-Transport-Security`；`access.log` 记录的是真实客户端 IP 而不是 127.0.0.1。

注意：开了 `WSS_DEPLOY_TRUST_PROXY` 后，本机上任何能连 127.0.0.1:8765 的账号都可以自填 `X-Forwarded-For` 绕开按地址的限速（按用户名的限速与失败锁定仍然有效）；服务与 nginx 在同一台机器时这是可接受的取舍。
