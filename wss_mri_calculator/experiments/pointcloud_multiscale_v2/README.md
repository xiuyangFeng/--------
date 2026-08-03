# Point-cloud multi-scale WSS v2

本实验只使用壁面/内部点云坐标、壁面法向和内部速度 `u/v/w`，不读取 CFD 网格拓扑，
也不使用 WSS 真值决定任何局部修正。

v2 保留 v1 adaptive-CV 的梯度方向，在多个稳定邻域上估计有限深度梯度，外推到
零采样深度，并把外推结果转换为经过壁面点云局部中值正则的正值幅度修正。

冻结配置见 [`config_frozen_v2.json`](config_frozen_v2.json)，正式结果见
[`RESULTS.md`](RESULTS.md)。核心实现位于 `src/wss_multiscale.py`，验证入口为
`src/compare_multiscale_v2.py`。

单病例使用：

```bash
cd wss_mri_calculator/src
python calculate_wss_cfd.py \
  --case-dir <病例目录> \
  --neighbor-mode multiscale_v2
```

使用全壁面 train138 推导并冻结的全局标量：

```bash
python calculate_wss_cfd.py \
  --case-dir <病例目录> \
  --neighbor-mode multiscale_v2 \
  --prediction-scale 1.157066322432233
```

v1 仍是默认模式；需要显式指定 `multiscale_v2`，避免改变已冻结的 v1 复现口径。
