# Point-cloud adaptive WSS v1

## 目标

只使用壁面点云、内部单元中心坐标和 `u/v/w`，不使用 CFD 网格连接关系，降低
velocity→WSS 相对 CFD wall-shear 真值的误差。正式病例池固定为 WSS-PINN 的
`train138 + test35` split。

## 不变量

- 壁面法向：局部 PCA，并用附近内部点定向到流体侧；
- 速度边界条件：静止壁面无滑移；
- 流变：与 Fluent UDF 一致的 Carreau–Yasuda；
- 时间步：默认使用 `data_wss_min/bundle.npz` 的 `peak_step`；
- 局部邻域选择不得读取 CFD WSS 真值；
- 所有超参数先在 train 上确定，冻结后再运行 test；
- baseline 与 candidate 必须使用相同病例、时间步、随机种子和壁面抽样点。

## 方法

Baseline：固定 `K=64`、二次无滑移剖面。

冻结 Candidate：对 `K={16,20,24,28,32,36,40,44,48,64}` 分别拟合归一化剖面

```text
u_t(z) = b1*z + b2*z^2,  z = eta / median(eta)
```

用相对 leave-one-out 速度重建误差选择局部邻域。在接近最小误差的候选中取最小 K，
避免过大的邻域跨出近壁边界层。选择过程只看坐标和速度，不看 WSS 真值。

train138 冻结的容忍系数为 `1.25`。最终结果见 [RESULTS.md](RESULTS.md)。

## 实验阶段

1. `stage_a_train_subset`：train 子集扫描固定 K 与 CV tolerance；
2. `stage_b_train138`：全 train138 验证并冻结配置；
3. `stage_c_split173`：冻结配置后，对 train138+test35 做最终成对比较。

结果 JSON、命令和摘要均保存在本目录，不更新 `docs/02-推进与变更/`。

