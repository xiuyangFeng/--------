# 稳态每个独立损失项（2026-09-13）

沿用旧V4、seed1234稳态8臂与原日志，全部英文绘图。每张图上排为未乘损失权重的raw，下排为加权后对总目标的贡献；左PN、右PNPP。同排两骨干统一纵轴，纵轴为log。淡线原始逐epoch均值，实线51轮居中移动均值。无新增训练/评估。

[11页合订PDF](individual_losses_all.pdf)。各图均有同名PNG/PDF/SVG。

| 图 | 独立项 | 绘制的实验臂 |
| --- | --- | --- |
| [01](01_data_u.png) | u速度数据MSE | 全4臂 |
| [02](02_data_v.png) | v速度数据MSE | 全4臂 |
| [03](03_data_w.png) | w速度数据MSE | 全4臂 |
| [04](04_data_p.png) | 压力数据MSE | 全4臂 |
| [05](05_no_slip.png) | 壁面无滑移 | BC、固定PDE、EMA |
| [06](06_inlet_bc.png) | 入口边界 | BC、固定PDE、EMA |
| [07](07_continuity.png) | 连续性 | 固定PDE、EMA |
| [08](08_momentum_x.png) | x动量 | 固定PDE、EMA |
| [09](09_momentum_y.png) | y动量 | 固定PDE、EMA |
| [10](10_momentum_z.png) | z动量 | 固定PDE、EMA |
| [11](11_momentum_group.png) | 三方向动量均值，补充汇总项 | 固定PDE、EMA |

实际独立项为前10项，第11图为动量组合项，不可与三个分量重复计入总损失。数据raw为标准化MSE，边界与PDE raw为缩放后的无量纲MSE，不是有量纲残差RMS。

data各通道权重1/4；稳态no-slip/inlet各为lambda_BC/2；continuity为lambda_PDE/2；momentum各方向为lambda_PDE/6，momentum_group为lambda_PDE/2。实际绘制日志mean_*_weighted，不用epoch平均lambda乘epoch平均raw近似，因为两者乘积的平均一般不等于平均的乘积。

稳态没有启用RCR损失和瞬态时间项，因此不画全零占位线；DATA臂中边界/PDE关闭，BC臂中PDE关闭，同样不画成“已满足物理”。这些缺席不是漏图。开启项偶有零值时，仅在log显示中遮蔽，不加epsilon；数量记录于CSV。

[统计CSV](individual_loss_summary.csv)：每个项、每个视图、各run的最后500轮原始均值、零值数和字段名。
[验证及来源](verification.json)：全部8臂epoch连续唯一、每个开启项非负有限；逐epoch前10个加权项之和与mean_total校验通过（rtol=1e-5、atol=1e-6）。

复现：在仓库根目录运行 `python3 docs/03-汇报材料/V4汇报/V4_稳态梯度与损失_2026-09-12/individual_losses/plot_individual_losses.py`。

## 按最新要求简化为单实验两张图

[PointNet++ EMA 单实验两图](single_pnpp_ema/README.md)：一张u/v/w/p，一张no-slip/inlet/continuity/三方向momentum，均在同一坐标轴展示未加权损失。
