# 单个PointNet++稳态实验：两张损失图

按用户要求仅展示 `V4-SP-PNPP-BC-PDE-EMA-s1234`，选择EMA臂以对应此前动态权重讨论，未按测试成绩筛选。

- [u/v/w/p数据损失](01_uvwp.png)：同一坐标轴4条曲线。
- [边界与PDE分项](02_bc_pde.png)：同一坐标轴6条曲线，即no-slip、inlet、continuity、momentum x/y/z。动量本身属于PDE，不再将PDE总和作为额外独立项重复展示。
- [两页合订PDF](single_pnpp_losses.pdf)。各图另有PDF/SVG。

全部是未加权的逐epoch平均损失，51轮居中移动均值，淡线保留原始值。数据项是标准化MSE，物理项是缩放后无量纲MSE；曲线大小不能直接解释为加权总损失贡献或梯度强弱。只使用现有日志，没有新增实验。横轴从epoch0至9984，不外推至10000。

来源：`/public/newhome/cy/Digital_twin/GNN/outputs/wss_pinn/volume_uvwp_bc_rcr_v4/steady_peak/V4-SP-PNPP-BC-PDE-EMA-s1234/epoch_progress.jsonl`。字段与完整原始曲线见 `source_curves.csv`。复现脚本位于上一级 `plot_single_pnpp.py`。
