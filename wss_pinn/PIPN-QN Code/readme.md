
      
          
**总体概述**
- 项目目的：基于点云的物理信息神经网络（PINN）/残差PINN（rPINN），用于血流CFD场的时空预测，输出速度分量（u/v/w）、压力（p）与壁面剪切应力（wss），并通过数据项与物理约束（PDE残差、连续性、边界条件）联合优化。
- 核心思想：PointNet 提取点云时空特征（输入为 `x,y,z,t`），分支回归物理量；可选残差结构 `Qres_net` 与系数缩放；训练同时最小化数据误差（p/wss）与物理损失。

**主要入口**
- 训练入口：
  - `Pinn_Cfd_Blood_model3.py` → `main()`：大点数（`point_number=1500`、`batch_sizes=[15000]`），保存到 `N/` 目录。
  - `Pinn_Cfd_Blood_model2.py` → `main()`：同样为1500点数配置，保存到 `N/` 或同路径设定。
  - `Pinn_Cfd_Blood_pv2.py` → `main()`：小点数（`point_number=50`、`batch_sizes=[500]`），保存到 `weight/` 目录。
- 推理/评估入口：
  - `Pinn_Test.py` → `main()`：加载 `weight/PINN_CFD_BLOOD_weight_1500_res.pth` 等权重，对一个样本进行时序预测与误差统计、绘图。
  - `pinn_wss_test.py` → `__main__`：加载 `weight/PINN_CFD_BLOOD_weight_1500.pth` 等，逐时刻批量预测并输出MSE/MAE及可视化。
- 数据生成入口：
  - `dataset.py` → `__main__`：从 `pv_model/` 与 `test_data/` 目录的 `*.txt` 构建训练集 `pv_model_50.pkl` 与测试集 `test_model_data.pkl`。

**模型组件**
- `pointnet_wss.py`
  - `get_model(ues_coef, use_res, p_res, wss_res)`：PointNet骨干（两段MLP+全局最大池化）+ 5 个分支输出 `[u,v,w,p,wss]`。
  - `use_res=True` 时采用 `Qres_net` 残差结构分别用于 `p/wss` 分支；`use_res=False` 使用普通两层MLP `dnet`。
  - 可学习系数 `nn.Parameter`：`p` 与 `wss`；若 `ues_coef=True`，在激活前施加系数缩放。
- `Pinn_Cfd_Blood_*`
  - 训练管线：设置优化器（Adam，分阶段 `learning_rates=[3e-4,1e-4,1e-5]`，`epochs=[500,500,500]`），批次训练、累计并归档损失。
  - 损失记录键：`total, f_loss, wss_loss, loss_equ, loss_continuous, loss_bc`；并记录系数历史 `p_coef, wss_coef`。
  - 最佳模型保存：`Pinn_Cfd_Blood_pv2.py` 保存至 `weight/PINN_CFD_BLOOD_weight_50.pth` 与历史 `weight/history_50.csv`；`Pinn_Cfd_Blood_model3.py` 保存至 `N/PINN_CFD_BLOOD_weight_resa.pth` 与历史 `N/history_resa.csv`。

**数据与格式**
- 训练数据 `pv_model_50.pkl`：列表结构，元素为 `[pressure(N×T), u(N×T), v(N×T), w(N×T), wss(N×T), xyz(N×3)]`；`T=period=10`。
- 原始数据来源目录（需自行准备）：`pv_model/` 与 `test_data/` 下的 `*.txt`（通过 `dataset.py::read_ti` 读取并拼装）。
- 采样/过滤：保留 `datesets[:,6]!=0`，按值域对 `wss` 进行阈值点裁剪等；包含简单归一化/反归一化函数。

**训练/推理脚本行为**
- 训练：从 `pv_model_50.pkl` 读取全人群数据（`people=50`），按设定批次、学习率、权重进行迭代；动态打印当前学习率与各损失；当 `total` 损失改善时保存权重。
- 推理：对单个样本逐时刻构造时间 `t=(k+1)*0.08` 并与坐标拼接为 `xyzt` 后送入模型；统计每时刻的 `MSE/MAE`，输出均值曲线与3D点云可视化（压力与wss的真实/预测/误差）。

**依赖**
- 深度学习：`torch`, `torch.nn`, `torch.nn.functional`
- 科学计算与数据处理：`numpy`, `pandas`, `sklearn.metrics`
- 可视化：`matplotlib.pyplot`
- 其他：`pickle`, `csv`, `glob`, `math`, `datetime`, `os`
- 未提供 `requirements.txt`；需用户自备GPU环境（脚本默认 `cuda:0`）。

**输出与文件**
- 权重文件：`weight/PINN_CFD_BLOOD_weight_50.pth`，`weight/PINN_CFD_BLOOD_weight_1500[_res/_a/_r].pth`，或 `N/PINN_CFD_BLOOD_weight_resa.pth`。
- 训练日志：`weight/history_50.csv`，`N/history_resa.csv`。
- 可视化：推理脚本内绘图（默认不保存，示例保存代码注释在 `pinn_wss_test.py`）。

**快速使用**
- 先生成数据：`python dataset.py`（确保 `pv_model/` 与 `test_data/` 有所需 `*.txt`）。
- 训练（小样本）：`python Pinn_Cfd_Blood_pv2.py`。
- 训练（大样本）：`python Pinn_Cfd_Blood_model3.py` 或 `python Pinn_Cfd_Blood_model2.py`。
- 推理评估：`python Pinn_Test.py` 或 `python pinn_wss_test.py`（确认 `weight/` 或 `N/` 中的 `.pth` 权重路径存在）。
        
          
            
filePath: /Users/xiuyang/研究生学习/文献阅读/组内PINN代码/代码/PIPN-QN Code/pinn_loss_wss.py
          
**总体概述**
- 项目是一个基于点云输入的物理信息神经网络（PINN/rPINN），用于血流CFD场的时空预测与评估，预测变量包含速度分量`u/v/w`、压力`p`和壁面剪切应力`wss`。
- 输入为每个点的空间坐标与时间`[x,y,z,t]`，网络以PointNet式特征汇聚为骨干，并在`p/wss`分支中支持特殊残差结构（Qres）与可学习系数缩放。
- 训练同时最小化数据项（`p/wss`）与物理约束（动量方程残差、连续性、边界条件），并按阶段调整学习率与批大小。

**主要入口**
- 训练（小样本，N=50，批500）在`Pinn_Cfd_Blood_pv2.py:631-664`的`main()`中配置并调用类`Pinn_Cfd_Blood.train`；权重与历史保存到`weight/`（`Pinn_Cfd_Blood_pv2.py:619-629`）。
- 训练（大样本，N=1500，批15000）在`Pinn_Cfd_Blood_model3.py:631-664`的`main()`中进行；权重与历史保存到`N/`（`Pinn_Cfd_Blood_model3.py:619-629`）。
- 推理评估（曲线与点云图）：
  - 单样本时序评估与绘图在`Pinn_Test.py:282-311`，加载权重于`Pinn_Test.py:301`。
  - 逐时刻批量预测与指标输出在`pinn_wss_test.py:298-479`，模型加载于`pinn_wss_test.py:311-315`。

**模型架构**
- 主模型类`get_model`定义于`pointnet_wss.py:6-116`，PointNet特征抽取两段MLP后做全局最大池化与广播（`pointnet_wss.py:131-147`），再走五个输出分支分别预测`u/v/w/p/wss`（速度分支：`pointnet_wss.py:144-155`）。
- `p`与`wss`分支可选：
  - 使用Qres残差结构：`pointnet_wss.py:158-167`，其残差核心单元`Qres`为`h1⊙h2 + (h1 + b)`（`pointnet_wss.py:232-251`）。
  - 使用普通两层MLP：`pointnet_wss.py:163-167`，对应`dnet`顺序前馈（`pointnet_wss.py:174-196`）。
- 学习型系数参数：`p`与`wss`在`pointnet_wss.py:64,68`以`nn.Parameter`注册，用于前向中系数缩放与残差分支的调制。
- 简易损失模块`get_loss`提供MSE/MAE（`pointnet_wss.py:253-279`）。
- 另有一个简化版PointNet在`pointnet.py:6-46`，输出为5维合并分支，供参考或早期版本。

**训练与损失**
- 训练主类`Pinn_Cfd_Blood`在两脚本中结构一致（`Pinn_Cfd_Blood_pv2.py:31-96`，`Pinn_Cfd_Blood_model3.py:31-96`），设置学习率/轮次/批大小与损失权重，采用Adam优化器（`set_optimizers`于`Pinn_Cfd_Blood_pv2.py:515-521`）。
- 数据损失仅对`p/wss`分支按权重组合（`Pinn_Cfd_Blood_pv2.py:357-361`）。
- 物理项：
  - 边界条件速度模长平方和`fun_u_b`（`Pinn_Cfd_Blood_pv2.py:113-127`）。
  - 连续性约束`u_x+v_y+w_z`（`Pinn_Cfd_Blood_pv2.py:98-112`，应用于`406-408`）。
  - 动量方程残差`fun_xy/fun_z`（`Pinn_Cfd_Blood_pv2.py:146-162`与`129-145`），一阶/二阶导数通过`autograd`自动微分（`Pinn_Cfd_Blood_pv2.py:298-311`与在`369-392`中调用）。
- 总损失为数据项与PINN项加权和（`Pinn_Cfd_Blood_pv2.py:409-414`）；历史记录包含各分项与系数（`Pinn_Cfd_Blood_pv2.py:553-572`）。

**数据来源**
- 训练/测试数据由`dataset.py:138-217`构建，读取`pv_model/`与`test_data/`中的`*.txt`，每人多时刻拼接压力/速度/WSS与坐标，形成列表元素`[pressure(N×T), u, v, w, wss, xyz(N×3)]`，并保存为`pv_model_50.pkl`与`test_model_data.pkl`。
- 文本读取函数`read_ti`见`dataset.py:14-34`；数据过滤与采样在`dataset.py:58-91`。
- 训练脚本内部使用固定统计参数做归一化/反归一化（例如`Pinn_Cfd_Blood_pv2.py:444-449`与`compute_fun_loss`中的`335-337`）。

**评估与可视化**
- `Pinn_Test`在每个时刻构造`time=(idx+1)*0.08`与`xyzt`并预测，输出MSE/MAE均值与压力/WSS对比曲线（`Pinn_Test.py:227-281`）。
- `pinn_wss_test.py`提供更详细的分时段预测、误差统计与3D点云可视化（WSS与压力真实/预测/误差）（`pinn_wss_test.py:425-471`、误差统计于`pinn_wss_test.py:371-396`）。
- 训练历史对比与参数敏感性绘图在`pinn_loss_wss.py:52-114`、`115-236`与`239-276`。

**依赖与运行**
- 主要依赖：`torch`、`numpy`、`pandas`、`sklearn.metrics`、`matplotlib`、`pickle`。默认设备为`cuda:0`（例如`Pinn_Test.py:298`、`Pinn_Cfd_Blood_pv2.py:74`）。
- 快速流程：
  - 生成数据：`python dataset.py`（确保`pv_model/`与`test_data/`存在对应`*.txt`）。
  - 训练（N=50）：`python Pinn_Cfd_Blood_pv2.py`。
  - 训练（N=1500）：`python Pinn_Cfd_Blood_model3.py`。
  - 推理评估：`python Pinn_Test.py` 或 `python pinn_wss_test.py`（修改权重路径如`weight/PINN_CFD_BLOOD_weight_1500_res.pth`）。

