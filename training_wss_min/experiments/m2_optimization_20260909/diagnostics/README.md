# M2 完整捕获的 CPU 审查与原壁面图

单seed1234、已暴露test34的训练后机制诊断；激活与门控分布不是因果重要性。三例在本轮结果出现前固定，不能代表总体收益。图显示原始壁面全部顶点，未重建/平滑几何。门控作用在固定5000个support点；解码熵和查询残差覆盖完整壁面。

GPU捕获作业13982：FAILED / 1:0，完成42×34次推理；捕获strict_passed=False。
CPU finalizer作业13984重算保存数组的225,624项逐病例指标并绘图；没有再次推理。
原容差严格通过225,622项，原严格失败2项，后验边界审核2项，未审查失败0项。

原容差核验不代表逐位相等；strict_passed=false和所有原失败条目仍保留。每项后验审核绑定特定捕获、病例、空间、字段、逐点probe与来源SHA；原指标、保护线和工作簿数值未改变。

[best原壁面对照](wall_figures/wall_overview_best.png) · [last原壁面对照](wall_figures/wall_overview_last.png)

geometry和predictions中的文件是同一次GPU捕获的原数组副本/硬链接。源diagnostics_collection及失败历史记录保持不变。
图中候选为MO-S2（raw R²最高，未过完整门槛），不替换历史M2或改变筛选。
