# VF6：管内点云

每个病例一个 `.vtp`，只保留距壁 **> 1 mm** 的体单元中心。没有切开；需要看截面时在 ParaView 里自己 Clip。

Representation 选 **Points**，颜色选 `speed_cfd` 或 `speed_pred`。

| 病例 | 点数 | 文件 |
| --- | ---: | --- |
| Best · ZHANG_LIANG | 210,412 | [interior.vtp](best/VF6__best__AG__fast__ZHANG_LIANG__interior.vtp) |
| Median · LI_HUAN_GE | 146,581 | [interior.vtp](median/VF6__median__AG__slow__LI_HUAN_GE__interior.vtp) |
| Worst · SUN_SHU_MING | 263,396 | [interior.vtp](worst/VF6__worst__AAA__unruputer__SUN_SHU_MING__interior.vtp) |

正式 R² 和 selfmax 分母仍用完整域。
