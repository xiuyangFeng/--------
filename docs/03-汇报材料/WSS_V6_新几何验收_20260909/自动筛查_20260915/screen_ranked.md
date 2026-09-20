# 几何候选自动筛查排名（172 例）

零标记（可以只看拼图）：51 例。有标记：121 例。

| # | 病例 | 分数 | 原因 |
|---|---|---|---|
| 1 | AAA/ruputer/FENG_LI_XIN | 51 | 截面积>1.5×体网格 12 站(最大 +17.0)；截面积<0.6×体网格 7 站；A/πRmis²>3 16 站；相邻站跳变>1.5× 6 处；非星形轮廓 9 站；点云/面几何分歧 1 站；分叉未定 1；队列离群: pc_valid_frac=0.539(z-3.7), wall_ambiguous_frac=0.255(z+5.3), J_seg1_s_mm=52.8(z+8.2) |
| 2 | AAA/ruputer/GAO_FENG_SHAN | 47 | 截面积>1.5×体网格 8 站(最大 +2.8)；截面积<0.6×体网格 13 站；A/πRmis²>3 14 站；相邻站跳变>1.5× 3 处；非星形轮廓 11 站；点云/面几何分歧 2 站；分叉未定 1；队列离群: pc_valid_frac=0.444(z-5.2), J_seg2_s_mm=44.2(z+7.0) |
| 3 | AAA/ruputer/WANG_FU_SHUN | 41 | 截面积>1.5×体网格 5 站(最大 +4.4)；A/πRmis²>3 5 站；相邻站跳变>1.5× 9 处；非星形轮廓 8 站；队列离群: trunk_tortuosity=1.24(z+3.9) |
| 4 | AAA/ruputer/LI_LAO_PING | 40 | 截面积>1.5×体网格 5 站(最大 +1.5)；截面积<0.6×体网格 3 站；A/πRmis²>3 14 站；相邻站跳变>1.5× 3 处；非星形轮廓 7 站；队列离群: trunk_tortuosity=1.34(z+6.0) |
| 5 | AAA/unruputer/SUN_SHU_MING | 40 | 截面积>1.5×体网格 11 站(最大 +2.7)；截面积<0.6×体网格 2 站；A/πRmis²>3 5 站；相邻站跳变>1.5× 2 处；非星形轮廓 1 站；点云/面几何分歧 2 站；分叉未定 1；队列离群: trunk_tortuosity=1.36(z+6.4), cia_left_tortuosity=1.32(z+3.6), cia_right_tortuosity=1.42(z+5.0), J_seg2_s_mm=43.8(z+6.9) |
| 6 | ILO/AN_GUANG_JIE-0/before | 39 | 截面积>1.5×体网格 5 站(最大 +2.0)；截面积<0.6×体网格 5 站；A/πRmis²>3 9 站；相邻站跳变>1.5× 2 处；非星形轮廓 7 站 |
| 7 | AAA/ruputer/WU_GUANG_CUN | 39 | 截面积>1.5×体网格 8 站(最大 +1.2)；截面积<0.6×体网格 6 站；A/πRmis²>3 9 站；相邻站跳变>1.5× 3 处；非星形轮廓 2 站；点云/面几何分歧 1 站 |
| 8 | ILO/ZHANG_JIN_CHUN-1/before | 38 | 截面积>1.5×体网格 4 站(最大 +2.7)；截面积<0.6×体网格 3 站；A/πRmis²>3 4 站；相邻站跳变>1.5× 5 处；非星形轮廓 5 站 |
| 9 | AAA/unruputer/ZHU_ZI_HAI | 38 | 截面积>1.5×体网格 4 站(最大 +3.9)；截面积<0.6×体网格 5 站；A/πRmis²>3 4 站；相邻站跳变>1.5× 3 处；非星形轮廓 4 站；队列离群: trunk_tortuosity=1.41(z+7.7), cia_left_tortuosity=1.41(z+4.9), inlet_area_mm2=999(z+5.3) |
| 10 | AAA/unruputer/LIU_WEN_QI | 38 | 截面积>1.5×体网格 4 站(最大 +3.9)；截面积<0.6×体网格 5 站；A/πRmis²>3 4 站；相邻站跳变>1.5× 3 处；非星形轮廓 4 站；队列离群: trunk_tortuosity=1.41(z+7.7), cia_left_tortuosity=1.41(z+4.9), inlet_area_mm2=999(z+5.3) |
| 11 | AAA/unruputer/CAO_HONG_TAI | 36 | 截面积>1.5×体网格 5 站(最大 +8.2)；A/πRmis²>3 2 站；相邻站跳变>1.5× 5 处；非星形轮廓 3 站；分叉未定 1；队列离群: trunk_tortuosity=1.23(z+3.6), J_seg1_s_mm=38.5(z+5.0) |
| 12 | ILO/LIU_LIAN_YOU-0/before | 34 | 截面积>1.5×体网格 1 站(最大 +1.6)；截面积<0.6×体网格 1 站；A/πRmis²>3 8 站；相邻站跳变>1.5× 9 处；非星形轮廓 5 站；分叉未定 1；解剖量出界: cia_right_length=15 |
| 13 | ILO/WANG_TIAN_QING-1/before | 34 | 截面积>1.5×体网格 6 站(最大 +1.7)；A/πRmis²>3 1 站；相邻站跳变>1.5× 6 处；非星形轮廓 7 站；点云/面几何分歧 2 站 |
| 14 | AAA/ruputer/WANG_AN | 31 | 截面积>1.5×体网格 1 站(最大 +0.7)；截面积<0.6×体网格 6 站；A/πRmis²>3 6 站；相邻站跳变>1.5× 5 处；非星形轮廓 1 站；队列离群: wall_ambiguous_frac=0.231(z+4.4), J_seg0_s_mm=60.2(z+3.5) |
| 15 | AAA/unruputer/CHEN_FU_YE | 31 | 截面积>1.5×体网格 4 站(最大 +1.8)；A/πRmis²>3 3 站；相邻站跳变>1.5× 3 处；非星形轮廓 3 站；点云/面几何分歧 2 站；队列离群: trunk_tortuosity=1.23(z+3.6), cia_right_tortuosity=1.54(z+6.8) |
| 16 | AAA/ruputer/LIU_YONG_LAN | 27 | 截面积>1.5×体网格 7 站(最大 +1.3)；A/πRmis²>3 4 站；相邻站跳变>1.5× 1 处；非星形轮廓 2 站 |
| 17 | AAA/ruputer/DING_JUN_FENG | 25 | 截面积>1.5×体网格 1 站(最大 +0.9)；A/πRmis²>3 5 站；相邻站跳变>1.5× 6 处；非星形轮廓 2 站 |
| 18 | ILO/WANG_XIAO_LIN-0/before | 25 | 截面积>1.5×体网格 2 站(最大 +2.3)；A/πRmis²>3 1 站；相邻站跳变>1.5× 5 处；非星形轮廓 3 站；点云/面几何分歧 1 站；队列离群: trunk_daughter_angle_deg=34.1(z+4.0), cia_left_parent_angle_deg=30.4(z+4.1), cia_right_tortuosity=1.82(z+10.8) |
| 19 | AAA/ruputer/MENG_GUANG_QIN | 25 | 截面积>1.5×体网格 5 站(最大 +1.0)；截面积<0.6×体网格 1 站；相邻站跳变>1.5× 2 处；非星形轮廓 4 站；队列离群: cia_left_tortuosity=1.42(z+5.1) |
| 20 | ILO/YANG_WEN_TAI-0/before | 24 | 截面积>1.5×体网格 2 站(最大 +17.0)；截面积<0.6×体网格 5 站；A/πRmis²>3 2 站；相邻站跳变>1.5× 2 处；非星形轮廓 3 站；队列离群: trunk_length_mm=338(z+3.7), trunk_tortuosity=1.27(z+4.5) |
| 21 | AAA/unruputer/LIU_KANG_WEN | 24 | 截面积>1.5×体网格 3 站(最大 +2.1)；A/πRmis²>3 3 站；非星形轮廓 3 站；解剖量出界: cia_left_length=150; cia_right_length=134；队列离群: trunk_tortuosity=1.3(z+5.3), cia_left_length_mm=150(z+4.4), J_seg0_s_mm=96.2(z+7.3) |
| 22 | ILO/YANG_WANG_QI-1/before | 23 | 截面积>1.5×体网格 3 站(最大 +1.1)；相邻站跳变>1.5× 7 处；非星形轮廓 4 站 |
| 23 | ILO/GAO_SHU_CAI-0/before | 22 | 截面积>1.5×体网格 2 站(最大 +5.1)；A/πRmis²>3 2 站；相邻站跳变>1.5× 3 处；非星形轮廓 5 站；队列离群: trunk_length_mm=330(z+3.5) |
| 24 | AAA/unruputer/YAN_FU_TANG | 22 | 截面积>1.5×体网格 4 站(最大 +1.6)；A/πRmis²>3 2 站；相邻站跳变>1.5× 2 处；非星形轮廓 1 站；点云/面几何分歧 1 站 |
| 25 | AAA/unruputer/YANG_BEN_RUI | 22 | 截面积>1.5×体网格 3 站(最大 +2.6)；A/πRmis²>3 3 站；相邻站跳变>1.5× 2 处；非星形轮廓 2 站；点云/面几何分歧 1 站 |
| 26 | AG/slow/MI_DE_XI | 20 | 截面积>1.5×体网格 2 站(最大 +1.8)；A/πRmis²>3 1 站；相邻站跳变>1.5× 7 处；非星形轮廓 1 站；点云/面几何分歧 1 站 |
| 27 | AG/slow/GONG_HUI_XIA | 19 | 截面积>1.5×体网格 3 站(最大 +4.4)；A/πRmis²>3 1 站；相邻站跳变>1.5× 2 处；非星形轮廓 2 站；点云/面几何分歧 2 站 |
| 28 | AAA/unruputer/SHEN_CHUN_WANG | 19 | 截面积>1.5×体网格 3 站(最大 +3.9)；A/πRmis²>3 2 站；相邻站跳变>1.5× 1 处；非星形轮廓 2 站；队列离群: trunk_daughter_angle_deg=52.1(z+6.9), cia_left_parent_angle_deg=29.2(z+3.9) |
| 29 | AAA/unruputer/ZHANG_YONG_ZHI | 19 | 截面积>1.5×体网格 1 站(最大 +4.0)；A/πRmis²>3 1 站；相邻站跳变>1.5× 6 处；非星形轮廓 2 站；队列离群: cia_right_tortuosity=1.34(z+3.8), cia_right_daughter_angle_deg=50.5(z+4.3) |
| 30 | AG/slow/QIN_SI_FU | 18 | 截面积>1.5×体网格 2 站(最大 +2.1)；A/πRmis²>3 2 站；相邻站跳变>1.5× 2 处；非星形轮廓 2 站；点云/面几何分歧 1 站；队列离群: cia_left_tortuosity=1.36(z+4.1) |
| 31 | AG/fast/ZHANG_XIU_ZHEN | 18 | 截面积>1.5×体网格 3 站(最大 +1.2)；相邻站跳变>1.5× 2 处；非星形轮廓 1 站；点云/面几何分歧 1 站；队列离群: trunk_tortuosity=1.3(z+5.2), cia_left_tortuosity=1.32(z+3.6), cia_right_tortuosity=1.34(z+3.9) |
| 32 | AAA/ruputer/ZHANG_MAO_JIN | 18 | 截面积>1.5×体网格 3 站(最大 +1.4)；截面积<0.6×体网格 1 站；相邻站跳变>1.5× 1 处；非星形轮廓 5 站；队列离群: cia_right_tortuosity=1.36(z+4.1) |
| 33 | AAA/ruputer/XIE_JIN_QUAN | 16 | 截面积>1.5×体网格 2 站(最大 +4.8)；A/πRmis²>3 2 站；相邻站跳变>1.5× 2 处；非星形轮廓 2 站 |
| 34 | AG/slow/WU_FENG_YAN | 16 | 截面积>1.5×体网格 2 站(最大 +3.2)；A/πRmis²>3 2 站；相邻站跳变>1.5× 2 处；非星形轮廓 2 站 |
| 35 | AAA/unruputer/LIU_XING_GUO | 16 | 截面积>1.5×体网格 3 站(最大 +1.3)；相邻站跳变>1.5× 2 处；非星形轮廓 1 站；分叉未定 1 |
| 36 | AAA/unruputer/GUO_BAO_CHUN | 15 | 截面积>1.5×体网格 2 站(最大 +1.7)；相邻站跳变>1.5× 1 处；非星形轮廓 4 站；队列离群: outlet_area_sum_over_inlet=0.956(z+5.3), ref_valid_frac=0.599(z-3.6), J_seg2_s_mm=38.2(z+5.5) |
| 37 | AG/fast/ZHANG_HAO | 15 | 截面积>1.5×体网格 2 站(最大 +2.4)；A/πRmis²>3 1 站；相邻站跳变>1.5× 2 处；非星形轮廓 2 站；点云/面几何分歧 1 站 |
| 38 | AG/slow/YIN_YU_RONG | 15 | 截面积>1.5×体网格 2 站(最大 +3.7)；A/πRmis²>3 2 站；相邻站跳变>1.5× 1 处；非星形轮廓 2 站；点云/面几何分歧 1 站 |
| 39 | AAA/unruputer/CHEN_SHU_LIN | 15 | 截面积>1.5×体网格 2 站(最大 +2.1)；A/πRmis²>3 1 站；相邻站跳变>1.5× 1 处；非星形轮廓 3 站；点云/面几何分歧 2 站 |
| 40 | ILO/GUO_YU_SHU-0/before | 14 | 截面积>1.5×体网格 2 站(最大 +2.0)；A/πRmis²>3 1 站；相邻站跳变>1.5× 1 处；非星形轮廓 2 站；点云/面几何分歧 1 站；队列离群: trunk_daughter_angle_deg=32.5(z+3.7) |
| 41 | ILO/PAN_YIN_GEN-1/before | 14 | 截面积>1.5×体网格 3 站(最大 +2.1)；A/πRmis²>3 1 站；相邻站跳变>1.5× 1 处；非星形轮廓 1 站 |
| 42 | ILO/WENG_ZHAO_GUANG-0/before | 13 | 截面积>1.5×体网格 1 站(最大 +2.4)；A/πRmis²>3 2 站；相邻站跳变>1.5× 1 处；非星形轮廓 4 站 |
| 43 | AAA/unruputer/GAO_WEI_LI | 13 | 截面积>1.5×体网格 2 站(最大 +1.7)；A/πRmis²>3 1 站；相邻站跳变>1.5× 1 处；非星形轮廓 2 站；队列离群: wall_ambiguous_frac=0.216(z+3.8) |
| 44 | AAA/unruputer/GAO_DIAN_WEN | 13 | 截面积>1.5×体网格 2 站(最大 +1.5)；A/πRmis²>3 1 站；相邻站跳变>1.5× 1 处；非星形轮廓 2 站；点云/面几何分歧 1 站 |
| 45 | AG/slow/WANG_YAN_LING | 13 | 截面积>1.5×体网格 2 站(最大 +2.4)；A/πRmis²>3 2 站；非星形轮廓 1 站；点云/面几何分歧 1 站；队列离群: cia_right_daughter_angle_deg=48.9(z+4.1) |
| 46 | AAA/ruputer/TONG_XUE_LIAN | 13 | 截面积>1.5×体网格 2 站(最大 +1.0)；相邻站跳变>1.5× 2 处；非星形轮廓 1 站；队列离群: cia_left_tortuosity=1.33(z+3.7), cia_right_tortuosity=1.34(z+3.9) |
| 47 | AAA/ruputer/HUO_SHI_LIANG | 13 | 截面积>1.5×体网格 2 站(最大 +2.7)；A/πRmis²>3 1 站；相邻站跳变>1.5× 1 处；非星形轮廓 3 站 |
| 48 | AAA/unruputer/FENG_ZHI_YING | 13 | 截面积>1.5×体网格 1 站(最大 +4.0)；A/πRmis²>3 1 站；相邻站跳变>1.5× 3 处；非星形轮廓 2 站 |
| 49 | ILO/LI_YOU_ZHI-0/before | 12 | 截面积>1.5×体网格 2 站(最大 +3.4)；相邻站跳变>1.5× 2 处；非星形轮廓 1 站；队列离群: J_seg2_s_mm=43(z+6.7) |
| 50 | ILO/LIU_BAO_JUN-0/before | 12 | 截面积>1.5×体网格 1 站(最大 +0.5)；相邻站跳变>1.5× 3 处；非星形轮廓 2 站；队列离群: cia_right_tortuosity=1.42(z+5.0) |
| 51 | AG/slow/ZANG_YU_SHU | 12 | 截面积>1.5×体网格 2 站(最大 +1.4)；A/πRmis²>3 1 站；相邻站跳变>1.5× 1 处；非星形轮廓 2 站 |
| 52 | ILO/XIE_ZHI_FU-0/before | 11 | 截面积>1.5×体网格 1 站(最大 +2.0)；A/πRmis²>3 1 站；相邻站跳变>1.5× 1 处；非星形轮廓 3 站；队列离群: J_seg2_s_mm=30.5(z+3.6) |
| 53 | AG/fast/HAN_JIAN_JUN | 11 | 截面积>1.5×体网格 1 站(最大 +1.2)；A/πRmis²>3 1 站；相邻站跳变>1.5× 1 处；非星形轮廓 1 站；点云/面几何分歧 2 站；队列离群: cia_left_tortuosity=1.5(z+6.2) |
| 54 | ILO/WANG_JIN_MING-0/before | 11 | 相邻站跳变>1.5× 8 处；非星形轮廓 1 站 |
| 55 | AAA/unruputer/CHEN_LIANG_FU | 11 | 截面积>1.5×体网格 1 站(最大 +1.0)；A/πRmis²>3 1 站；相邻站跳变>1.5× 1 处；非星形轮廓 2 站；队列离群: J_seg1_s_mm=41.2(z+5.6), J_seg2_s_mm=34.5(z+4.6) |
| 56 | AG/slow/GUAN_TONG_XIANG | 11 | 截面积>1.5×体网格 3 站(最大 +1.6)；非星形轮廓 2 站 |
| 57 | AG/slow/ZHANG_QING_SHUI | 11 | 截面积>1.5×体网格 1 站(最大 +1.5)；相邻站跳变>1.5× 3 处；队列离群: cia_left_tortuosity=1.88(z+11.8), cia_right_tortuosity=1.75(z+9.8) |
| 58 | AG/fast/FAN_JIAN_MING | 11 | 截面积>1.5×体网格 1 站(最大 +1.1)；相邻站跳变>1.5× 1 处；点云/面几何分歧 1 站；解剖量出界: cia_left_tort=2.23；队列离群: cia_left_tortuosity=2.23(z+17.0), cia_right_tortuosity=1.73(z+9.6) |
| 59 | AG/slow/WANG_GUI | 10 | 截面积>1.5×体网格 1 站(最大 +3.5)；A/πRmis²>3 1 站；相邻站跳变>1.5× 2 处；非星形轮廓 1 站 |
| 60 | ILO/WAGN_LI_JUN-1/before | 10 | 截面积>1.5×体网格 1 站(最大 +1.1)；相邻站跳变>1.5× 2 处；非星形轮廓 1 站；队列离群: cia_right_tortuosity=1.39(z+4.6), J_seg0_s_mm=62(z+3.7) |
| 61 | ILO/GONG_HAI_ZENG-1/before | 10 | 相邻站跳变>1.5× 8 处 |
| 62 | AAA/unruputer/SHEN_FANG_JIN | 10 | 截面积>1.5×体网格 1 站(最大 +2.2)；A/πRmis²>3 1 站；相邻站跳变>1.5× 1 处；非星形轮廓 1 站；点云/面几何分歧 1 站；队列离群: wall_ambiguous_frac=0.22(z+4.0) |
| 63 | ILO/ZHAO_CHANG_SHAN-0/before | 10 | 相邻站跳变>1.5× 4 处；非星形轮廓 2 站 |
| 64 | ILO/YANG_YU_QING-1/before | 10 | 相邻站跳变>1.5× 4 处；非星形轮廓 2 站 |
| 65 | ILO/LIN_CHUN_YANG-1/before | 9 | 截面积<0.6×体网格 4 站；非星形轮廓 4 站；队列离群: outlet_area_sum_over_inlet=0.828(z+4.2) |
| 66 | AG/slow/ZHANG_SONG_TIAN | 9 | 截面积>1.5×体网格 2 站(最大 +1.2)；相邻站跳变>1.5× 1 处；非星形轮廓 1 站 |
| 67 | AAA/unruputer/LIU_LI_FENG | 9 | A/πRmis²>3 1 站；相邻站跳变>1.5× 3 处；非星形轮廓 1 站 |
| 68 | ILO/WANG_SHU_SHENG-0/before | 8 | 相邻站跳变>1.5× 1 处；非星形轮廓 5 站；队列离群: trunk_daughter_angle_deg=38.5(z+4.7) |
| 69 | AG/slow/ZHANG_LING_GUANG | 8 | 截面积>1.5×体网格 1 站(最大 +2.3)；A/πRmis²>3 1 站；相邻站跳变>1.5× 1 处；非星形轮廓 1 站 |
| 70 | AAA/ruputer/LI_ZHEN_HUA | 8 | 相邻站跳变>1.5× 2 处；解剖量出界: cia_left_length=120; cia_right_length=123；队列离群: J_seg0_s_mm=79.5(z+5.6) |
| 71 | ILO/ZHANG_HE_PING-0/before | 7 | 截面积>1.5×体网格 1 站(最大 +4.4)；A/πRmis²>3 1 站；非星形轮廓 2 站 |
| 72 | AG/slow/LU_ZHEN_QING | 7 | 截面积>1.5×体网格 1 站(最大 +1.5)；相邻站跳变>1.5× 2 处 |
| 73 | AG/slow/ZHANG_WEI_XIAN | 7 | 截面积<0.6×体网格 2 站；A/πRmis²>3 1 站；相邻站跳变>1.5× 1 处；非星形轮廓 1 站 |
| 74 | AAA/unruputer/WANG_MAN_TIAN | 7 | 相邻站跳变>1.5× 2 处；非星形轮廓 1 站；队列离群: L2_ref_vs_volume_median_relerr=0.0141(z+3.6), J_seg2_s_mm=33.2(z+4.3) |
| 75 | ILO/CUI_WEI_PING-0/before | 7 | 相邻站跳变>1.5× 1 处；非星形轮廓 10 站 |
| 76 | AAA/unruputer/ZHANG_XUN_LIAN | 6 | 截面积>1.5×体网格 1 站(最大 +0.7)；相邻站跳变>1.5× 1 处；队列离群: L2_ref_vs_volume_median_relerr=0.0145(z+3.7) |
| 77 | AAA/ruputer/LI_BING_JIANG | 6 | 相邻站跳变>1.5× 1 处；非星形轮廓 1 站；分叉未定 1；队列离群: J_seg2_s_mm=31.2(z+3.8) |
| 78 | AG/fast/LI_SHI_QIANG | 5 | 分叉未定 1；解剖量出界: cia_right_length=15 |
| 79 | AAA/unruputer/QIAO_XIU_YUN | 5 | 点云/面几何分歧 4 站；队列离群: trunk_tortuosity=1.29(z+5.0) |
| 80 | AAA/unruputer/LIU_CHENG_ZHANG | 5 | 相邻站跳变>1.5× 2 处；队列离群: trunk_tortuosity=1.23(z+3.5) |
| 81 | AG/slow/KANG_XI_MING | 4 | 相邻站跳变>1.5× 2 处 |
| 82 | AAA/ruputer/GUO_AI_JUN | 4 | 相邻站跳变>1.5× 2 处 |
| 83 | ILO/SUN_XU_XIA-1/before | 4 | A/πRmis²<0.85 3 站；相邻站跳变>1.5× 2 处 |
| 84 | AG/fast/LIU_FENG_MING | 4 | 相邻站跳变>1.5× 2 处 |
| 85 | AG/slow/ZHANG_JING_SHUN | 4 | 相邻站跳变>1.5× 1 处；队列离群: cia_left_tortuosity=1.68(z+8.9), cia_right_tortuosity=1.83(z+11.1) |
| 86 | AAA/ruputer/ZHOU_KE_XUN | 4 | 相邻站跳变>1.5× 1 处；非星形轮廓 2 站 |
| 87 | AAA/ruputer/ZOU_LI_SHUN | 4 | 解剖量出界: cia_left_length=128; cia_right_length=130；队列离群: cia_right_tortuosity=1.39(z+4.6) |
| 88 | AAA/ruputer/ZUO_DAO_SHENG | 4 | 非星形轮廓 2 站；队列离群: wall_valid_frac=0.608(z-4.2), wall_ambiguous_frac=0.252(z+5.2) |
| 89 | AG/slow/WEI_BAO_XING | 4 | 相邻站跳变>1.5× 1 处；点云/面几何分歧 2 站 |
| 90 | AG/slow/HOU_SHEN_QIAN | 4 | 相邻站跳变>1.5× 2 处 |
| 91 | AAA/unruputer/LI_YOU_YU | 3 | 截面积>1.5×体网格 1 站(最大 +0.8) |
| 92 | AAA/unruputer/LIN_JIAN_RONG | 3 | 截面积<0.6×体网格 1 站；队列离群: trunk_daughter_angle_deg=41.4(z+5.2), cia_left_parent_angle_deg=36.1(z+5.3) |
| 93 | AAA/ruputer/YU_TIAN_HAI | 3 | 相邻站跳变>1.5× 1 处；非星形轮廓 1 站 |
| 94 | AAA/unruputer/LIU_JIE | 3 | 相邻站跳变>1.5× 1 处；非星形轮廓 1 站 |
| 95 | AAA/ruputer/KANG_YONG | 3 | 分叉未定 1；队列离群: J_seg2_s_mm=31(z+3.7) |
| 96 | AAA/ruputer/LIU_JI_XIN | 3 | 截面积<0.6×体网格 1 站；相邻站跳变>1.5× 1 处 |
| 97 | AG/slow/CAO_FENG_CHI | 3 | 截面积>1.5×体网格 1 站(最大 +1.8) |
| 98 | AG/slow/LV_FU_LONG | 2 | 非星形轮廓 2 站 |
| 99 | AG/fast/LOU_YANG | 2 | 相邻站跳变>1.5× 1 处 |
| 100 | ILO/LIU_YUE_DONG-0/before | 2 | 相邻站跳变>1.5× 1 处 |
| 101 | AG/slow/DING_LIAN_ZHONG | 2 | 点云/面几何分歧 2 站 |
| 102 | ILO/GUO_QING_SHAN-0/before | 2 | 相邻站跳变>1.5× 1 处 |
| 103 | ILO/LV_JIAN_HUA-0/before | 2 | 非星形轮廓 2 站 |
| 104 | AG/slow/LI_HUAN_GE | 2 | 相邻站跳变>1.5× 1 处 |
| 105 | AG/fast/LI_ZHEN_SHAN | 2 | 非星形轮廓 1 站；队列离群: cia_left_tortuosity=1.38(z+4.4) |
| 106 | AG/fast/LIU_JUN_FENG | 2 | 分叉未定 1 |
| 107 | AAA/unruputer/ZHANG_ZHI_HUA | 2 | 相邻站跳变>1.5× 1 处 |
| 108 | ILO/BAO_EN_YUN-0/before | 2 | 非星形轮廓 1 站；队列离群: cia_left_daughter_angle_deg=57.8(z+4.1) |
| 109 | ILO/HOU_SHI_GUO-0/before | 2 | 截面积<0.6×体网格 1 站；非星形轮廓 1 站 |
| 110 | ILO/YAO_GUO_CHEN-0/before | 2 | 相邻站跳变>1.5× 1 处 |
| 111 | ILO/WU_JUN-0/before | 2 | 队列离群: cia_left_parent_angle_deg=31.2(z+4.3), cia_right_parent_angle_deg=32.8(z+3.7) |
| 112 | AG/slow/SUN_ZONG_GE | 2 | 相邻站跳变>1.5× 1 处 |
| 113 | AG/slow/ZHANG_JUN_HUA | 2 | 非星形轮廓 1 站；点云/面几何分歧 1 站 |
| 114 | AG/slow/HE_SHU_ZHEN | 1 | 队列离群: outlet_area_sum_over_inlet=0.847(z+4.4) |
| 115 | AG/slow/LIU_ZONG_YANG | 1 | 队列离群: trunk_daughter_angle_deg=31.2(z+3.5) |
| 116 | AG/slow/YANG_YOU_SHENG | 1 | 队列离群: cia_right_daughter_angle_deg=51.8(z+4.5) |
| 117 | AG/fast/WANG_DAO_CHUN | 1 | 点云/面几何分歧 1 站 |
| 118 | AG/slow/CHENG_GUANG_SEN | 1 | 队列离群: trunk_daughter_angle_deg=40(z+4.9) |
| 119 | AG/fast/WANG_YONG_FAN | 1 | 队列离群: trunk_daughter_angle_deg=40.9(z+5.1) |
| 120 | ILO/YU_XIANG_SHENG-1/before | 1 | 非星形轮廓 1 站 |
| 121 | AG/fast/CHEN_SHI_MING | 1 | 队列离群: cia_right_tortuosity=1.39(z+4.6) |

## 零标记病例

AAA/unruputer/ZHOU_XI_SHENG, ILO/SUN_DE_QI-1/before, ILO/SUN_DONG_XIN-0/before, AAA/ruputer/WANG_KUI_WU, AAA/ruputer/LIN_LIANG_XIAO, ILO/ZHANG_JIAN_JUN-1/before, ILO/ZHANG_YAN_SHAN-0/before, ILO/ZHANG_YONG_SHENG-0/before, AG/slow/A_MIN_SE_HA_CHI, AG/slow/BAI_WEN_JIE, AG/slow/CHENG_LU_LI, AG/slow/WU_YI_FA, AG/slow/XI_CHENG_JIANG, AG/slow/XU_YI_CAI, AG/slow/QU_HE_PING, AG/slow/SHEN_ZHI_GANG, AG/slow/WANG_GUANG_CHAO, AG/slow/WANG_KE, AG/slow/WANG_YONG_LI, AG/fast/ZHANG_QING_WANG, AG/fast/ZHANG_XIU_WEN, AG/slow/ZHANG_QUAN_YOU, AG/slow/ZHANG_ZHI_JUN, AG/fast/SUN_ZHI_YU, AG/fast/WANG_CHUN_MING, AG/fast/YAO_CUN_HONG, AG/fast/ZHANG_CHUN, AG/fast/ZHANG_LIANG, ILO/GENG_CHUN_LAI-1/before, AG/fast/LIU_LI_QUN, AG/slow/CHEN_JING_RU, AG/slow/GUO_XI_JIANG, AG/slow/MA_TIAN_YI, AG/slow/MA_YU, AG/slow/NI_YAN_BIN, AG/fast/LIU_YI_BING, AG/fast/LI_SHU_KUN, AG/fast/LI_ZHI_LIN, AG/fast/RAN_QING_BO, ILO/LI_SHENG_WEN-0/before, ILO/SUN_YU_SHENG-0/before, AG/slow/LIU_JIN_LIANG, AG/slow/LI_BING_YI, AG/slow/LI_CHONG_ZENG, AG/slow/LI_GUI_YING, ILO/LU_FU_SHAN-0/before, AAA/unruputer/MA_JIN_HE, AAA/unruputer/HAN_JIAN_FU, AAA/ruputer/MA_XIAO_DONG, AG/slow/LIU_FA_CHUN, AG/slow/LIU_FENG
