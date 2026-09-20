# P6 沿程稳定性自动审查摘要（2026-09-15）

读取 170 个病例，排除重复病例 HOU_SHEN_QIAN, LIU_WEN_QI；P6 有效基础截面 3549 个。

## 自动放行候选

共 **0** 例：无

## 需要人工确定

共 **170** 例。触发条件包括 gate flip、闭合轮廓/交叉组变化、非局部 crossing、源基线无效、无可比面积或面积变化 p95≥20%；p95 在 10–20% 之间也不会自动放行。

- `ILO/WANG_XIAO_LIN-0/before`：p95=0.4709711492890032，invalid=23，gate_flip=30，contour_change=19，group_change=19，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid;area_p95_ge_20pct
- `AG/fast/HAN_JIAN_JUN`：p95=0.36982408101712966，invalid=25，gate_flip=28，contour_change=13，group_change=13，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_20pct;source_baseline_invalid
- `ILO/ZHANG_JIN_CHUN-1/before`：p95=0.17240929447217548，invalid=18，gate_flip=23，contour_change=16，group_change=16，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid;area_p95_ge_10pct
- `AG/slow/CHEN_JING_RU`：p95=0.13259582762455235，invalid=23，gate_flip=25，contour_change=14，group_change=14，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_10pct;source_baseline_invalid
- `AG/slow/ZHANG_LING_GUANG`：p95=0.09064847364936818，invalid=23，gate_flip=27，contour_change=12，group_change=12，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid
- `AG/slow/WEI_BAO_XING`：p95=0.12752313690463848，invalid=24，gate_flip=26，contour_change=12，group_change=12，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid;area_p95_ge_10pct
- `AG/fast/LOU_YANG`：p95=0.1601469240651161，invalid=21，gate_flip=26，contour_change=11，group_change=11，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid;area_p95_ge_10pct
- `AAA/unruputer/ZHANG_XUN_LIAN`：p95=0.07982039017823121，invalid=23，gate_flip=25，contour_change=12，group_change=12，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid
- `AAA/unruputer/CAO_HONG_TAI`：p95=0.23515069808964303，invalid=22，gate_flip=25，contour_change=11，group_change=11，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid;area_p95_ge_20pct
- `AAA/unruputer/WANG_MAN_TIAN`：p95=0.23442577745055265，invalid=21，gate_flip=23，contour_change=13，group_change=13，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid;area_p95_ge_20pct
- `ILO/LIU_BAO_JUN-0/before`：p95=0.20329357049142552，invalid=24，gate_flip=24，contour_change=12，group_change=12，nonlocal=0；source_baseline_invalid;gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_20pct
- `AAA/unruputer/LIU_LI_FENG`：p95=0.20073906212468662，invalid=21，gate_flip=26，contour_change=10，group_change=10，nonlocal=0；source_baseline_invalid;gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_20pct
- `AG/fast/YAO_CUN_HONG`：p95=0.15509122309249648，invalid=26，gate_flip=26，contour_change=10，group_change=10，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_10pct
- `AAA/unruputer/SUN_SHU_MING`：p95=0.12147247082354348，invalid=19，gate_flip=23，contour_change=13，group_change=13，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid;area_p95_ge_10pct
- `ILO/LV_JIAN_HUA-0/before`：p95=0.09461733055764265，invalid=15，gate_flip=23，contour_change=12，group_change=12，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid
- `ILO/YANG_YU_QING-1/before`：p95=0.2646106683137886，invalid=20，gate_flip=21，contour_change=13，group_change=13，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid;area_p95_ge_20pct
- `AG/slow/MI_DE_XI`：p95=0.22240554691225134，invalid=16，gate_flip=22，contour_change=12，group_change=12，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_20pct;source_baseline_invalid;area_p95_ge_10pct
- `ILO/GUO_YU_SHU-0/before`：p95=0.10782952203973763，invalid=16，gate_flip=24，contour_change=10，group_change=10，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid;area_p95_ge_10pct
- `AG/slow/WANG_YONG_LI`：p95=0.09129029907235368，invalid=26，gate_flip=26，contour_change=8，group_change=8，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change
- `AAA/unruputer/ZHANG_YONG_ZHI`：p95=0.6059067544903508，invalid=17，gate_flip=21，contour_change=12，group_change=12，nonlocal=0；source_baseline_invalid;gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_20pct;area_p95_ge_10pct
- `AG/slow/KANG_XI_MING`：p95=0.16969755383669036，invalid=19，gate_flip=25，contour_change=8，group_change=8，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid;area_p95_ge_10pct
- `AAA/unruputer/CHEN_FU_YE`：p95=0.1652115752593764，invalid=21，gate_flip=22，contour_change=11，group_change=11，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid;area_p95_ge_10pct
- `AAA/unruputer/QIAO_XIU_YUN`：p95=0.15132045922454906，invalid=22，gate_flip=23，contour_change=10，group_change=10，nonlocal=0；area_p95_ge_10pct;source_baseline_invalid;gate_flip_or_became_invalid;closed_contour_change;crossing_group_change
- `AG/fast/WANG_YONG_FAN`：p95=0.11831844822481906，invalid=19，gate_flip=23，contour_change=10，group_change=10，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_10pct;source_baseline_invalid
- `ILO/ZHANG_YONG_SHENG-0/before`：p95=0.09207819604280168，invalid=24，gate_flip=24，contour_change=8，group_change=8，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change
- `AG/slow/WU_FENG_YAN`：p95=0.08999657372173712，invalid=24，gate_flip=24，contour_change=8，group_change=8，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid
- `AG/slow/WANG_GUI`：p95=0.21661989576033142，invalid=23，gate_flip=24，contour_change=7，group_change=7，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid;area_p95_ge_20pct
- `AAA/unruputer/LIU_XING_GUO`：p95=0.1992918745663566，invalid=16，gate_flip=21，contour_change=10，group_change=10，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid;area_p95_ge_10pct
- `AG/slow/LU_ZHEN_QING`：p95=0.1726464060205736，invalid=22，gate_flip=23，contour_change=8，group_change=8，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_10pct;source_baseline_invalid
- `AAA/unruputer/ZHOU_XI_SHENG`：p95=0.1616385747523948，invalid=23，gate_flip=23，contour_change=8，group_change=8，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_10pct
- `AG/fast/FAN_JIAN_MING`：p95=0.15760764064855895，invalid=25，gate_flip=25，contour_change=6，group_change=6，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid;area_p95_ge_10pct
- `AG/slow/LV_FU_LONG`：p95=0.13873439720932443，invalid=19，gate_flip=24，contour_change=7，group_change=7，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid;area_p95_ge_10pct
- `AG/fast/RAN_QING_BO`：p95=0.1289790073198422，invalid=15，gate_flip=22，contour_change=9，group_change=9，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_10pct;source_baseline_invalid
- `AAA/ruputer/ZHOU_KE_XUN`：p95=0.12423080732257658，invalid=20，gate_flip=24，contour_change=7，group_change=7，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid;area_p95_ge_10pct
- `AG/fast/ZHANG_LIANG`：p95=0.11552367737782226，invalid=22，gate_flip=24，contour_change=7，group_change=7，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_10pct;source_baseline_invalid
- `AG/slow/YANG_YOU_SHENG`：p95=0.1088161638881058，invalid=13，gate_flip=21，contour_change=10，group_change=10，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid;area_p95_ge_10pct
- `AAA/ruputer/WANG_FU_SHUN`：p95=0.29736571547782387，invalid=21，gate_flip=21，contour_change=7，group_change=8，nonlocal=1；source_baseline_invalid;gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_20pct;nonlocal_crossing
- `AG/slow/YIN_YU_RONG`：p95=0.1665754320588602，invalid=18，gate_flip=21，contour_change=9，group_change=9，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_10pct;source_baseline_invalid
- `AG/slow/WANG_GUANG_CHAO`：p95=0.15681697104329828，invalid=21，gate_flip=21，contour_change=9，group_change=9，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_10pct;source_baseline_invalid
- `AG/slow/QIN_SI_FU`：p95=0.15530859603357042，invalid=19，gate_flip=23，contour_change=7，group_change=7，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid;area_p95_ge_10pct
- `ILO/XIE_ZHI_FU-0/before`：p95=0.12758541848432006，invalid=19，gate_flip=21，contour_change=9，group_change=9，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid;area_p95_ge_10pct
- `AG/slow/ZHANG_SONG_TIAN`：p95=0.09995475674538092，invalid=20，gate_flip=23，contour_change=7，group_change=7，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid
- `AG/slow/WANG_YAN_LING`：p95=0.09105022597964746，invalid=20，gate_flip=21，contour_change=9，group_change=9，nonlocal=0；source_baseline_invalid;gate_flip_or_became_invalid;closed_contour_change;crossing_group_change
- `AG/slow/XU_YI_CAI`：p95=0.07303685501230711，invalid=22，gate_flip=22，contour_change=8，group_change=8，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change
- `ILO/ZHAO_CHANG_SHAN-0/before`：p95=0.2252614036925182，invalid=15，gate_flip=19，contour_change=10，group_change=10，nonlocal=0；source_baseline_invalid;gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_20pct
- `ILO/LIN_CHUN_YANG-1/before`：p95=0.15639322950989404，invalid=18，gate_flip=20，contour_change=9，group_change=9，nonlocal=0；source_baseline_invalid;gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_10pct
- `ILO/LIU_LIAN_YOU-0/before`：p95=0.15090833592751068，invalid=15，gate_flip=19，contour_change=10，group_change=10，nonlocal=0；source_baseline_invalid;gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_10pct
- `AG/fast/ZHANG_CHUN`：p95=0.13984336403155057，invalid=22，gate_flip=22，contour_change=7，group_change=7，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_10pct
- `AG/slow/GUO_XI_JIANG`：p95=0.1292463841704095，invalid=22，gate_flip=22，contour_change=7，group_change=7，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_10pct
- `ILO/HOU_SHI_GUO-0/before`：p95=0.11739603053813696，invalid=19，gate_flip=22，contour_change=7，group_change=7，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid;area_p95_ge_10pct
- `AG/fast/ZHANG_XIU_ZHEN`：p95=0.09937359276039329，invalid=22，gate_flip=23，contour_change=4，group_change=5，nonlocal=1；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid;nonlocal_crossing
- `ILO/GUO_QING_SHAN-0/before`：p95=0.09876699505929218，invalid=16，gate_flip=20，contour_change=9，group_change=9，nonlocal=0；source_baseline_invalid;gate_flip_or_became_invalid;closed_contour_change;crossing_group_change
- `AAA/unruputer/CHEN_SHU_LIN`：p95=0.08872026028706467，invalid=21，gate_flip=21，contour_change=8，group_change=8，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid
- `AG/fast/WANG_DAO_CHUN`：p95=0.2951119557184095，invalid=21，gate_flip=22，contour_change=6，group_change=6，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_20pct;source_baseline_invalid;area_p95_ge_10pct
- `ILO/YANG_WEN_TAI-0/before`：p95=0.2104815253973611，invalid=13，gate_flip=18，contour_change=10，group_change=10，nonlocal=0；source_baseline_invalid;gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_20pct
- `AG/slow/ZHANG_JUN_HUA`：p95=0.16726336602033137，invalid=22，gate_flip=22，contour_change=6，group_change=6，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_10pct
- `AAA/ruputer/GUO_AI_JUN`：p95=0.15693666776375254，invalid=19，gate_flip=22，contour_change=6，group_change=6，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid;area_p95_ge_10pct
- `AG/slow/SHEN_ZHI_GANG`：p95=0.15637245979140912，invalid=16，gate_flip=19，contour_change=9，group_change=9，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid;area_p95_ge_10pct
- `ILO/YU_XIANG_SHENG-1/before`：p95=0.13118260204914875，invalid=20，gate_flip=20，contour_change=8，group_change=8，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_10pct
- `AG/fast/LIU_FENG_MING`：p95=0.13012113865419317，invalid=18，gate_flip=21，contour_change=7，group_change=7，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid;area_p95_ge_10pct
- `AAA/unruputer/ZHANG_ZHI_HUA`：p95=0.12366902578991029，invalid=18，gate_flip=21，contour_change=7，group_change=7，nonlocal=0；source_baseline_invalid;gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_10pct
- `AAA/ruputer/WANG_KUI_WU`：p95=0.11869746738829398，invalid=16，gate_flip=18，contour_change=10，group_change=10，nonlocal=0；source_baseline_invalid;gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_10pct
- `AG/slow/A_MIN_SE_HA_CHI`：p95=0.07289422433797944，invalid=18，gate_flip=21，contour_change=7，group_change=7，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid
- `ILO/WAGN_LI_JUN-1/before`：p95=0.2127272915319251，invalid=16，gate_flip=19，contour_change=8，group_change=8，nonlocal=0；source_baseline_invalid;gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_20pct
- `AG/slow/ZHANG_QING_SHUI`：p95=0.20056378355193616，invalid=18，gate_flip=20，contour_change=7，group_change=7，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid;area_p95_ge_20pct
- `AG/slow/WANG_KE`：p95=0.17052162781394203，invalid=15，gate_flip=19，contour_change=8，group_change=8，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid;area_p95_ge_10pct
- `AG/slow/SUN_ZONG_GE`：p95=0.15180307091837436，invalid=21，gate_flip=21，contour_change=6，group_change=6，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_10pct
- `ILO/WENG_ZHAO_GUANG-0/before`：p95=0.11028199320639476，invalid=15，gate_flip=18，contour_change=9，group_change=9，nonlocal=0；source_baseline_invalid;gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_10pct
- `AG/slow/LIU_JIN_LIANG`：p95=0.10999835935665515，invalid=17，gate_flip=20，contour_change=7，group_change=7，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid;area_p95_ge_10pct
- `AG/slow/WU_YI_FA`：p95=0.09398331011235916，invalid=13，gate_flip=18，contour_change=9，group_change=9，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid
- `AG/slow/ZHANG_JING_SHUN`：p95=0.06726167632890082，invalid=16，gate_flip=19，contour_change=8，group_change=8，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid
- `ILO/YANG_WANG_QI-1/before`：p95=0.30231320397312944，invalid=15，gate_flip=19，contour_change=7，group_change=7，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid;area_p95_ge_20pct
- `AAA/ruputer/FENG_LI_XIN`：p95=0.1432323875437997，invalid=18，gate_flip=22，contour_change=4，group_change=4，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid;area_p95_ge_10pct
- `AAA/unruputer/CHEN_LIANG_FU`：p95=0.11843447689036242，invalid=16，gate_flip=18，contour_change=8，group_change=8，nonlocal=0；source_baseline_invalid;gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_10pct
- `AAA/ruputer/LIU_YONG_LAN`：p95=0.11730716265665961，invalid=20，gate_flip=20，contour_change=6，group_change=6，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_10pct
- `AAA/unruputer/GAO_DIAN_WEN`：p95=0.10424034590241466，invalid=20，gate_flip=20，contour_change=6，group_change=6，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid;area_p95_ge_10pct
- `ILO/LI_SHENG_WEN-0/before`：p95=0.0952115177750478，invalid=18，gate_flip=19，contour_change=7，group_change=7，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid
- `ILO/GENG_CHUN_LAI-1/before`：p95=0.08285972975737263，invalid=19，gate_flip=19，contour_change=7，group_change=7，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change
- `AAA/ruputer/WANG_AN`：p95=0.358285845320285，invalid=16，gate_flip=20，contour_change=5，group_change=5，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid;area_p95_ge_20pct;area_p95_ge_10pct
- `AAA/unruputer/LIN_JIAN_RONG`：p95=0.28984712855968964，invalid=11，gate_flip=15，contour_change=10，group_change=10，nonlocal=0；source_baseline_invalid;gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_20pct
- `AAA/ruputer/WU_GUANG_CUN`：p95=0.2170419535741544，invalid=17，gate_flip=19，contour_change=6，group_change=6，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_10pct;source_baseline_invalid;area_p95_ge_20pct
- `AAA/unruputer/YAN_FU_TANG`：p95=0.17356289709231634，invalid=17，gate_flip=17，contour_change=8，group_change=8，nonlocal=0；source_baseline_invalid;gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_10pct
- `ILO/SUN_XU_XIA-1/before`：p95=0.16375276477712156，invalid=14，gate_flip=19，contour_change=6，group_change=6，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid;area_p95_ge_10pct
- `AG/slow/NI_YAN_BIN`：p95=0.15903853152151967，invalid=17，gate_flip=21，contour_change=4，group_change=4，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid;area_p95_ge_10pct
- `AG/slow/LIU_FA_CHUN`：p95=0.13706175746640775，invalid=18，gate_flip=18，contour_change=7，group_change=7，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid;area_p95_ge_10pct
- `AG/fast/LIU_YI_BING`：p95=0.13218306464466997，invalid=21，gate_flip=21，contour_change=4，group_change=4，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_10pct
- `AAA/unruputer/MA_JIN_HE`：p95=0.11660910914350191，invalid=19，gate_flip=19，contour_change=6，group_change=6，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_10pct
- `AG/fast/LI_ZHI_LIN`：p95=0.1007831033050009，invalid=19，gate_flip=20，contour_change=5，group_change=5，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_10pct;source_baseline_invalid
- `AG/slow/LIU_ZONG_YANG`：p95=0.09717442477494309，invalid=13，gate_flip=18，contour_change=7，group_change=7，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid
- `AG/fast/SUN_ZHI_YU`：p95=0.09131408726380041，invalid=19，gate_flip=19，contour_change=6，group_change=6，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change
- `AG/fast/ZHANG_XIU_WEN`：p95=0.0884896769645987，invalid=20，gate_flip=20，contour_change=5，group_change=5，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change
- `ILO/AN_GUANG_JIE-0/before`：p95=0.08334015778097068，invalid=18，gate_flip=18，contour_change=7，group_change=7，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change
- `AG/slow/ZHANG_WEI_XIAN`：p95=0.08056901238219694，invalid=19，gate_flip=21，contour_change=4，group_change=4，nonlocal=0；source_baseline_invalid;gate_flip_or_became_invalid;closed_contour_change;crossing_group_change
- `AAA/unruputer/HAN_JIAN_FU`：p95=0.07468066773333133，invalid=18，gate_flip=20，contour_change=5，group_change=5，nonlocal=0；source_baseline_invalid;gate_flip_or_became_invalid;closed_contour_change;crossing_group_change
- `ILO/ZHANG_HE_PING-0/before`：p95=0.07043809052427552，invalid=19，gate_flip=20，contour_change=5，group_change=5，nonlocal=0；source_baseline_invalid;gate_flip_or_became_invalid;closed_contour_change;crossing_group_change
- `AG/slow/ZHANG_QUAN_YOU`：p95=0.06368684275598277，invalid=20，gate_flip=20，contour_change=5，group_change=5，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change
- `AAA/ruputer/GAO_FENG_SHAN`：p95=0.6446022083003495，invalid=15，gate_flip=18，contour_change=6，group_change=6，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_20pct;source_baseline_invalid
- `AG/slow/DING_LIAN_ZHONG`：p95=0.17123926933206277，invalid=19，gate_flip=19，contour_change=5，group_change=5，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_10pct
- `AG/fast/ZHANG_QING_WANG`：p95=0.1613507674385756，invalid=16，gate_flip=19，contour_change=5，group_change=5，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid;area_p95_ge_10pct
- `AAA/unruputer/LI_YOU_YU`：p95=0.12716401001887206，invalid=17，gate_flip=17，contour_change=7，group_change=7，nonlocal=0；source_baseline_invalid;gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_10pct
- `ILO/ZHANG_YAN_SHAN-0/before`：p95=0.11640901314043517，invalid=15，gate_flip=17，contour_change=7，group_change=7，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid;area_p95_ge_10pct
- `AAA/unruputer/SHEN_FANG_JIN`：p95=0.10975246363993937，invalid=15，gate_flip=15，contour_change=9，group_change=9，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_10pct
- `AG/slow/LIU_FENG`：p95=0.10080237927757264，invalid=15，gate_flip=17，contour_change=7，group_change=7，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid;area_p95_ge_10pct
- `ILO/PAN_YIN_GEN-1/before`：p95=0.09594969684290891，invalid=13，gate_flip=17，contour_change=7，group_change=7，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid
- `AAA/unruputer/YANG_BEN_RUI`：p95=0.08972972304566235，invalid=17，gate_flip=18，contour_change=6，group_change=6，nonlocal=0；source_baseline_invalid;gate_flip_or_became_invalid;closed_contour_change;crossing_group_change
- `ILO/WANG_TIAN_QING-1/before`：p95=0.31954768830961，invalid=16，gate_flip=18，contour_change=5，group_change=5，nonlocal=0；source_baseline_invalid;gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_20pct
- `ILO/WANG_JIN_MING-0/before`：p95=0.2904147110733677，invalid=17，gate_flip=19，contour_change=4，group_change=4，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid;area_p95_ge_20pct
- `AG/slow/GUAN_TONG_XIANG`：p95=0.18200794560315628，invalid=19，gate_flip=19，contour_change=4，group_change=4，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_10pct;source_baseline_invalid
- `AAA/unruputer/ZHU_ZI_HAI`：p95=0.13576306148504685，invalid=15，gate_flip=19，contour_change=4，group_change=4，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid;area_p95_ge_10pct
- `AG/fast/LI_SHI_QIANG`：p95=0.12437995141073441，invalid=17，gate_flip=19，contour_change=4，group_change=4，nonlocal=0；source_baseline_invalid;gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_10pct
- `ILO/LI_YOU_ZHI-0/before`：p95=0.12031147717439465，invalid=18，gate_flip=20，contour_change=3，group_change=3，nonlocal=0；source_baseline_invalid;gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_10pct
- `AG/slow/CHENG_LU_LI`：p95=0.11512325434586217，invalid=17，gate_flip=17，contour_change=6，group_change=6，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_10pct
- `AG/slow/MA_YU`：p95=0.11027582164093168，invalid=18，gate_flip=18，contour_change=5，group_change=5，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_10pct
- `AG/fast/LIU_JUN_FENG`：p95=0.10239108821189603，invalid=13，gate_flip=17，contour_change=6，group_change=6，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_10pct;source_baseline_invalid
- `ILO/ZHANG_JIAN_JUN-1/before`：p95=0.09175429175657238，invalid=18，gate_flip=19，contour_change=4，group_change=4，nonlocal=0；source_baseline_invalid;gate_flip_or_became_invalid;closed_contour_change;crossing_group_change
- `AG/fast/CHEN_SHI_MING`：p95=0.06539625763469806，invalid=18，gate_flip=18，contour_change=5，group_change=5，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change
- `AG/slow/HE_SHU_ZHEN`：p95=0.06401625886841764，invalid=15，gate_flip=18，contour_change=5，group_change=5，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid
- `ILO/GONG_HAI_ZENG-1/before`：p95=0.22700636907652694，invalid=12，gate_flip=14，contour_change=8，group_change=8，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid;area_p95_ge_20pct
- `AAA/ruputer/LI_ZHEN_HUA`：p95=0.20275254270174148，invalid=19，gate_flip=19，contour_change=3，group_change=3，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_20pct
- `AG/fast/LI_ZHEN_SHAN`：p95=0.12111952104346775，invalid=18，gate_flip=19，contour_change=3，group_change=3，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid;area_p95_ge_10pct
- `AAA/unruputer/SHEN_CHUN_WANG`：p95=0.11359963066331734，invalid=18，gate_flip=18，contour_change=4，group_change=4，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid;area_p95_ge_10pct
- `ILO/LIU_YUE_DONG-0/before`：p95=0.11109366782180088，invalid=12，gate_flip=16，contour_change=6，group_change=6，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid;area_p95_ge_10pct
- `AG/slow/XI_CHENG_JIANG`：p95=0.10930762580327241，invalid=17，gate_flip=17，contour_change=5，group_change=5，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_10pct
- `AAA/ruputer/LIN_LIANG_XIAO`：p95=0.10161197453442558，invalid=18，gate_flip=19，contour_change=3，group_change=3，nonlocal=0；source_baseline_invalid;gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_10pct
- `AG/slow/BAI_WEN_JIE`：p95=0.04508861227132205，invalid=17，gate_flip=18，contour_change=4，group_change=4，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid
- `ILO/GAO_SHU_CAI-0/before`：p95=0.22645634612087925，invalid=17，gate_flip=17，contour_change=4，group_change=4，nonlocal=0；source_baseline_invalid;gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_20pct
- `AG/slow/LI_HUAN_GE`：p95=0.2084411535762081，invalid=18，gate_flip=18，contour_change=3，group_change=3，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_20pct
- `AAA/ruputer/LI_BING_JIANG`：p95=0.14784111619389573，invalid=12，gate_flip=14，contour_change=7，group_change=7，nonlocal=0；source_baseline_invalid;gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_10pct
- `AG/fast/WANG_CHUN_MING`：p95=0.13978440397347036，invalid=18，gate_flip=18，contour_change=3，group_change=3，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_10pct
- `AG/slow/LI_BING_YI`：p95=0.11139529857640872，invalid=18，gate_flip=18，contour_change=3，group_change=3，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_10pct
- `AAA/ruputer/YU_TIAN_HAI`：p95=0.10269562619065122，invalid=17，gate_flip=17，contour_change=4，group_change=4，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_10pct
- `AAA/unruputer/FENG_ZHI_YING`：p95=0.2234235391378534，invalid=17，gate_flip=18，contour_change=2，group_change=2，nonlocal=0；source_baseline_invalid;gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_20pct
- `AAA/ruputer/HUO_SHI_LIANG`：p95=0.2056599067193328，invalid=17，gate_flip=18，contour_change=2，group_change=2，nonlocal=0；source_baseline_invalid;gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_20pct
- `AAA/ruputer/MENG_GUANG_QIN`：p95=0.17820952261783288，invalid=15，gate_flip=15，contour_change=5，group_change=5，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid;area_p95_ge_10pct
- `ILO/WANG_SHU_SHENG-0/before`：p95=0.16333561159860877，invalid=17，gate_flip=17，contour_change=3，group_change=3，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_10pct
- `ILO/CUI_WEI_PING-0/before`：p95=0.1199233867459363，invalid=17，gate_flip=17，contour_change=3，group_change=3，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_10pct
- `ILO/WU_JUN-0/before`：p95=0.06919664371081719，invalid=14，gate_flip=15，contour_change=5，group_change=5，nonlocal=0；source_baseline_invalid;gate_flip_or_became_invalid;closed_contour_change;crossing_group_change
- `AAA/unruputer/LIU_JIE`：p95=0.20240211457898058，invalid=15，gate_flip=15，contour_change=4，group_change=4，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid;area_p95_ge_20pct
- `AAA/ruputer/LIU_JI_XIN`：p95=0.19813599807837007，invalid=14，gate_flip=15，contour_change=4，group_change=4，nonlocal=0；source_baseline_invalid;gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_10pct
- `AAA/ruputer/ZOU_LI_SHUN`：p95=0.15465130616943057，invalid=16，gate_flip=16，contour_change=3，group_change=3，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid;area_p95_ge_10pct
- `AAA/ruputer/LI_LAO_PING`：p95=0.12243351746916609，invalid=14，gate_flip=15，contour_change=4，group_change=4，nonlocal=0；source_baseline_invalid;gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_10pct
- `AAA/ruputer/ZHANG_MAO_JIN`：p95=0.12184345491483486，invalid=16，gate_flip=16，contour_change=3，group_change=3，nonlocal=0；source_baseline_invalid;gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_10pct
- `AG/slow/GONG_HUI_XIA`：p95=0.1130008992532747，invalid=17，gate_flip=17，contour_change=2，group_change=2，nonlocal=0；source_baseline_invalid;gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_10pct
- `AAA/unruputer/GUO_BAO_CHUN`：p95=0.0920636213302774，invalid=14，gate_flip=14，contour_change=5，group_change=5，nonlocal=0；source_baseline_invalid;gate_flip_or_became_invalid;closed_contour_change;crossing_group_change
- `AG/slow/ZHANG_ZHI_JUN`：p95=0.0803530100199221，invalid=17，gate_flip=17，contour_change=2，group_change=2，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change
- `AAA/ruputer/DING_JUN_FENG`：p95=0.26634995329602396，invalid=16，gate_flip=16，contour_change=2，group_change=2，nonlocal=0；source_baseline_invalid;gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_20pct
- `AAA/unruputer/LIU_KANG_WEN`：p95=0.16861583571135505，invalid=14，gate_flip=15，contour_change=3，group_change=3，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid;area_p95_ge_10pct
- `AAA/ruputer/XIE_JIN_QUAN`：p95=0.14929669713385105，invalid=16，gate_flip=16，contour_change=2，group_change=2，nonlocal=0；source_baseline_invalid;gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_10pct
- `AG/slow/MA_TIAN_YI`：p95=0.13609527671438124，invalid=17，gate_flip=17，contour_change=1，group_change=1，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_10pct
- `AAA/ruputer/KANG_YONG`：p95=0.1317954387147846，invalid=15，gate_flip=15，contour_change=3，group_change=3，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_10pct
- `AG/fast/ZHANG_HAO`：p95=0.11702260992366137，invalid=12，gate_flip=14，contour_change=4，group_change=4，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid;area_p95_ge_10pct
- `ILO/SUN_DE_QI-1/before`：p95=0.10591849906095238，invalid=13，gate_flip=13，contour_change=5，group_change=5，nonlocal=0；source_baseline_invalid;gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_10pct
- `AG/slow/QU_HE_PING`：p95=0.05045213120648459，invalid=16，gate_flip=17，contour_change=1，group_change=1，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid
- `ILO/LU_FU_SHAN-0/before`：p95=0.22681700545061095，invalid=14，gate_flip=15，contour_change=2，group_change=2，nonlocal=0；source_baseline_invalid;gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_20pct
- `AAA/unruputer/LIU_CHENG_ZHANG`：p95=0.1596898153816991，invalid=11，gate_flip=12，contour_change=5，group_change=5，nonlocal=0；source_baseline_invalid;gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_10pct
- `AAA/ruputer/ZUO_DAO_SHENG`：p95=0.13598516103804914，invalid=14，gate_flip=14，contour_change=3，group_change=3，nonlocal=0；source_baseline_invalid;gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_10pct
- `AG/slow/CAO_FENG_CHI`：p95=0.1003331830287558，invalid=16，gate_flip=16，contour_change=1，group_change=1，nonlocal=0；source_baseline_invalid;gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_10pct
- `AG/slow/LI_GUI_YING`：p95=0.09463017630767101，invalid=11，gate_flip=14，contour_change=3，group_change=3，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid
- `AAA/unruputer/GAO_WEI_LI`：p95=0.16429955619790845，invalid=16，gate_flip=16，contour_change=0，group_change=0，nonlocal=0；source_baseline_invalid;gate_flip_or_became_invalid;area_p95_ge_10pct
- `AG/slow/ZANG_YU_SHU`：p95=0.14189801756781234，invalid=14，gate_flip=14，contour_change=2，group_change=2，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;source_baseline_invalid;area_p95_ge_10pct
- `AAA/ruputer/MA_XIAO_DONG`：p95=0.08766625501147246，invalid=15，gate_flip=15，contour_change=1，group_change=1，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change
- `AAA/ruputer/TONG_XUE_LIAN`：p95=0.2084656663466183，invalid=12，gate_flip=13，contour_change=2，group_change=2，nonlocal=0；source_baseline_invalid;gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_20pct
- `ILO/YAO_GUO_CHEN-0/before`：p95=0.17336065406634404，invalid=11，gate_flip=11，contour_change=4，group_change=4，nonlocal=0；source_baseline_invalid;gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_10pct
- `AG/slow/CHENG_GUANG_SEN`：p95=0.13406034980172155，invalid=14，gate_flip=14，contour_change=1，group_change=1，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_10pct
- `AG/fast/LIU_LI_QUN`：p95=0.08298672495433213，invalid=13，gate_flip=13，contour_change=2，group_change=2，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change
- `ILO/BAO_EN_YUN-0/before`：p95=0.14792835147639383，invalid=11，gate_flip=11，contour_change=3，group_change=3，nonlocal=0；source_baseline_invalid;gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_10pct
- `ILO/SUN_DONG_XIN-0/before`：p95=0.12133996384282775，invalid=8，gate_flip=10，contour_change=3，group_change=3，nonlocal=0；source_baseline_invalid;gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_10pct
- `AG/slow/LI_CHONG_ZENG`：p95=0.1559267323610806，invalid=12，gate_flip=12，contour_change=0，group_change=0，nonlocal=0；gate_flip_or_became_invalid;area_p95_ge_10pct
- `ILO/SUN_YU_SHENG-0/before`：p95=0.1130917503254452，invalid=10，gate_flip=10，contour_change=2，group_change=2，nonlocal=0；gate_flip_or_became_invalid;closed_contour_change;crossing_group_change;area_p95_ge_10pct
- `AG/fast/LI_SHU_KUN`：p95=0.1081433711803528，invalid=12，gate_flip=12，contour_change=0，group_change=0，nonlocal=0；gate_flip_or_became_invalid;area_p95_ge_10pct

## 沿程 segment 汇总

|segment|category|base sections|valid rate|gate flips|group changes|nonlocal|area p95|
|---:|---|---:|---:|---:|---:|---:|---:|
|0|pointcloud_full_half|111|0.930|17|17|0|0.010582336978631974|
|0|shift_pm1mm|111|0.850|274|86|1|0.06225487505837759|
|0|tilt_pm5deg|111|0.947|73|73|1|0.020763932010113516|
|1|pointcloud_full_half|107|0.884|19|19|0|0.034229534203152855|
|1|shift_pm1mm|107|0.893|141|46|0|0.14143895640656196|
|1|tilt_pm5deg|107|0.924|68|24|0|0.03261383868444958|
|2|pointcloud_full_half|129|0.895|18|18|0|0.03832951919080619|
|2|shift_pm1mm|129|0.897|133|34|0|0.14706895023294825|
|2|tilt_pm5deg|129|0.923|76|30|0|0.02830635458051239|
|3|pointcloud_full_half|139|0.869|64|64|0|0.1133871413456568|
|3|shift_pm1mm|139|0.733|496|108|0|0.19565726566904057|
|3|tilt_pm5deg|139|0.918|83|50|0|0.03537696165465208|
|4|pointcloud_full_half|133|0.893|27|27|0|0.04149523093160684|
|4|shift_pm1mm|133|0.746|485|107|0|0.1554372851319037|
|4|tilt_pm5deg|133|0.931|77|35|0|0.02766705214688397|
|5|pointcloud_full_half|149|0.922|36|36|0|0.039014593720012904|
|5|shift_pm1mm|149|0.758|465|80|0|0.1300395948266425|
|5|tilt_pm5deg|149|0.952|56|22|0|0.025378148308007612|
|6|pointcloud_full_half|153|0.866|63|63|0|0.10659897344380265|
|6|shift_pm1mm|153|0.730|514|104|0|0.20856602021564952|
|6|tilt_pm5deg|153|0.922|72|38|0|0.034295636846549994|

明细见 `p6_summary.json` 和 `p6_summary.csv`；脚本为 `p6_summary.py`。
