from pathlib import Path
import json,csv
D=Path(__file__).resolve().parent
parts=[]
def table(name,header,rows):
    with (D/(name+'.tsv')).open('w') as f:
        w=csv.writer(f,delimiter='\t');w.writerow(header);w.writerows(rows)
    return '\n'.join(['| '+' | '.join(header)+' |','|'+'|'.join(['---']*len(header))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
parts.append('本周工作汇总（截至 2026-09-27）\n\n本次汇总读取已有评估与保存预测，未启动训练或推理。X5Dcaps 按仓库正式名称写作 X5Dcap，即采用开口半径计算 Murray 分流先验的版本。除时间表另行说明外，R² 为物理空间病例等权场 R²_cb，MAE 也按病例等权；WSS/TAWSS 的 MAE 单位为 Pa，OSI 无量纲。± 为训练 seed 间样本标准差，集成指先对物理预测取均值再计算指标。')
parts.append('第一组：数据扩充与峰值 WSS 模型对比。数据从 170 个病例单元增加到 261 个（新增 91 个）。')
base=[
['V5.1','X5D','CV3，136例折外','单seed 1234','0.7100','1.5040'],
['V5.2','X5D','CV5，261例折外','三seed均值','0.7538 ± 0.0022','1.2810 ± 0.0023'],
['V5.2','X5Dcap','CV5，261例折外','三seed均值','0.7608 ± 0.0019','1.2534 ± 0.0058'],
['V5.2','X5D','170例训练→新增91例测试','五seed集成','0.7514','1.0919'],
['V5.2','X5Dcap','170例训练→新增91例测试','五seed集成','0.7644','1.0607'],
]
parts.append(table('01_数据增量与主线对比',['版本','模型','评估协议','重复/集成方式','R²_cb ↑','MAE（Pa）↓'],base))
parts.append('可汇报结论：扩数据后，X5D 折外 R² 从 0.7100 提高至 0.7538，MAE 从 1.5040 降至 1.2810 Pa；V5.2 同协议下，X5Dcap 在 CV5 的 R² 再提高约 0.0070、MAE 降低约 2.2%，新增91例测试集集成 R² 提高约 0.0130、MAE 降低约 2.9%。跨版本同时改变了训练量、数据修正和 CV3/CV5 划分，只能解释为版本整体进展。V5.1 的 CV3 在原 train136 内进行，旧 test34 不在该表第一行；CV5 每折训练204–212例。新增91例按病例单元划分，不能自动等同于外部中心或患者独立验证。')
parts.append('第二组：WSS 时间建模尝试。下表为 V5.1、单 seed、三折指标平均；全周期 R² 先逐帧计算再平均，谷底指协议流量最低的20帧。')
t=json.loads((D/'time_metrics.json').read_text())['rows']
labels={
'T-null (B_scale)':'T-null：峰值预测＋训练折时间统计',
'T0':'T0：几何＋相位查询，端到端',
'TB8':'TB8：8维时间基系数',
'TB16':'TB16：16维时间基系数',
'P-mlp':'P-mlp：冻结隐藏特征＋时间MLP',
'P-mlp-qx':'P-mlp-qx：去掉病例级向量',
'C-raw':'C-raw：原始几何＋峰值锚定MLP',
'TL-warm':'TL-warm：峰值权重初始化＋部分解冻',
'TL-random':'TL-random：随机初始化对照',
'TT-warm':'TT-warm：隐藏特征＋跨帧注意力',
'TT-warm-noattn':'TT-warm-noattn：隐藏特征＋对角注意力',
'TT-raw':'TT-raw：原始几何＋跨帧注意力',
'TT-raw-noattn':'TT-raw-noattn：原始几何＋对角注意力'}
tr=[]
for r in t:
 vals=[f'{r[k]:.4f}' for k in ('full_cycle_r2_pa','trough_r2_pa','peak_r2_pa','tawss_r2_pa')]
 if r['arm'].startswith('T-null'): vals[3]='约0.6960'
 tr.append([labels[r['arm']],*vals,'—' if r['full_cycle_mae_pa'] is None else f"{r['full_cycle_mae_pa']:.3f}"])
parts.append(table('02_时间建模对比',['尝试','全周期R² ↑','谷底R² ↑','峰值R² ↑','TAWSS R² ↑','全周期MAE（Pa）↓'],tr))
parts.append('可汇报结论：T0 相对 T-null 提高了全周期及 TAWSS 精度，但个别折损害峰值；锚定模型能保持峰值。当前 Transformer 的跨帧注意力增量在隐藏特征组为 −0.0015、原始几何组为 +0.0017，尚无稳定的全周期 Pa 优势。跨批次训练预算和选模不同，不能把表中排序解释为纯结构消融，也不能把约0.50视为已证明的可预测上限。—表示该合同下未保存全周期MAE；TT四臂的TAWSS MAE另存于time_metrics.tsv。时间表沿用历史81帧均值，下面周期量直接回归的标签使用向量WSS的0–79帧，二者不是完全相同的积分合同。')
parts.append('第三组：直接回归 TAWSS、OSI，以及指定论文的对照。先列主要交付模型；本工作模型行均为三 seed 集成。test34 是已用于开发的旧评估集，V5.2 列为261例五折折外结果。')
c=json.loads((D/'cycle_metrics.json').read_text())['rows']
sel=[('tawss','v5.1 A1 test34 ens3','A1单头','V5.1 test34'),('tawss','v5.1 M1 test34 ens3','M1三头','V5.1 test34'),('tawss','v5.2 M1cap CV5 ens3','M1cap三头','V5.2 CV5，261例'),('osi','v5.1 O1 test34 ens3','O1线性单头','V5.1 test34'),('osi','v5.1 O2 test34 ens3','O2 logit单头','V5.1 test34'),('osi','v5.1 M1 test34 ens3','M1三头','V5.1 test34'),('osi','v5.2 M1cap CV5 ens3','M1cap三头','V5.2 CV5，261例')]
cr=[]
for target,label,model,protocol in sel:
 r=next(x for x in c if x['label']==label and x['target']==target)
 cr.append([target.upper(),model,protocol,*[f'{r[k]:.4f}' for k in ('mae_cb','r2_cb','approx_disp_casemean','cosine_spatial_casemean','spearman_casemean')]])
parts.append(table('03_周期量主要模型对比',['目标','模型（三seed集成）','评估协议','MAE ↓','R²_cb ↑','Approx. disp. ↓','Cos.（空间场）↑','Spearman ρ ↑'],cr))
parts.append('为观察版本进展，再固定为同一批136例折外病例；这里比较V5.1单seed模型与V5.2三seed的单模型指标均值，未混入集成收益。')
common=[]
for target in ('tawss','osi'):
 for label,version in [('v5.1 M1 cv3 s1234','V5.1 M1，CV3，单seed'),('v5.2 M1cap CV5 seedmean3 common136','V5.2 M1cap，CV5，三seed均值')]:
  r=next(x for x in c if x['label']==label and x['target']==target)
  common.append([target.upper(),version,*[f'{r[k]:.4f}' for k in ('mae_cb','r2_cb','approx_disp_casemean','cosine_spatial_casemean','spearman_casemean')]])
parts.append(table('04_周期量同136例版本对比',['目标','版本/模型（共同136例）','MAE ↓','R²_cb ↑','Approx. disp. ↓','Cos.（空间场）↑','Spearman ρ ↑'],common))
parts.append('同136例上，TAWSS R²提高0.0685、MAE下降0.0510 Pa；OSI R²提高0.0609、MAE下降0.0045。评估病例已对齐，但训练规模、折数及cap配方仍同时变化。最新261例CV5三seed单模型均值为：TAWSS R² 0.7656 ± 0.0032、MAE 0.3222 ± 0.0025 Pa；OSI R² 0.5285 ± 0.0026、MAE 0.0581 ± 0.0001。')
parts.append('指定论文《Wall Shear Stress Estimation in Abdominal Aortic Aneurysms: Towards Generalisable Neural Surrogate Models》的数值已回到原文Table 1核对，以下为其LaB-GATr结果。')
lit=json.loads((D/'literature_verified.json').read_text())
lr=[]
for r in lit['rows']:
 lr.append([r['target'],r['dataset'],f"{r['mae']:.3f} ± {r['mae_sd']:.3f}",'未报告',f"{r['approx_disp']:.3f} ± {r['approx_disp_sd']:.3f}",'周期标量未报告','未报告'])
parts.append(table('05_指定论文对比',['目标','论文评估协议','MAE ↓','R²','Approx. disp. ↓','Cos. similarity','Spearman'],lr))
parts.append('指标说明：Approx. disp.＝逐例相对L2误差后取病例均值；本表空间场Cos＝将一例所有壁面点的标量值组成向量后计算余弦，再取病例均值；Spearman同样先逐例计算。论文明确只对瞬态三维WSS向量报告方向余弦（AAA-100为0.657、AAA-L为0.684），没有TAWSS/OSI余弦，也没有对应R²或Spearman，不能把这些WSS向量数值填到周期量栏。论文±为评估样本间标准差，与本工作的seed标准差不同。')
parts.append('可汇报结论：按相对L2，本工作最新TAWSS为0.3052，处于论文0.333–0.387的相近水平；OSI为0.4290，高于论文0.298–0.381，仍有差距。TAWSS绝对MAE也高于该论文。由于队列、入口工况、划分和直接回归/向量积分方法不同，这是跨研究参考，不能据此宣称同条件精度领先。')
parts.append('可追溯文件：x5d_metrics.json/tsv（峰值主线）；time_metrics.json/tsv（时间主表，含未放入正文的TAWSS MAE），time_phase_metrics与time_volume_metrics（分相位/体场附表）；cycle_metrics.json/tsv（完整周期量、单seed与集成）；cycle_attempts.json/tsv（OSI损失/几何/辅助目标尝试）；validation.json（4个原始run字段复现最大误差3.4e-16）；literature_verified.json（原文来源）。周期量OOF统一按正式病例等权函数重算，历史文档中折均值、点池化MAE和旧文献脚本中心均值造成的差异不混入主表。')
parts.append('原始入口：[X5D主线跟踪](../../02-推进与变更/01-X5D主线与新数据/X5D主线_实验跟踪.md)；[时间建模复核](../../../training_wss_min/experiments/wss_time_transformer_v51_20260924_r3/analysis_20260927/time_modeling_review.md)；[周期量跟踪](../../02-推进与变更/03-周期量TAWSS_OSI/TAWSS_OSI_实验跟踪.md)；[论文Table 1](https://arxiv.org/html/2507.22817v1#S3.T1)。')
(D/'本周工作汇总_可复制表格.md').write_text('\n\n'.join(parts)+'\n')
print(D/'本周工作汇总_可复制表格.md')
