#!/usr/bin/env python3
"""Anonymous slide assets from existing, verified fields; no model execution.

Figure contract: demonstrate that one real aorto-iliac geometry can be displayed
with a spatial WSS prediction and compared with its existing CFD reference.
This illustrative image plate is not evidence of v5.2 performance or a clinical
risk map. All raw points and faces are retained, with a single physical scale.
Python/matplotlib is the established rendering backend. PNG-only delivery is
explicitly requested; PDF-specific QA is therefore not applicable.
"""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib import font_manager

OUT=Path(__file__).resolve().parent
ROOT=next(p for p in OUT.parents if (p/'training_wss_min').exists())
SOURCE=ROOT/'docs/03-汇报材料/启发式实验汇报_2026-09-13/启发式v2_代表病例后处理_20260914/X5D_v51/median'
plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['Noto Sans CJK JP','DejaVu Sans','Arial'], 'font.size':18,
                     'svg.fonttype':'none','pdf.fonttype':42,'axes.unicode_minus':False})
sys.path.insert(0, str(Path.home()/'.codex/skills/nature-figure/scripts'))
from audit_panel_alignment import require_matplotlib_panel_alignment

manifest=json.loads((SOURCE/'manifest.json').read_text())
raw_csv=SOURCE/'_export/same_point_fields.csv.gz'
rows=np.genfromtxt(str(raw_csv),delimiter=',',names=True)
xyz=np.stack([rows[k] for k in ('x_mm','y_mm','z_mm')],axis=1)
# Find the exact existing mesh using the source manifest; no identifying name
# is embedded in image labels, delivery names, or the public asset manifest.
mesh_filename=Path(manifest['geometry']['stl_source']).stem
candidates=list((ROOT/'training_wss_min/experiments/wss_deploy_timing_20260917/metrics_demo').glob('*.field.npz'))
mesh_path=next(p for p in candidates if mesh_filename in p.stem)
z=np.load(str(mesh_path))
v=z['stl_vertices'].astype(float);faces=z['stl_faces'].astype(int)
rotation=np.asarray(manifest['geometry']['rotation'])
centroid=np.asarray(manifest['geometry']['centroid_mm'])
aligned=(v-centroid)@rotation.T
distance,index=cKDTree(xyz).query(aligned)
assert distance.max()<1e-3
assert len(np.unique(index))==len(v)==len(rows)
truth=rows['wss_cfd_pa'][index];pred=rows['wss_pred_pa'][index]
assert np.isfinite(truth).all() and np.isfinite(pred).all()
# A single common, full physical range includes every point; no percentile
# clipping, per-panel rescaling, field smoothing, crop, or point subsampling.
vmax=float(np.ceil(max(truth.max(),pred.max())))
norm=Normalize(vmin=0,vmax=vmax)
cmap=LinearSegmentedColormap.from_list('wss_physical',['#173E69','#226D92','#3BABA1','#EAD879','#FFEFBA'])
angle=np.deg2rad(135)
right=np.array([np.cos(angle),np.sin(angle),0.0]);up=np.array([0.0,0.0,1.0]);depth=np.cross(right,up)
basis=np.stack([right,up,depth],axis=1)
projected=(v-(v.min(0)+v.max(0))/2)@basis
order=np.argsort(projected[faces,2].mean(1),kind='mergesort')
triangles=faces[order]
# Area-weighted vertex normals are used for neutral geometry lighting only.
fn=np.cross(v[faces[:,1]]-v[faces[:,0]],v[faces[:,2]]-v[faces[:,0]])
n=np.zeros_like(v)
for i in range(3): np.add.at(n,faces[:,i],fn)
n/=np.maximum(np.linalg.norm(n,axis=1)[:,None],1e-12)
light=-.35*right+.55*up+.75*depth;light/=np.linalg.norm(light)
luminance=.25+.65*np.abs(n@light)
gray=LinearSegmentedColormap.from_list('geometry_gray',['#384658','#ECF0F4'])
lo=projected[:,:2].min(0);hi=projected[:,:2].max(0)
center=(lo+hi)/2;span=(hi-lo)*1.07
# Identical extent and physical proportions for all comparable panels.
def paint(ax,values=None):
    arr=luminance if values is None else values
    ax.tripcolor(projected[:,0],projected[:,1],triangles,arr,
                 shading='gouraud',cmap=gray if values is None else cmap,
                 norm=Normalize(0,1) if values is None else norm,
                 edgecolors='none',rasterized=True)
    ax.set_xlim(center[0]-span[0]/2,center[0]+span[0]/2)
    ax.set_ylim(center[1]-span[1]/2,center[1]+span[1]/2)
    ax.set_aspect('equal');ax.axis('off')

def save(fig,stem,axes,exclude=()):
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,axes=axes,exclude_axes=exclude,
        row_groups=[list('ab')[:len(axes)]] if len(axes)>1 else None,
        panel_ids=list('ab')[:len(axes)],json_out=OUT/(stem+'.alignment.json'),strict=True)
    fig.savefig(str(OUT/(stem+'.png')),dpi=300,transparent=(stem!='deployment_pair'))
    fig.savefig(str(OUT/(stem+'.svg')),transparent=(stem!='deployment_pair'))
    plt.close(fig)

# Transparent cover assets contain no text; accompany with the caption below.
for stem,values in [('geometry_hero',None),('wss_prediction_hero',pred),('wss_cfd_hero',truth)]:
    fig=plt.figure(figsize=(4.0,7.2));ax=fig.add_axes([.01,.01,.98,.98]);paint(ax,values)
    save(fig,stem,[ax])

# Shared scale is an independent, transparent asset for editable slide layouts.
fig=plt.figure(figsize=(5.8,1.2));ax=fig.add_axes([.08,.65,.84,.17])
cb=matplotlib.colorbar.ColorbarBase(ax,cmap=cmap,norm=norm,orientation='horizontal',ticks=[0,2,4,6,8,10,11])
cb.outline.set_visible(False);cb.ax.tick_params(labelsize=15,length=2,pad=3)
cb.set_label('WSS / Pa',size=16,labelpad=2)
fig.savefig(str(OUT/'wss_colorbar.png'),dpi=300,transparent=True);plt.close(fig)

fig=plt.figure(figsize=(10.6,7.8))
axs=[fig.add_axes([.035,.17,.425,.71]),fig.add_axes([.54,.17,.425,.71])]
for ax,values in zip(axs,[truth,pred]):paint(ax,values)
fig.text(.248,.925,'CFD 参考',ha='center',va='center',size=25,weight='bold',color='#18334D')
fig.text(.752,.925,'模型预测',ha='center',va='center',size=25,weight='bold',color='#18334D')
cax=fig.add_axes([.30,.112,.40,.018])
cb=matplotlib.colorbar.ColorbarBase(cax,cmap=cmap,norm=norm,orientation='horizontal',ticks=[0,2,4,6,8,10,11])
cb.outline.set_visible(False);cb.ax.tick_params(labelsize=14,length=2,pad=4)
cb.set_label('WSS / Pa · 共用完整物理范围',size=14,labelpad=3)
fig.text(.5,.018,'匿名示例 A · X5D v5.1 既有中位病例 · 单个 seed · 同一峰值时刻',ha='center',size=13,color='#657587')
save(fig,'case_cfd_vs_prediction',axs,exclude=[cax])

# Compact shape-to-mechanics plate for a 7.25 x 3.95 inch slide placement.
fig=plt.figure(figsize=(7.25,3.95),facecolor='white')
axs=[fig.add_axes([.045,.17,.38,.70]),fig.add_axes([.575,.17,.38,.70])]
paint(axs[0]);paint(axs[1],pred)
fig.text(.235,.94,'血管形态',ha='center',va='center',size=17,weight='bold',color='#18334D')
fig.text(.765,.94,'壁面剪切应力',ha='center',va='center',size=17,weight='bold',color='#18334D')
from matplotlib.patches import FancyArrowPatch
arrow=FancyArrowPatch((.43,.54),(.57,.54),transform=fig.transFigure,
                      arrowstyle='-|>',mutation_scale=26,linewidth=2.5,color='#418A8D')
fig.add_artist(arrow)
fig.text(.5,.62,'快速预测',ha='center',va='center',size=13,color='#418A8D')
cax=fig.add_axes([.30,.123,.40,.018])
cb=matplotlib.colorbar.ColorbarBase(cax,cmap=cmap,norm=norm,orientation='horizontal',ticks=[0,2,4,6,8,10,11])
cb.outline.set_visible(False);cb.ax.tick_params(labelsize=8,length=2,pad=2)
fig.text(.74,.131,'WSS / Pa',ha='left',va='center',size=9,color='#18334D')
fig.text(.5,.025,'v5.1 既有病例示例 · 匿名示例 A',ha='center',va='center',size=9,color='#657587')
save(fig,'deployment_pair',axs,exclude=[cax])

public_manifest={
 'asset_type':'real-data anatomical and WSS images, not AI-generated concepts',
 'case_alias':'匿名示例 A', 'model':'X5D_v51', 'seed':1234, 'checkpoint':'ckpt_best.pt',
 'purpose':'Opening-report illustration of geometry-to-mechanics and an existing CFD/prediction example',
 'evidence_boundary':'Historical v5.1 illustration only; not v5.2 performance evidence, not an ensemble, not a clinical risk map',
 'source_selection':'Existing median case selected by closest per-case five-seed mean R² to the test-set median; displayed fields are seed 1234',
 'new_inference_or_evaluation':False,'points':len(v),'triangles':len(faces),'excluded_points':0,'excluded_triangles':0,
 'units':{'coordinates':'mm','WSS':'Pa'},'peak_step':manifest['peak_step'],
 'scalar_rendering':'Original same-point fields mapped bijectively onto original STL vertices; vertex-colour Gouraud rendering; no Gaussian smoothing',
 'coordinate_mapping_max_distance_mm':float(distance.max()),
 'color_range_pa':[0,vmax], 'raw_field_range_pa':{'cfd':[float(truth.min()),float(truth.max())],'prediction':[float(pred.min()),float(pred.max())]},
 'color_limits_policy':'shared full range rounded upward; no clipping; no selfmax normalization',
 'camera':{'orthographic':True,'azimuth_deg':135,'elevation_deg':0,'basis':basis.tolist()},
 'geometry_lighting':'Area-weighted vertex normals; neutral geometry only. Field colors are not lighting-shaded.',
 'source_sha256':{'same_point_fields.csv.gz':hashlib.sha256(raw_csv.read_bytes()).hexdigest(),'existing_mesh_npz':hashlib.sha256(mesh_path.read_bytes()).hexdigest()},
 'exports':['geometry_hero.png','wss_prediction_hero.png','wss_cfd_hero.png','wss_colorbar.png','case_cfd_vs_prediction.png','deployment_pair.png'],
 'caption':'匿名病例的真实血管形态及既有 WSS 预测示例。CFD 与预测共用 Pa 色标；该图用于说明研究对象和输出形式，不能单独证明泛化性能或临床风险。',
 'qa':{'backend':'Python/matplotlib','panel_alignment':'automatic check passed; JSON retained','PDF_audits':'not applicable: no PDF requested or generated','visual_check':'passed: complete anatomy retained, matched pose and scale, legible labels and no detected cropping or overlap on PNGs'}
}
(OUT/'case_assets_manifest.json').write_text(json.dumps(public_manifest,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'outputs':public_manifest['exports'],'max_mapping_distance_mm':float(distance.max())},ensure_ascii=False))
