/* Shared report helpers (contract ANALYSIS_CONTRACT.md §12): centreline measurements, export helpers,
   built-in presets, glossary popover and the parent-page message protocol.  UMD: WssReportCommon in the
   browser, module.exports under node.  Pure functions come first and never touch the DOM or three.js. */
(function(root){
'use strict';
// ---------------------------------------------------------------- §12.1 centreline groups & measurements
function dist3(a,b){return Math.hypot(a[0]-b[0],a[1]-b[1],a[2]-b[2]);}
function straightDistance(a,b){return dist3(a,b);}
function _projectToGroup(g,p){
  const k=g.s.length,x=g.xyz;
  if(!k)return null;
  if(k===1){const q=[x[0],x[1],x[2]];return {j:0,t:0,s:0,radius_mm:g.radius[0],xyz:q,dist_mm:dist3(p,q),row_local:0};}
  let best=null;
  for(let j=0;j<k-1;j++){
    const ax=x[3*j],ay=x[3*j+1],az=x[3*j+2],bx=x[3*j+3]-ax,by=x[3*j+4]-ay,bz=x[3*j+5]-az;
    const l2=bx*bx+by*by+bz*bz;let t=l2>0?((p[0]-ax)*bx+(p[1]-ay)*by+(p[2]-az)*bz)/l2:0;t=Math.max(0,Math.min(1,t));
    const q=[ax+bx*t,ay+by*t,az+bz*t],d=dist3(p,q);
    if(!best||d<best.dist_mm)best={j,t,xyz:q,dist_mm:d};
  }
  const j=best.j,t=best.t;
  best.s=g.s[j]+t*(g.s[j+1]-g.s[j]);best.radius_mm=g.radius[j]+t*(g.radius[j+1]-g.radius[j]);best.row_local=t<.5?j:j+1;
  return best;
}
function buildCenterlineGroups(cl,branchNames,opts){
  cl=cl||{};opts=opts||{};const xyz=cl.xyz||[],radius=cl.radius||cl.radius_mm||null,edges=cl.edges||null,seg=cl.segment||null;
  const n=Math.floor(xyz.length/3),rowsBySeg=new Map();
  for(let i=0;i<n;i++){const sid=seg?Number(seg[i]):0;if(!rowsBySeg.has(sid))rowsBySeg.set(sid,[]);rowsBySeg.get(sid).push(i);}
  const groups=[];
  for(const [sid,rows] of rowsBySeg){
    let ordered=rows;
    if(edges&&edges.length&&rows.length>1){
      // Exported edges follow sample_index order (proximal → distal); rows may be interleaved on disk.
      const members=new Set(rows),next=new Map(),preceded=new Set();
      for(let i=0;i<edges.length;i+=2){const a=edges[i],b=edges[i+1];if(members.has(a)&&members.has(b)){next.set(a,b);preceded.add(b);}}
      const starts=rows.filter(i=>!preceded.has(i));
      if(starts.length===1){const o=[],seen=new Set();let i=starts[0];while(i!==undefined&&!seen.has(i)){o.push(i);seen.add(i);i=next.get(i);}if(o.length===rows.length)ordered=o;}
    }
    const k=ordered.length,pts=new Float32Array(3*k),rad=new Float32Array(k),s=new Float32Array(k);let acc=0;
    for(let j=0;j<k;j++){const i=ordered[j];pts[3*j]=xyz[3*i];pts[3*j+1]=xyz[3*i+1];pts[3*j+2]=xyz[3*i+2];rad[j]=radius?Number(radius[i])||0:0;
      if(j>0)acc+=Math.hypot(pts[3*j]-pts[3*j-3],pts[3*j+1]-pts[3*j-2],pts[3*j+2]-pts[3*j-1]);s[j]=acc;}
    const named=branchNames&&branchNames[String(sid)];
    groups.push({segment_id:sid,name:typeof named==='string'?named:('分支 '+sid),rows:ordered,xyz:pts,radius:rad,s,length_mm:acc,parent_id:null,parent_s:0,children:[]});
  }
  const bySid=new Map(groups.map(g=>[g.segment_id,g]));
  for(const g of groups){
    const forced=opts.parents&&opts.parents[String(g.segment_id)];
    if(forced!==undefined&&forced!==null&&bySid.has(Number(forced))&&Number(forced)!==g.segment_id){const pg=bySid.get(Number(forced));const pr=_projectToGroup(pg,[g.xyz[0],g.xyz[1],g.xyz[2]]);g.parent_id=pg.segment_id;g.parent_s=pr?pr.s:pg.length_mm;continue;}
    if(!g.s.length)continue;
    const p0=[g.xyz[0],g.xyz[1],g.xyz[2]],thr=Math.max(3,1.5*(g.radius[0]||0));let best=null,fallback=null;
    for(const o of groups){
      if(o===g||!o.s.length)continue;const pr=_projectToGroup(o,p0);if(!pr||pr.dist_mm>=thr)continue;
      // A sibling that starts at the same junction projects onto its own start; the parent is met at its interior or end.
      const atStart=pr.s<=Math.min(2,.1*o.length_mm+1e-9);
      if(atStart){if(!fallback||pr.dist_mm<fallback.dist_mm)fallback={o,pr,dist_mm:pr.dist_mm};continue;}
      if(!best||pr.dist_mm<best.dist_mm)best={o,pr,dist_mm:pr.dist_mm};
    }
    const pick=best||null;
    if(pick){g.parent_id=pick.o.segment_id;g.parent_s=pick.pr.s;}
  }
  // Break accidental cycles (mutual parents) so tree walks always terminate.
  for(const g of groups){const seen=new Set([g.segment_id]);let cur=g;while(cur.parent_id!==null){if(seen.has(cur.parent_id)){cur.parent_id=null;cur.parent_s=0;break;}seen.add(cur.parent_id);cur=bySid.get(cur.parent_id);if(!cur)break;}}
  for(const g of groups)g.children=[];
  for(const g of groups)if(g.parent_id!==null&&bySid.has(g.parent_id))bySid.get(g.parent_id).children.push(g.segment_id);
  return groups;
}
function projectToCenterline(groups,p){
  let best=null;
  for(let gi=0;gi<(groups||[]).length;gi++){const g=groups[gi],pr=_projectToGroup(g,p);if(!pr)continue;if(!best||pr.dist_mm<best.dist_mm)best={...pr,group_index:gi,segment_id:g.segment_id,name:g.name,row:g.rows[pr.row_local]};}
  if(!best)return null;
  return {segment_id:best.segment_id,name:best.name,s:best.s,radius_mm:best.radius_mm,row:best.row,row_local:best.row_local,xyz:best.xyz,dist_mm:best.dist_mm,group_index:best.group_index};
}
function _chain(groups,proj){
  const bySid=new Map(groups.map(g=>[g.segment_id,g]));const out=[];let g=bySid.get(proj.segment_id),s=proj.s;const seen=new Set();
  while(g&&!seen.has(g.segment_id)){out.push({g,s});seen.add(g.segment_id);if(g.parent_id===null)break;const p=bySid.get(g.parent_id);s=g.parent_s;g=p;}
  return out;
}
function arcDistance(groups,a,b){
  const pa=projectToCenterline(groups,a),pb=projectToCenterline(groups,b);
  if(!pa||!pb)return null;
  if(pa.segment_id===pb.segment_id)return {value_mm:Math.abs(pa.s-pb.s),same_branch:true,path:[pa.segment_id],a:pa,b:pb};
  const ca=_chain(groups,pa),cb=_chain(groups,pb);
  let ia=-1,ib=-1;
  outer:for(let i=0;i<ca.length;i++)for(let j=0;j<cb.length;j++)if(ca[i].g===cb[j].g){ia=i;ib=j;break outer;}
  if(ia<0)return {value_mm:NaN,same_branch:false,path:[],a:pa,b:pb,disconnected:true};
  let total=0;const path=[];
  for(let i=0;i<ia;i++){total+=ca[i].s;path.push(ca[i].g.segment_id);}
  for(let j=0;j<ib;j++)total+=cb[j].s;
  total+=Math.abs(ca[ia].s-cb[ib].s);path.push(ca[ia].g.segment_id);
  for(let j=ib-1;j>=0;j--)path.push(cb[j].g.segment_id);
  return {value_mm:total,same_branch:false,path,a:pa,b:pb};
}
function arcFromRoot(groups,p){
  const pr=projectToCenterline(groups,p);if(!pr)return null;const chain=_chain(groups,pr);let s=0;for(const c of chain)s+=c.s;return {s_from_root_mm:s,projection:pr,path:chain.map(c=>c.g.segment_id)};
}
function localDiameter(groups,p){
  const pr=projectToCenterline(groups,p);if(!pr)return null;const g=groups[pr.group_index];
  const cands=[];if(g.parent_id!==null)cands.push(pr.s);if(g.children&&g.children.length)cands.push(Math.max(0,g.length_mm-pr.s));
  return {diameter_mm:2*pr.radius_mm,radius_mm:pr.radius_mm,segment_id:pr.segment_id,name:pr.name,s:pr.s,dist_to_junction_mm:cands.length?Math.min(...cands):null,xyz:pr.xyz,dist_mm:pr.dist_mm};
}
// ---------------------------------------------------------------- §12.2 ids, labels, probe export
function newId(prefix,existing){
  const used=new Set();for(const item of existing||[]){const id=item&&typeof item==='object'?item.id:item;if(id!=null)used.add(String(id));}
  let n=1;while(used.has(prefix+n))n++;return prefix+n;
}
function fmtNum(x,d){return Number.isFinite(+x)?(+x).toFixed(d==null?1:d):'—';}
function measurementLabel(m,lang){
  m=m||{};const en=lang==='en';const kind={distance:['直线距离','Distance'],arc:['弧长距离','Arc distance'],diameter:['管径','Diameter'],segment:['分段长度','Segment length']}[m.kind]||[m.kind||'测量',m.kind||'Measurement'];
  let text=kind[en?1:0]+' '+fmtNum(m.value_mm,1)+' mm';
  if(m.branch)text+=en?' ('+englishLabel(m.branch,'en')+')':'（'+m.branch+'）';
  if(m.kind==='diameter')text+=en?' · 2 × inscribed radius':' · 内切半径 × 2';
  return text;
}
function _probeColumns(rows,lang){
  const en=lang==='en',base=[['id','编号','ID'],['x','x_mm','x_mm'],['y','y_mm','y_mm'],['z','z_mm','z_mm'],['branch','分支','branch'],['segment_id','分支编号','segment_id'],['s_from_root_mm','距入口弧长_mm','s_from_root_mm'],['radius_mm','半径_mm','radius_mm']];
  const keys=[];for(const r of rows||[])for(const k of Object.keys((r&&r.values)||{}))if(!keys.includes(k))keys.push(k);
  return {head:base.map(b=>b[en?2:1]).concat(keys),cells:r=>{const xyz=Array.isArray(r.xyz_mm)?r.xyz_mm:[null,null,null];const v=r.values||{};return [r.id??'',xyz[0]??'',xyz[1]??'',xyz[2]??'',r.branch??'',r.segment_id??'',r.s_from_root_mm??'',r.radius_mm??''].concat(keys.map(k=>v[k]??''));}};
}
function _cell(x,sep){const s=x===null||x===undefined?'':(typeof x==='number'?(Number.isFinite(x)?String(+x.toFixed(4)):''):String(x));return sep===','&&/[",\r\n]/.test(s)?'"'+s.replace(/"/g,'""')+'"':s.replace(/[\t\r\n]+/g,' ');}
function probeToTSV(rows,lang){const c=_probeColumns(rows,lang);return [c.head.join('\t')].concat((rows||[]).map(r=>c.cells(r).map(x=>_cell(x,'\t')).join('\t'))).join('\n');}
function probeToCSV(rows,lang){const c=_probeColumns(rows,lang);return '\ufeff'+[c.head.join(',')].concat((rows||[]).map(r=>c.cells(r).map(x=>_cell(x,',')).join(','))).join('\r\n')+'\r\n';}
// ---------------------------------------------------------------- §12.3 colour bars, English labels, filenames
const CMAPS={rainbow:[[0,0,143],[0,32,255],[0,160,255],[0,255,255],[64,255,160],[160,255,64],[255,255,0],[255,160,0],[255,64,0],[190,0,0]],
  turbo:[[48,18,59],[70,107,227],[36,182,213],[37,241,150],[128,254,66],[210,239,35],[253,183,31],[240,107,14],[199,40,6],[122,4,3]],
  bwr:[[31,78,156],[247,247,247],[192,57,43]],viridis:[[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]]};
CMAPS.bluewhitered=CMAPS.bwr;
function colormapNames(){return ['rainbow','turbo','bwr','viridis'];}
function colormapStops(name){const c=CMAPS[name]||CMAPS.rainbow;return c.map((rgb,i)=>[i/(c.length-1),'#'+rgb.map(v=>v.toString(16).padStart(2,'0')).join('')]);}
function _hex2rgb(h){const m=/^#?([0-9a-f]{6})$/i.exec(String(h));if(!m)return [0,0,0];const v=parseInt(m[1],16);return [v>>16&255,v>>8&255,v&255];}
function colorAt(stops,t){t=Math.max(0,Math.min(1,+t||0));if(!stops||!stops.length)return '#000000';let i=0;while(i<stops.length-2&&stops[i+1][0]<t)i++;const [t0,c0]=stops[i],[t1,c1]=stops[Math.min(i+1,stops.length-1)];const u=t1>t0?(t-t0)/(t1-t0):0;const a=_hex2rgb(c0),b=_hex2rgb(c1);return '#'+[0,1,2].map(k=>Math.round(a[k]+(b[k]-a[k])*Math.max(0,Math.min(1,u))).toString(16).padStart(2,'0')).join('');}
function _valueAt(t,lo,hi,log,floor){floor=floor>0?floor:.05;if(log){const l0=Math.log(Math.max(lo,floor)),l1=Math.log(Math.max(hi,floor*1.001));return Math.exp(l0+t*(l1-l0));}return lo+t*(hi-lo);}
function _xml(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function _tickFmt(v,lo,hi,log){if(log){const a=Math.abs(+v);if(!(a>0))return '0';const d=Math.max(0,2-Math.floor(Math.log10(a)));return String(+(+v).toFixed(Math.min(d,4)));}const span=Math.abs(hi-lo);const d=span>=100?0:span>=10?1:span>=1?2:3;return (+v).toFixed(d);}
function colorbarSVG(o){
  o=o||{};const stops=Array.isArray(o.stops)&&o.stops.length?o.stops:colormapStops(o.colormap||'rainbow');
  const vertical=(o.orientation||'vertical')!=='horizontal',W=o.width||(vertical?110:380),H=o.height||(vertical?320:80);
  const bands=Math.max(0,Math.floor(o.bands)||0),lo=Number.isFinite(+o.min)?+o.min:0,hi=Number.isFinite(+o.max)?+o.max:1,log=!!o.log,lang=o.lang||'zh';
  const title=o.title!=null?String(o.title):englishLabel('wss',lang),units=o.units!=null?String(o.units):'';
  const barX=vertical?12:24,barY=vertical?30:24,barW=vertical?18:W-48,barH=vertical?H-52:18;
  let out='<svg xmlns="http://www.w3.org/2000/svg" width="'+W+'" height="'+H+'" viewBox="0 0 '+W+' '+H+'" font-family="Helvetica, Arial, sans-serif" font-size="11" fill="#203049">';
  out+='<title>'+_xml(title+(units?' · '+units:''))+'</title>';
  out+='<text x="'+(vertical?12:24)+'" y="16" font-size="12" font-weight="600">'+_xml(title)+(units?' <tspan fill="#5f7086">('+_xml(units)+')</tspan>':'')+'</text>';
  const ticks=[];
  if(bands>0){
    for(let i=0;i<bands;i++){const c=colorAt(stops,(i+.5)/bands);const f0=i/bands,f1=(i+1)/bands;
      if(vertical)out+='<rect x="'+barX+'" y="'+(barY+barH*(1-f1)).toFixed(2)+'" width="'+barW+'" height="'+(barH/bands).toFixed(2)+'" fill="'+c+'"/>';
      else out+='<rect x="'+(barX+barW*f0).toFixed(2)+'" y="'+barY+'" width="'+(barW/bands).toFixed(2)+'" height="'+barH+'" fill="'+c+'"/>';}
    for(let i=0;i<=bands;i++)ticks.push({f:i/bands,v:_valueAt(i/bands,lo,hi,log)});
  }else{
    out+='<defs><linearGradient id="g" x1="0" y1="'+(vertical?1:0)+'" x2="'+(vertical?0:1)+'" y2="0">'+stops.map(([t,c])=>'<stop offset="'+(t*100).toFixed(1)+'%" stop-color="'+c+'"/>').join('')+'</linearGradient></defs>';
    out+='<rect x="'+barX+'" y="'+barY+'" width="'+barW+'" height="'+barH+'" fill="url(#g)"/>';
    for(let i=0;i<=4;i++)ticks.push({f:i/4,v:_valueAt(i/4,lo,hi,log)});
  }
  out+='<rect x="'+barX+'" y="'+barY+'" width="'+barW+'" height="'+barH+'" fill="none" stroke="#7c8794" stroke-width="1"/>';
  for(const t of ticks){const label=_tickFmt(t.v,lo,hi,log);
    if(vertical){const y=barY+barH*(1-t.f);out+='<line x1="'+(barX+barW)+'" y1="'+y.toFixed(2)+'" x2="'+(barX+barW+4)+'" y2="'+y.toFixed(2)+'" stroke="#7c8794"/><text class="tick" x="'+(barX+barW+7)+'" y="'+(y+4).toFixed(2)+'">'+label+'</text>';}
    else{const x=barX+barW*t.f;out+='<line x1="'+x.toFixed(2)+'" y1="'+(barY+barH)+'" x2="'+x.toFixed(2)+'" y2="'+(barY+barH+4)+'" stroke="#7c8794"/><text class="tick" x="'+x.toFixed(2)+'" y="'+(barY+barH+16)+'" text-anchor="middle">'+label+'</text>';}}
  if(log)out+='<text x="'+(vertical?12:24)+'" y="'+(H-4)+'" font-size="9" fill="#5f7086">'+(lang==='en'?'log scale':'对数色标')+'</text>';
  return out+'</svg>';
}
const LABELS={wss:['WSS · 壁面切应力','WSS · wall shear stress'],wss_short:['WSS','WSS'],pressure:['相对压力','Relative pressure'],speed:['速度大小','Speed'],velocity:['速度','Velocity'],
  radius:['局部半径','Local radius'],arc:['沿程弧长','Arc length'],theta:['周向角','Circumferential angle'],dist_junction:['到分叉距离','Distance to bifurcation'],centerline_radius:['中心线半径','Centreline radius'],
  distance:['直线距离','Distance'],arc_distance:['弧长距离','Arc distance'],diameter:['管径','Diameter'],segment_length:['分段长度','Segment length'],annotation:['标注','Annotation'],measurement:['测量','Measurement'],probe:['探针','Probe'],
  peak_max:['全场最大值','Field maximum'],p99:['空间 p99','Spatial p99'],highlight:['高亮','Highlight'],fixed_frame:['固定收缩期帧','Fixed systolic frame'],colorbar:['色标','Colour bar'],branch:['分支','Branch'],case:['病例','Case'],
  legend_note:['统计：预测点云 · 壁面：Gaussian 插值','Statistics on prediction points · wall colours: Gaussian interpolation'],units_pa:['Pa','Pa'],units_dyn:['dyn/cm²','dyn/cm²'],units_mmhg:['mmHg','mmHg'],units_ms:['m/s','m/s'],units_cms:['cm/s','cm/s'],mm:['mm','mm'],
  view_front:['前','Front'],view_back:['后','Back'],view_left:['左','Left'],view_right:['右','Right'],view_top:['上','Top'],view_bottom:['下','Bottom'],manual:['人工','Manual'],inlet:['入口','Inlet']};
const ZH2EN={'主动脉入口':'Aortic inlet','入口（主动脉）':'Inlet (aorta)','主动脉':'Aorta','左髂总':'Left CIA','右髂总':'Right CIA','左髂外':'Left EIA','右髂外':'Right EIA','左髂内':'Left IIA','右髂内':'Right IIA','入口':'Inlet',
  '高 WSS 区':'high-WSS region','低 WSS 区':'low-WSS region','全场最大值':'Field maximum','全场最大 WSS':'Field maximum WSS','最大直径':'Max diameter','最小半径':'Min radius','最大速度':'Max speed','最低压力':'Min pressure','压降':'Pressure drop','低速区':'Low-speed region',
  '前':'Front','后':'Back','左':'Left','右':'Right','上':'Top','下':'Bottom','人工':'Manual','标注':'Annotation','测量':'Measurement','探针':'Probe','分支':'Branch','半径':'Radius','弧长':'Arc length','管径':'Diameter'};
const ZH2EN_SORTED=Object.entries(ZH2EN).sort((a,b)=>b[0].length-a[0].length);
function englishLabel(key,lang){
  if(key===null||key===undefined)return '';const k=String(key),en=lang==='en';
  if(Object.prototype.hasOwnProperty.call(LABELS,k))return LABELS[k][en?1:0];
  if(!en)return k;
  if(Object.prototype.hasOwnProperty.call(ZH2EN,k))return ZH2EN[k];
  let out=k;for(const [zh,e] of ZH2EN_SORTED)if(out.includes(zh))out=out.split(zh).join(e+' ');
  return out.replace(/\s+/g,' ').trim();
}
function safeName(x){return String(x??'case').replace(/[^\w\u3400-\u9fff-]+/g,'_').replace(/^_+|_+$/g,'')||'case';}
function exportFilename(o){o=o||{};const k=Math.max(1,Math.round(+o.scale||1)),ext=String(o.ext||'png').replace(/^\./,'');return safeName(o.case_id)+'_'+safeName(o.view||'custom')+'_'+safeName(o.field||'wss')+'_'+k+'x.'+ext;}
// ---------------------------------------------------------------- §12.4 built-in presets
function _aortaSid(ctx){
  const names=(ctx&&ctx.branchNames)||{};for(const [sid,name] of Object.entries(names))if(name==='主动脉')return Number(sid);
  for(const [sid,name] of Object.entries(names))if(typeof name==='string'&&/主动脉|aorta/i.test(name))return Number(sid);
  const prof=ctx&&ctx.profiles&&Array.isArray(ctx.profiles.branches)?ctx.profiles.branches:[];const root=prof.find(b=>b.parent_id===null||b.parent_id===-1||b.parent_id===undefined);
  return root?Number(root.segment_id):-1;
}
function _findings(ctx){const f=ctx&&ctx.findings;if(Array.isArray(f))return f;if(f&&Array.isArray(f.items))return f.items;const m=ctx&&ctx.meta;return m&&m.findings&&Array.isArray(m.findings.items)?m.findings.items:[];}
function _profileBranch(ctx,sid){const prof=ctx&&ctx.profiles&&Array.isArray(ctx.profiles.branches)?ctx.profiles.branches:(ctx&&ctx.meta&&ctx.meta.profiles&&Array.isArray(ctx.meta.profiles.branches)?ctx.meta.profiles.branches:[]);return prof.find(b=>Number(b.segment_id)===sid)||prof[0]||null;}
function _cam(ctx,name){const sv=ctx&&ctx.standardViews;return sv&&sv[name]?sv[name]:undefined;}
const PRESETS={
  wall:[
    {name:'瘤囊低 WSS 区',description:'固定 0–2 Pa 色标、六段离散、等值线、只看主动脉、前视',build(ctx){const a=_aortaSid(ctx);return {preset_name:'瘤囊低 WSS 区',mode:'wss',range:{mode:'fixed',min:0,max:2},log:false,bands:6,overlay:{contours:true,trust:false},highlight:{branch:a,top:false,peak:true,feature:'wss'},camera:_cam(ctx,'front')};}},
    {name:'髂分叉热点',description:'飞到发现列表第一条高 WSS 簇并高亮最高 1%',build(ctx){const f=_findings(ctx);const hit=f.find(x=>x&&x.kind==='high_wss_cluster')||f.find(x=>x&&x.kind==='max_wss')||null;return {preset_name:'髂分叉热点',mode:'wss',range:{mode:'case'},highlight:{branch:-1,top:true,top_pct:1,peak:true,feature:'wss',finding:hit?hit.id:null}};}},
    {name:'主动脉沿程',description:'展开沿程菜单，只看主动脉，高亮 p99 最高的 s 带',build(ctx){const a=_aortaSid(ctx),b=_profileBranch(ctx,a);let s=null;if(b&&b.wss&&Array.isArray(b.wss.p99_pa)){let best=-Infinity;b.wss.p99_pa.forEach((v,i)=>{if(Number.isFinite(v)&&v>best){best=v;s=(b.s_from_root_mm||[])[i];}});}
      return {preset_name:'主动脉沿程',mode:'wss',highlight:{branch:a,top:false,peak:false,feature:'wss'},ui:{menu:'profiles',profile:{branch:b?Number(b.segment_id):a,x:'s_from_root_mm',s:Number.isFinite(s)?s:null}},camera:_cam(ctx,'front')};}},
    {name:'临床视图',description:'前视、彩虹色标、默认阈值，三维上钉出前 5 条发现与分支名',build(ctx){return {preset_name:'临床视图',mode:'wss',colormap:'rainbow',range:{mode:'case'},log:false,bands:0,thresholds_pa:[0.4,4,7],opacity:1,overlay:{trust:false,contours:false},highlight:{branch:-1,top:false,peak:false},labels:{findings:5,branches:true},camera:_cam(ctx,'front')};}}],
  volume:[
    {name:'沿程压降',description:'展开沿程菜单显示主动脉压力曲线',build(ctx){const a=_aortaSid(ctx);return {preset_name:'沿程压降',mode:'pressure',field:'pressure',ui:{menu:'profiles',profile:{field:'pressure',branch:a,x:'s_from_root_mm'}}};}},
    {name:'流线全貌',description:'流线模式、壁面透明度 0.08、前视',build(ctx){return {preset_name:'流线全貌',mode:'streamlines',field:'velocity',opacity:.08,camera:_cam(ctx,'front')};}},
    {name:'瘤囊截面系列',description:'主动脉 40 / 60 / 80% 弧长自动截面',build(ctx){const a=_aortaSid(ctx);return {preset_name:'瘤囊截面系列',mode:'slice',field:'velocity',slice:{kind:'auto',branch:a,fractions:[.4,.6,.8]}};}},
    {name:'临床视图',description:'前视、体内速度点云、彩虹色标，三维上钉出前 5 条发现与分支名',build(ctx){return {preset_name:'临床视图',mode:'cloud',field:'velocity',colormap:'rainbow',bands:0,opacity:.12,overlay:{trust:false},labels:{findings:5,branches:true},camera:_cam(ctx,'front')};}}]
};
function builtinPresets(family,meta){const list=PRESETS[family]||[];return list.map(p=>({name:p.name,description:p.description,family,builtin:true,build:ctx=>{const out=p.build(Object.assign({meta},ctx||{}));for(const k of Object.keys(out))if(out[k]===undefined)delete out[k];return out;}}));}
// ---------------------------------------------------------------- DOM / three helpers (fail soft)
function renderOffscreen(o){
  o=o||{};const renderer=o.renderer,scene=o.scene,camera=o.camera;if(!renderer||!scene||!camera)throw new Error('renderOffscreen needs renderer, scene and camera');
  const T=o.THREE||root.THREE||null,dom=renderer.domElement;
  const cssW=Math.max(1,Math.round(o.width||parseFloat(dom.style&&dom.style.width)||dom.clientWidth||dom.width||1)),cssH=Math.max(1,Math.round(o.height||parseFloat(dom.style&&dom.style.height)||dom.clientHeight||dom.height||1));
  const prevRatio=typeof renderer.getPixelRatio==='function'?renderer.getPixelRatio():1,prevAspect=camera.aspect,prevBg=scene.background;
  let prevClear=null,prevAlpha=1;
  try{if(T&&T.Color&&typeof renderer.getClearColor==='function'){prevClear=renderer.getClearColor(new T.Color());prevAlpha=typeof renderer.getClearAlpha==='function'?renderer.getClearAlpha():1;}}catch(_){prevClear=null;}
  const attempt=k=>{
    const w=Math.round(cssW*k),h=Math.round(cssH*k);
    try{
      renderer.setPixelRatio(1);renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix();
      if(o.background==='white'){scene.background=null;renderer.setClearColor(0xffffff,1);}
      else if(o.background==='transparent'){scene.background=null;renderer.setClearColor(0x000000,0);}
      renderer.render(scene,camera);
      if(typeof o.compose==='function')o.compose(dom,{width:w,height:h,scale:k});
      const url=dom.toDataURL('image/png');
      if(!url||url.length<32)throw new Error('empty canvas');
      return {dataURL:url,width:w,height:h,scale:k};
    }finally{
      try{scene.background=prevBg;if(prevClear)renderer.setClearColor(prevClear,prevAlpha);}catch(_){}
      try{renderer.setPixelRatio(prevRatio);renderer.setSize(cssW,cssH,false);camera.aspect=prevAspect;camera.updateProjectionMatrix();renderer.render(scene,camera);}catch(_){}
    }
  };
  const k=Math.max(1,Math.min(8,+o.scale||1));
  try{return Object.assign(attempt(k),{downgraded:false});}
  catch(err){if(k<=1)throw err;const out=attempt(1);out.downgraded=true;out.error=String(err&&err.message||err);return out;}
}
function glossaryPopover(o){
  o=o||{};const terms=(o.glossary&&o.glossary.terms)||o.glossary||{};let lang=o.lang||'zh',box=null,bound=null;
  const D=typeof document!=='undefined'?document:null;
  function ensure(){if(box||!D||!D.createElement)return box;box=D.createElement('div');box.className='gloss-pop';box.hidden=true;box.setAttribute('role','dialog');
    const mount=o.mount||D.body;if(mount&&mount.appendChild)mount.appendChild(box);return box;}
  function hide(){if(box)box.hidden=true;}
  function show(key,anchor){const t=terms[key];const b=ensure();if(!b)return false;if(!t){b.textContent=(lang==='en'?'No entry for ':'没有术语条目：')+key;}
    else{const name=lang==='en'?(t.en||t.zh):(t.zh||t.en),desc=lang==='en'?(t.en_desc||t.zh_desc):(t.zh_desc||t.en_desc);b.textContent='';
      if(D.createElement){const h=D.createElement('b');h.textContent=name||key;const p=D.createElement('span');p.textContent=desc||'';b.appendChild(h);b.appendChild(p);}else b.textContent=name+'：'+desc;}
    b.hidden=false;
    try{if(anchor&&anchor.getBoundingClientRect){const r=anchor.getBoundingClientRect();b.style.left=Math.max(6,r.left)+'px';b.style.top=(r.bottom+6+(root.scrollY||0))+'px';}}catch(_){}
    return true;}
  function bind(rootEl){const target=rootEl||D;if(!target||!target.addEventListener||bound)return;bound=true;
    target.addEventListener('click',ev=>{const t=ev.target;const btn=t&&t.closest?t.closest('[data-gloss]'):null;if(btn){ev.preventDefault&&ev.preventDefault();ev.stopPropagation&&ev.stopPropagation();if(box&&!box.hidden&&box.dataset&&box.dataset.key===btn.dataset.gloss){hide();return;}show(btn.dataset.gloss,btn);if(box&&box.dataset)box.dataset.key=btn.dataset.gloss;return;}if(box&&!box.hidden&&!(box.contains&&box.contains(t)))hide();});
    target.addEventListener('keydown',ev=>{if(ev.key==='Escape')hide();});}
  return {show,hide,bind,setLang(l){lang=l||'zh';},get lang(){return lang;},get element(){return box;}};
}
function viewMessaging(o){
  o=o||{};const W=typeof window!=='undefined'?window:null,L=typeof location!=='undefined'?location:null;
  const enabled=!!(W&&W.parent&&W.parent!==W&&L&&/^https?:$/.test(String(L.protocol||'')));
  const post=msg=>{if(!enabled)return false;try{W.parent.postMessage(msg,L.origin);return true;}catch(_){return false;}};
  const api={enabled,
    ready(extra){return post(Object.assign({type:'wss-view:ready',family:o.family||null,run_identity:o.runIdentity||null,case_id:o.caseId||null,webgl:o.webgl!==false},extra||{}));},
    postApplied(id){return post({type:'wss-view:applied',request_id:id??null});},
    postExported(id,r){return post(Object.assign({type:'wss-view:exported',request_id:id??null},r||{}));},
    postError(id,message){return post({type:'wss-view:error',request_id:id??null,message:String(message&&message.message||message||'error')});}};
  if(enabled&&typeof W.addEventListener==='function')W.addEventListener('message',ev=>{
    if(!ev||ev.origin!==L.origin)return;const d=ev.data;if(!d||typeof d!=='object')return;
    if(d.type==='wss-view:apply-state'){Promise.resolve().then(()=>o.onApplyState?o.onApplyState(d.state||{}):null).then(()=>api.postApplied(d.request_id),err=>api.postError(d.request_id,err));}
    else if(d.type==='wss-view:export'){Promise.resolve().then(()=>o.onExport?o.onExport(d.options||{}):null).then(r=>{if(!r)throw new Error('export unavailable');api.postExported(d.request_id,r);},err=>api.postError(d.request_id,err));}
  });
  return api;
}
// ---------------------------------------------------------------- §15 cross-sections from the wall mesh (2026-09-21)
// Plane ∩ triangle-mesh contour, chained into loops; the local lumen loop; polygon metrics (area, Feret diameters).
const _d3=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2],_s3=(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]],_x3=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
function _u3(v){const n=Math.hypot(v[0],v[1],v[2])||1;return [v[0]/n,v[1]/n,v[2]/n];}
function planeFrame(origin,normal){const n=_u3(normal),ref=Math.abs(n[2])<0.9?[0,0,1]:[0,1,0],u=_u3(_x3(ref,n)),v=_u3(_x3(n,u));return {origin:[+origin[0],+origin[1],+origin[2]],normal:n,u,v};}
function planeContour(vertices,faces,plane,wallValues,maxRadius){
  const n=_u3(plane.normal),o=plane.origin,u=plane.u,v=plane.v,nv=Math.floor(vertices.length/3),d=new Float32Array(nv),segs=[];
  for(let i=0;i<nv;i++)d[i]=(vertices[3*i]-o[0])*n[0]+(vertices[3*i+1]-o[1])*n[1]+(vertices[3*i+2]-o[2])*n[2];
  const crossing=(a,b)=>{const da=d[a],db=d[b];if((da>=0)===(db>=0))return null;const t=da/(da-db),key=a<b?a*nv+b:b*nv+a;
    const q=[vertices[3*a]+(vertices[3*b]-vertices[3*a])*t-o[0],vertices[3*a+1]+(vertices[3*b+1]-vertices[3*a+1])*t-o[1],vertices[3*a+2]+(vertices[3*b+2]-vertices[3*a+2])*t-o[2]];
    let val=NaN;if(wallValues){const wa=wallValues[a],wb=wallValues[b];if(Number.isFinite(wa)&&Number.isFinite(wb))val=wa+(wb-wa)*t;}
    return {x:_d3(q,u),y:_d3(q,v),val,key};};
  const r2=Number.isFinite(maxRadius)&&maxRadius>0?maxRadius*maxRadius:Infinity;
  for(let t=0;t+2<faces.length;t+=3){const a=faces[t],b=faces[t+1],c=faces[t+2],p=[crossing(a,b),crossing(b,c),crossing(c,a)].filter(Boolean);
    if(p.length!==2)continue;const mx=(p[0].x+p[1].x)/2,my=(p[0].y+p[1].y)/2;if(mx*mx+my*my>r2)continue;
    segs.push([p[0].x,p[0].y,p[1].x,p[1].y,(p[0].val+p[1].val)/2,p[0].key,p[1].key]);}
  return segs;
}
function contourLoops(segs){
  const byKey=new Map();segs.forEach((sg,i)=>{for(const k of [sg[5],sg[6]]){let a=byKey.get(k);if(!a){a=[];byKey.set(k,a);}a.push(i);}});
  const used=new Uint8Array(segs.length),loops=[];
  const walk=(from,exitKey,stopKey,out)=>{let seg=from,key=exitKey;
    for(let guard=0;guard<=segs.length;guard++){if(key===stopKey&&seg!==from)return true;const next=(byKey.get(key)||[]).find(j=>!used[j]);if(next===undefined)return key===stopKey&&seg!==from;
      used[next]=1;out.push(next);seg=next;key=segs[next][5]===key?segs[next][6]:segs[next][5];}
    return false;};
  for(let s0=0;s0<segs.length;s0++){if(used[s0])continue;used[s0]=1;const members=[s0],A=segs[s0][5],B=segs[s0][6];const closed=walk(s0,B,A,members);if(!closed)walk(s0,A,B,members);loops.push({segments:members,closed});}
  return loops;
}
function selectLoop(loops,segs,origin,maxDist){
  const px=origin?origin[0]:0,py=origin?origin[1]:0,limit=Number.isFinite(maxDist)&&maxDist>0?maxDist:Infinity;let best=null;const rank=c=>c.inside?3:c.wraps?2:1;
  for(const loop of loops){if(loop.segments.length<3&&loops.length>1)continue;
    const ss=loop.segments.map(i=>segs[i]);let crossings=0,dmin=Infinity,minx=Infinity,maxx=-Infinity,miny=Infinity,maxy=-Infinity;
    for(const sg of ss){const [x0,y0,x1,y1]=sg;if((y0<=py)!==(y1<=py)){const xi=x0+(x1-x0)*(py-y0)/(y1-y0);if(xi>px)crossings++;}
      dmin=Math.min(dmin,Math.hypot(x0-px,y0-py),Math.hypot(x1-px,y1-py));minx=Math.min(minx,x0,x1);maxx=Math.max(maxx,x0,x1);miny=Math.min(miny,y0,y1);maxy=Math.max(maxy,y0,y1);}
    let span=2*Math.PI;if(!loop.closed){const angs=ss.map(sg=>Math.atan2((sg[1]+sg[3])/2-py,(sg[0]+sg[2])/2-px)).sort((a,b)=>a-b);let gap=0;for(let i=0;i<angs.length;i++){const next=i+1<angs.length?angs[i+1]:angs[0]+2*Math.PI;gap=Math.max(gap,next-angs[i]);}span=2*Math.PI-gap;}
    const inside=loop.closed&&crossings%2===1;if(!inside&&dmin>limit)continue;
    const cand={segs:ss,inside,dmin,area:(maxx-minx)*(maxy-miny),closed:loop.closed,bbox:[minx,maxx,miny,maxy],span,wraps:!loop.closed&&span>Math.PI};
    const better=!best||rank(cand)>rank(best)||(rank(cand)===rank(best)&&(cand.inside?cand.area<best.area:cand.wraps?(cand.span>best.span+1e-6||(Math.abs(cand.span-best.span)<=1e-6&&cand.dmin<best.dmin)):cand.dmin<best.dmin));
    if(better)best=cand;}
  return best;
}
function closeChain(ss){
  const count=new Map();for(const sg of ss)for(const k of [sg[5],sg[6]])count.set(k,(count.get(k)||0)+1);
  const ends=[];for(const sg of ss){if(count.get(sg[5])===1)ends.push([sg[0],sg[1]]);if(count.get(sg[6])===1)ends.push([sg[2],sg[3]]);}
  if(ends.length!==2)return null;let length=0;for(const sg of ss)length+=Math.hypot(sg[2]-sg[0],sg[3]-sg[1]);
  const gap=Math.hypot(ends[1][0]-ends[0][0],ends[1][1]-ends[0][1]);if(!(gap<=Math.max(0.6*length,1e-6)))return null;
  return ss.concat([[ends[0][0],ends[0][1],ends[1][0],ends[1][1],NaN,-1,-1]]);
}
function pointInLoop(segs,x,y){let inside=false;for(const sg of segs){const [x0,y0,x1,y1]=sg;if((y0<=y)!==(y1<=y)){const xi=x0+(x1-x0)*(y-y0)/(y1-y0);if(xi>x)inside=!inside;}}return inside;}
// Ordered polygon (in-plane xy) from a loop's segment soup: chain by matching endpoints (tolerance = 1e-6 × extent).
function loopPolygon(segs){
  if(!segs||!segs.length)return [];const left=segs.slice();let cur=left.shift();const pts=[[cur[0],cur[1]],[cur[2],cur[3]]];let tail=pts[1];
  let ext=1;for(const sg of segs)ext=Math.max(ext,Math.abs(sg[0]),Math.abs(sg[1]),Math.abs(sg[2]),Math.abs(sg[3]));const tol=ext*1e-5;
  while(left.length){let bi=-1,bd=Infinity,flip=false;
    for(let i=0;i<left.length;i++){const sg=left[i],d0=Math.hypot(sg[0]-tail[0],sg[1]-tail[1]),d1=Math.hypot(sg[2]-tail[0],sg[3]-tail[1]);if(d0<bd){bd=d0;bi=i;flip=false;}if(d1<bd){bd=d1;bi=i;flip=true;}}
    if(bi<0||bd>Math.max(tol,ext*0.05))break;const sg=left.splice(bi,1)[0];tail=flip?[sg[0],sg[1]]:[sg[2],sg[3]];pts.push(tail);}
  if(pts.length>2&&Math.hypot(pts[0][0]-tail[0],pts[0][1]-tail[1])<=Math.max(tol,ext*0.05))pts.pop();
  return pts;
}
// Area (shoelace), perimeter, centroid, max/min Feret diameter, equivalent diameter, circularity of a closed polygon.
function sectionMetrics(poly){
  const n=poly?poly.length:0;if(n<3)return null;let a2=0,per=0,cx=0,cy=0;
  for(let i=0;i<n;i++){const [x0,y0]=poly[i],[x1,y1]=poly[(i+1)%n];const w=x0*y1-x1*y0;a2+=w;cx+=(x0+x1)*w;cy+=(y0+y1)*w;per+=Math.hypot(x1-x0,y1-y0);}
  const area=Math.abs(a2)/2;if(a2!==0){cx/=3*a2;cy/=3*a2;}else{cx=poly.reduce((s,p)=>s+p[0],0)/n;cy=poly.reduce((s,p)=>s+p[1],0)/n;}
  let dmax=0;for(let i=0;i<n;i++)for(let j=i+1;j<n;j++)dmax=Math.max(dmax,Math.hypot(poly[j][0]-poly[i][0],poly[j][1]-poly[i][1]));
  let dmin=Infinity;for(let k=0;k<90;k++){const t=k*Math.PI/90,c=Math.cos(t),s=Math.sin(t);let lo=Infinity,hi=-Infinity;for(const p of poly){const w=p[0]*c+p[1]*s;lo=Math.min(lo,w);hi=Math.max(hi,w);}dmin=Math.min(dmin,hi-lo);}
  return {area_mm2:area,perimeter_mm:per,centroid:[cx,cy],max_diameter_mm:dmax,min_diameter_mm:dmin,equivalent_diameter_mm:2*Math.sqrt(area/Math.PI),circularity:per>0?4*Math.PI*area/(per*per):0,n_points:n};
}
// One call for reports: the local cross-section of the wall mesh at (origin, normal) and its geometry.
function crossSection(o){
  const plane=planeFrame(o.origin,o.normal),segs=planeContour(o.vertices,o.faces,plane,o.wallValues||null,Infinity);
  if(!segs.length)return {plane,segs:[],polygon:[],polygon_world:[],metrics:null,closed:false,synthetic:false,open:false,found:false};
  const loop=selectLoop(contourLoops(segs),segs,[0,0],Number.isFinite(o.maxDist)&&o.maxDist>0?o.maxDist:Infinity);
  if(!loop)return {plane,segs,polygon:[],polygon_world:[],metrics:null,closed:false,synthetic:false,open:false,found:false};
  let contour=loop.segs,synthetic=false,open=false;
  if(!loop.closed){const c=closeChain(contour);if(c){contour=c;synthetic=true;}else open=true;}
  const polygon=loopPolygon(contour),metrics=open?null:sectionMetrics(polygon);
  const world=polygon.map(q=>[plane.origin[0]+plane.u[0]*q[0]+plane.v[0]*q[1],plane.origin[1]+plane.u[1]*q[0]+plane.v[1]*q[1],plane.origin[2]+plane.u[2]*q[0]+plane.v[2]*q[1]]);
  return {plane,segs:contour,polygon,polygon_world:world,metrics,closed:loop.closed||synthetic,synthetic,open,found:true,dmin:loop.dmin};
}
// ---------------------------------------------------------------- §15 figure helpers: montage, curve SVG, CSV
function _xmlEsc(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
// Panels ({image, label, caption}) laid out in a grid on one canvas with optional title, a shared colour-bar
// image and a scale bar.  ``image`` must be a drawable (Image/canvas); use loadImage() for data URLs.
function composeMontage(o){
  const D=o.document||(typeof document!=='undefined'?document:null);if(!D||!D.createElement)return null;
  const panels=(o.panels||[]).filter(p=>p&&p.image);if(!panels.length)return null;
  const cols=Math.max(1,Math.min(Number(o.columns)||Math.ceil(Math.sqrt(panels.length)),panels.length||1)),rows=Math.ceil(panels.length/cols);
  const gap=Number(o.gap)||16,font=Number(o.font)||22,pad=Number(o.pad)||24,title=o.title||'';
  const pw=Math.max(...panels.map(p=>p.image.width||p.image.naturalWidth||1)),ph=Math.max(...panels.map(p=>p.image.height||p.image.naturalHeight||1));
  const capH=panels.some(p=>p.caption)?Math.round(font*1.6):0,titleH=title?Math.round(font*2.2):0;
  const cbar=o.colorbar&&o.colorbar.image?o.colorbar:null,cbW=cbar?(cbar.image.width||cbar.image.naturalWidth):0,cbH=cbar?(cbar.image.height||cbar.image.naturalHeight):0;
  const W=pad*2+cols*pw+(cols-1)*gap+(cbar?gap+cbW:0),H=pad*2+titleH+rows*(ph+capH)+(rows-1)*gap;
  const c=D.createElement('canvas');c.width=W;c.height=Math.max(H,cbar?pad*2+titleH+cbH:0);const ctx=c.getContext('2d');if(!ctx)return null;
  ctx.fillStyle=o.background||'#ffffff';ctx.fillRect(0,0,c.width,c.height);ctx.fillStyle=o.ink||'#20374d';ctx.font=`600 ${font}px sans-serif`;ctx.textBaseline='top';
  if(title){ctx.textAlign='left';ctx.fillText(title,pad,pad);}
  panels.forEach((p,i)=>{const col=i%cols,row=Math.floor(i/cols),x=pad+col*(pw+gap),y=pad+titleH+row*(ph+capH+gap);
    ctx.drawImage(p.image,x,y,pw,ph);
    if(p.label!==undefined&&p.label!==null&&p.label!==''){ctx.font=`700 ${font}px sans-serif`;ctx.fillStyle='#ffffffd9';const lw=ctx.measureText(String(p.label)).width+font*.8;ctx.fillRect(x+6,y+6,lw,font*1.4);ctx.fillStyle=o.ink||'#20374d';ctx.textAlign='left';ctx.fillText(String(p.label),x+6+font*.4,y+6+font*.2);}
    if(p.caption){ctx.font=`${Math.round(font*.85)}px sans-serif`;ctx.fillStyle=o.muted||'#536a80';ctx.textAlign='center';ctx.fillText(String(p.caption),x+pw/2,y+ph+font*.35,pw);}});
  if(cbar){ctx.drawImage(cbar.image,W-pad-cbW,pad+titleH,cbW,cbH);}
  if(o.scaleBar&&o.scaleBar.px>0){const L=o.scaleBar.mm,px=o.scaleBar.px,x0=pad+8,y0=c.height-pad-8;ctx.strokeStyle=o.ink||'#20374d';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(x0,y0);ctx.lineTo(x0+px,y0);ctx.moveTo(x0,y0-6);ctx.lineTo(x0,y0+6);ctx.moveTo(x0+px,y0-6);ctx.lineTo(x0+px,y0+6);ctx.stroke();ctx.font=`${Math.round(font*.85)}px sans-serif`;ctx.fillStyle=o.ink||'#20374d';ctx.textAlign='left';ctx.fillText(`${L} mm`,x0,y0-font*1.1);}
  return c;
}
function loadImage(src,doc){return new Promise((resolve,reject)=>{const I=(doc&&doc.defaultView&&doc.defaultView.Image)||(typeof Image!=='undefined'?Image:null);if(!I)return reject(new Error('no Image'));const im=new I();im.onload=()=>resolve(im);im.onerror=()=>reject(new Error('image load failed'));im.src=src;});}
// Line chart as a standalone SVG string: series {name,x[],y[],color,dash}, axis labels, optional shaded bands.
function profileSVG(o){
  const W=Number(o.width)||900,H=Number(o.height)||420,ml=64,mr=18,mt=o.title?40:16,mb=52,pw=W-ml-mr,ph=H-mt-mb,font=Number(o.font)||13;
  const series=(o.series||[]).map(s=>({...s,pts:(s.x||[]).map((x,i)=>[+x,+(s.y||[])[i]]).filter(p=>Number.isFinite(p[0])&&Number.isFinite(p[1]))})).filter(s=>s.pts.length);
  let xmin=Infinity,xmax=-Infinity,ymin=Infinity,ymax=-Infinity;for(const s of series)for(const [x,y] of s.pts){xmin=Math.min(xmin,x);xmax=Math.max(xmax,x);ymin=Math.min(ymin,y);ymax=Math.max(ymax,y);}
  if(!(xmax>xmin)){xmin=0;xmax=1;}if(!(ymax>ymin)){ymin=0;ymax=1;}if(o.yZero!==false&&ymin>0)ymin=0;const yr=ymax-ymin;ymax+=yr*0.05;
  const sx=x=>ml+(x-xmin)/(xmax-xmin)*pw,sy=y=>mt+ph-(y-ymin)/(ymax-ymin)*ph;
  const ticks=(lo,hi,n)=>{const span=hi-lo,raw=span/n,mag=Math.pow(10,Math.floor(Math.log10(raw))),step=[1,2,5,10].map(m=>m*mag).find(v=>v>=raw)||mag;const out=[];for(let v=Math.ceil(lo/step)*step;v<=hi+1e-9;v+=step)out.push(+v.toFixed(6));return out;};
  const fmtT=v=>Math.abs(v)>=100?v.toFixed(0):Math.abs(v)>=10?v.toFixed(1):v.toFixed(2).replace(/\.?0+$/,'');
  const parts=[`<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" font-family="sans-serif" font-size="${font}">`,`<rect width="${W}" height="${H}" fill="${_xmlEsc(o.background||'#ffffff')}"/>`];
  for(const b of (o.bands||[]))parts.push(`<rect x="${sx(b.x0).toFixed(1)}" y="${mt}" width="${Math.max(0,sx(b.x1)-sx(b.x0)).toFixed(1)}" height="${ph}" fill="${_xmlEsc(b.color||'#e8f1fa')}"/>${b.label?`<text x="${((sx(b.x0)+sx(b.x1))/2).toFixed(1)}" y="${mt+font}" text-anchor="middle" fill="#536a80">${_xmlEsc(b.label)}</text>`:''}`);
  for(const v of ticks(xmin,xmax,6))parts.push(`<line x1="${sx(v).toFixed(1)}" y1="${mt}" x2="${sx(v).toFixed(1)}" y2="${mt+ph}" stroke="#e5ebf1"/><text x="${sx(v).toFixed(1)}" y="${mt+ph+font+6}" text-anchor="middle" fill="#3f5a72">${fmtT(v)}</text>`);
  for(const v of ticks(ymin,ymax,5))parts.push(`<line x1="${ml}" y1="${sy(v).toFixed(1)}" x2="${ml+pw}" y2="${sy(v).toFixed(1)}" stroke="#e5ebf1"/><text x="${ml-8}" y="${(sy(v)+font*.35).toFixed(1)}" text-anchor="end" fill="#3f5a72">${fmtT(v)}</text>`);
  parts.push(`<rect x="${ml}" y="${mt}" width="${pw}" height="${ph}" fill="none" stroke="#8093a2"/>`);
  series.forEach((s,i)=>{const d=s.pts.map((p,k)=>(k?'L':'M')+sx(p[0]).toFixed(1)+' '+sy(p[1]).toFixed(1)).join(' ');parts.push(`<path d="${d}" fill="none" stroke="${_xmlEsc(s.color||['#176ea2','#c0392b','#3f8f6b','#7d5ba6','#d97706'][i%5])}" stroke-width="${s.width||2}"${s.dash?` stroke-dasharray="${_xmlEsc(s.dash)}"`:''}/>`);});
  if(o.xLabel)parts.push(`<text x="${ml+pw/2}" y="${H-10}" text-anchor="middle" fill="#20374d">${_xmlEsc(o.xLabel)}</text>`);
  if(o.yLabel)parts.push(`<text transform="translate(16 ${mt+ph/2}) rotate(-90)" text-anchor="middle" fill="#20374d">${_xmlEsc(o.yLabel)}</text>`);
  if(o.title)parts.push(`<text x="${ml}" y="${font+10}" font-weight="600" font-size="${font+3}" fill="#20374d">${_xmlEsc(o.title)}</text>`);
  series.forEach((s,i)=>{const x=ml+pw-160,y=mt+12+i*(font+6);parts.push(`<line x1="${x}" y1="${y}" x2="${x+22}" y2="${y}" stroke="${_xmlEsc(s.color||['#176ea2','#c0392b','#3f8f6b','#7d5ba6','#d97706'][i%5])}" stroke-width="2"${s.dash?` stroke-dasharray="${_xmlEsc(s.dash)}"`:''}/><text x="${x+28}" y="${y+font*.35}" fill="#20374d">${_xmlEsc(s.name||'')}</text>`);});
  parts.push('</svg>');return parts.join('');
}
// Generic CSV (BOM + CRLF, RFC 4180 quoting); ``comments`` become leading "# …" lines.
function tableToCSV(columns,rows,comments){
  const cell=x=>{const s=x===null||x===undefined?'':(typeof x==='number'?(Number.isFinite(x)?String(+x.toPrecision(7)):''):String(x));return /[",\r\n]/.test(s)?'"'+s.replace(/"/g,'""')+'"':s;};
  const lines=(comments||[]).map(c=>'# '+String(c).replace(/[\r\n]+/g,' '));lines.push(columns.map(cell).join(','));
  for(const r of (rows||[]))lines.push((Array.isArray(r)?r:columns.map(k=>r[k])).map(cell).join(','));
  return '﻿'+lines.join('\r\n')+'\r\n';
}
const common={buildCenterlineGroups,projectToCenterline,arcDistance,arcFromRoot,localDiameter,straightDistance,newId,measurementLabel,probeToTSV,probeToCSV,
  colormapNames,colormapStops,colorAt,colorbarSVG,englishLabel,exportFilename,safeName,builtinPresets,renderOffscreen,glossaryPopover,viewMessaging,LABELS,ZH2EN,
  planeFrame,planeContour,contourLoops,selectLoop,closeChain,pointInLoop,loopPolygon,sectionMetrics,crossSection,composeMontage,loadImage,profileSVG,tableToCSV};
root.WssReportCommon=common;
if(typeof module!=='undefined'&&module.exports)module.exports=common;
})(typeof globalThis!=='undefined'?globalThis:this);
