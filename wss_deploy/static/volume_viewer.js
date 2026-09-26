/* Offline volume viewer. The numerical helpers are also exported for Node tests. */
(function (root) {
  'use strict';
  const add = (a,b) => a.map((v,i)=>v+b[i]);
  const sub = (a,b) => a.map((v,i)=>v-b[i]);
  const mul = (a,s) => a.map(v=>v*s);
  const dot = (a,b) => a.reduce((s,v,i)=>s+v*b[i],0);
  const cross = (a,b) => [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
  const norm = a => Math.hypot(...a);
  const unit = a => norm(a)>1e-10 ? mul(a,1/norm(a)) : [0,0,1];
  const point = (a,i) => [a[3*i],a[3*i+1],a[3*i+2]];
  const clamp = (x,a,b) => Math.max(a,Math.min(b,x));
  function planeBasis(normal) {
    const n=unit(normal), reference=Math.abs(n[2])<0.9 ? [0,0,1] : [0,1,0];
    const u=unit(cross(reference,n));
    return {normal:n,u,v:unit(cross(n,u))};
  }
  function rotate(v,axis,degrees) {
    const k=unit(axis), a=degrees*Math.PI/180;
    return add(add(mul(v,Math.cos(a)),mul(cross(k,v),Math.sin(a))),mul(k,dot(k,v)*(1-Math.cos(a))));
  }
  function rotatePlane(normal,pitch,yaw) {
    const initial=planeBasis(normal), n1=rotate(initial.normal,initial.u,pitch);
    return planeBasis(rotate(n1,rotate(initial.v,initial.u,pitch),yaw));
  }
  function groupCenterline(xyz,segments,tangents,edges=null) {
    const groups=new Map();
    for(let i=0;i<xyz.length/3;i++) {
      const sid=segments ? segments[i] : 0;
      if(!groups.has(sid)) groups.set(sid,{segment:sid,rows:[],points:[],tangents:[],arc:[0]});
      groups.get(sid).rows.push(i);
    }
    for(const g of groups.values()) {
      // Exported edge directions preserve sample_index ordering even if rows
      // in the atlas are interleaved or otherwise not stored in path order.
      if(edges&&edges.length&&g.rows.length>1) {
        const members=new Set(g.rows),next=new Map(),preceded=new Set();
        for(let i=0;i<edges.length;i+=2) if(members.has(edges[i])&&members.has(edges[i+1])) {next.set(edges[i],edges[i+1]);preceded.add(edges[i+1]);}
        const starts=g.rows.filter(i=>!preceded.has(i)),ordered=[],seen=new Set();
        if(starts.length===1) {let i=starts[0];while(i!==undefined&&!seen.has(i)){ordered.push(i);seen.add(i);i=next.get(i);}}
        if(ordered.length===g.rows.length) g.rows=ordered;
      }
      for(const i of g.rows) {
        const p=point(xyz,i);
        if(g.points.length)g.arc.push(g.arc[g.arc.length-1]+norm(sub(p,g.points[g.points.length-1])));
        g.points.push(p);g.tangents.push(tangents?point(tangents,i):null);
      }
    }
    return Array.from(groups.values());
  }
  function centerlinePlane(group,fraction) {
    const p=group.points, target=clamp(fraction,0,1)*group.arc[group.arc.length-1];
    if(!p.length) throw new Error('中心线分支为空');
    if(p.length===1) return {origin:p[0],normal:unit(group.tangents[0]||[0,0,1]),segment:group.segment};
    let j=1;
    while(j<group.arc.length-1 && group.arc[j]<target) j++;
    const f=(target-group.arc[j-1])/Math.max(1e-12,group.arc[j]-group.arc[j-1]);
    const origin=add(p[j-1],mul(sub(p[j],p[j-1]),f));
    const t0=group.tangents[j-1],t1=group.tangents[j];
    const normal=t0&&t1 ? unit(add(mul(t0,1-f),mul(t1,f))) : unit(sub(p[j],p[j-1]));
    return {origin,normal,segment:group.segment};
  }
  function automaticPlanes(groups) {
    const presets=[];
    for(const g of groups) {
      if(g.points.length<2 || g.arc[g.arc.length-1]<=0) continue;
      for(const fraction of [0.05,0.2,0.4,0.6,0.8,0.95]) {
        presets.push({...centerlinePlane(g,fraction),fraction,nearEnd:fraction<0.1 || fraction>0.9});
      }
    }
    return presets;
  }
  // Nearest centreline sample to a world point: its tangent orients a quick cross-section.
  function nearestTangent(groups,p) {
    let best=null;
    for(const g of groups) for(let i=0;i<g.points.length;i++) {
      const d=norm(sub(g.points[i],p));
      if(best===null||d<best.distance) {
        let t=g.tangents[i];
        if(!t) t=i+1<g.points.length ? sub(g.points[i+1],g.points[i]) : i>0 ? sub(g.points[i],g.points[i-1]) : [0,0,1];
        best={distance:d,tangent:unit(t),segment:g.segment,arc:g.arc[i]};
      }
    }
    return best;
  }
  // One pick: cross-section through the point, perpendicular to the local axis.
  // Two picks: the plane containing both points whose normal is as close as
  // possible to the local axis, i.e. the (oblique) cross-section through the
  // chord between them - the natural "cut between these two wall points".
  function planeFromPicks(picks,tangent) {
    const t=unit(tangent||[0,0,1]);
    if(!picks||!picks.length) throw new Error('需要至少一个选点');
    if(picks.length===1) return {origin:picks[0].slice(),normal:t,segment:null,picks:1};
    const p1=picks[0],p2=picks[picks.length-1],chord=sub(p2,p1);
    if(norm(chord)<1e-6) return {origin:p1.slice(),normal:t,segment:null,picks:1};
    const c=unit(chord);
    let n=sub(t,mul(c,dot(t,c)));
    if(norm(n)<1e-6) n=planeBasis(c).u;
    return {origin:mul(add(p1,p2),0.5),normal:unit(n),segment:null,picks:2,chord_mm:norm(chord)};
  }
  function slabIndices(pts,isWall,segments,plane,thickness,segment=null) {
    const indices=[], n=unit(plane.normal), half=thickness/2;
    if(!Number.isFinite(thickness)||thickness<=0) throw new Error('截面厚度必须为正数');
    for(let i=0;i<pts.length/3;i++) {
      if(isWall&&isWall[i]) continue;
      if(segment!==null && segments && segments[i]!==segment) continue;
      if(Math.abs(dot(sub(point(pts,i),plane.origin),n))<=half+1e-6) indices.push(i);
    }
    return indices;
  }
  function insideIndices(pts,isWall) {
    const out=[];
    for(let i=0;i<pts.length/3;i++) if(!isWall||!isWall[i]) out.push(i);
    return out;
  }
  function moduleSummary(segments,isWall) {
    const result=new Map();
    if(!segments) return [];
    for(let i=0;i<segments.length;i++) {
      const id=Number(segments[i]);
      if(!result.has(id)) result.set(id,{segment:id,count:0,interior:0,wall:0});
      const item=result.get(id);item.count++;
      if(isWall&&isWall[i]) item.wall++; else item.interior++;
    }
    return Array.from(result.values()).sort((a,b)=>a.segment-b.segment);
  }
  function moduleIndices(indices,segments,segment) {
    if(segment===null||segment===undefined||segment==='') return indices.slice();
    const id=Number(segment);return indices.filter(i=>segments&&Number(segments[i])===id);
  }
  function sideIndices(indices,pts,plane,side) {
    if(side===null||side===undefined) return indices.slice();
    const n=unit(plane.normal), sign=Number(side)>=0?1:-1;
    return indices.filter(i=>sign*dot(sub(point(pts,i),plane.origin),n)>=-1e-6);
  }
  function interactionDelta(deltaX,deltaY,modifiers={}) {
    const dx=Number(deltaX)||0,dy=Number(deltaY)||0;
    if(modifiers.ctrlKey||modifiers.metaKey||modifiers.altKey) return {zoom:-dy,position:0,offsetU:0};
    if(modifiers.shiftKey||Math.abs(dx)>Math.abs(dy)*1.35) return {zoom:0,position:0,offsetU:dx||dy};
    return {zoom:0,position:dy,offsetU:0};
  }
  // Slice gizmo (drag the plane in the 3-D view): ``screenDir`` is the on-screen direction (pixels, y down)
  // of a world axis at the plane origin, ``mmPerPx`` the world size of one pixel at that depth.
  function dragAlong(dx,dy,screenDir,mmPerPx) {
    const sx=Number(screenDir&&screenDir[0])||0,sy=Number(screenDir&&screenDir[1])||0,len=Math.hypot(sx,sy);
    if(!(len>1e-9))return 0;
    return ((Number(dx)||0)*sx+(Number(dy)||0)*sy)/len*(Number(mmPerPx)||0);
  }
  // A move of ``dMm`` along the plane normal expressed in the position slider's own unit
  // (pick basis: 0.5 mm per unit; centreline basis: % of branch arc; axis basis: % of the bbox extent).
  function positionStep(basis,dMm,arcMm,extentMm) {
    const d=Number(dMm)||0;
    if(basis==='pick')return d*2;
    if(basis==='centerline')return arcMm>0?d/arcMm*100:0;
    return extentMm>0?d/extentMm*100:0;
  }
  function speedField(velocity) {
    const result=new Float32Array(velocity.length/3);
    for(let i=0;i<result.length;i++) result[i]=Math.hypot(velocity[i*3],velocity[i*3+1],velocity[i*3+2]);
    return result;
  }
  function statistics(values,indices) {
    const sorted=indices.map(i=>values[i]).filter(Number.isFinite).sort((a,b)=>a-b);
    if(!sorted.length) return {count:0,mean:null,p99:null,min:null,max:null};
    const at=0.99*(sorted.length-1), lo=Math.floor(at), hi=Math.ceil(at);
    return {count:sorted.length,mean:sorted.reduce((s,v)=>s+v,0)/sorted.length,
      p99:sorted[lo]+(sorted[hi]-sorted[lo])*(at-lo),min:sorted[0],max:sorted[sorted.length-1]};
  }
  // The classic CFD rainbow (blue -> cyan -> green -> yellow -> red) is the default again (§19.9: the user asked
  // for rainbow on 2026-09-20); saved preferences keep their colour map.  "viridis" is the perceptually uniform
  // option, "turbo" Google's improved rainbow, "bwr" the blue-white-red diverging map.  ``bands`` > 0 quantises
  // the map into that many discrete steps (contour-band style).
  // v0.14 (F2): the stops come from WssReportCommon.PALETTES, the single palette source shared with the wall
  // report and every export (the volume report inlines report_common.js before this script; under Node it is
  // required next to this file).  Order = the menu order.
  const PALETTE_SOURCE=(()=>{
    if(root.WssReportCommon&&root.WssReportCommon.PALETTES)return root.WssReportCommon.PALETTES;
    if(typeof module==='undefined'||typeof require!=='function'||typeof __dirname!=='string')return null;
    // Node only (tests run this file bare): evaluate the shared library into a private object so neither the
    // global scope nor the require cache changes; a test that loads report_common.js later still installs it.
    const src=require('fs').readFileSync(require('path').join(__dirname,'report_common.js'),'utf8'),scope={};
    new Function('globalThis','module',src)(scope,undefined);
    return scope.WssReportCommon&&scope.WssReportCommon.PALETTES;})();
  if(!PALETTE_SOURCE)throw new Error('WssReportCommon.PALETTES is required (report_common.js must load before volume_viewer.js)');
  const COLORMAPS={};
  for(const id of ['rainbow','viridis','turbo','bwr'])COLORMAPS[id]={label:PALETTE_SOURCE[id].label,stops:PALETTE_SOURCE[id].stops};
  let currentMap='rainbow', currentBands=0;
  function setColormap(name) {currentMap=COLORMAPS[name]?name:'rainbow';return currentMap;}
  function setBands(n) {const v=Math.floor(Number(n)||0);currentBands=v>=2?Math.min(v,64):0;return currentBands;}
  function colormapNames() {return Object.keys(COLORMAPS).map(id=>({id,label:COLORMAPS[id].label}));}
  function sampleStops(stops,t) {
    const x=clamp(t,0,1)*(stops.length-1), lo=Math.floor(x), hi=Math.min(lo+1,stops.length-1);
    return stops[lo].map((v,i)=>(v+(stops[hi][i]-v)*(x-lo))/255);
  }
  function bandedT(t,bands) {
    if(!(bands>0)) return clamp(t,0,1);
    const index=Math.min(Math.floor(clamp(t,0,1)*bands),bands-1);
    return (index+0.5)/bands;
  }
  function colormapCSS(name,bands) {
    const stops=COLORMAPS[name||currentMap].stops, n=bands===undefined?currentBands:bands;
    if(n>0) {
      const parts=[];
      for(let i=0;i<n;i++){const c=sampleStops(stops,(i+0.5)/n).map(v=>Math.round(v*255));parts.push(`rgb(${c.join(',')}) ${(100*i/n).toFixed(2)}%`,`rgb(${c.join(',')}) ${(100*(i+1)/n).toFixed(2)}%`);}
      return 'linear-gradient(90deg,'+parts.join(',')+')';
    }
    return 'linear-gradient(90deg,'+stops.map((c,i)=>`rgb(${c.join(',')}) ${Math.round(100*i/(stops.length-1))}%`).join(',')+')';
  }
  function color(value,min,max,name,bands) {
    if(!Number.isFinite(value)) return [0.63,0.68,0.72];
    const stops=COLORMAPS[name||currentMap].stops;
    const t=bandedT((value-min)/Math.max(max-min,1e-12),bands===undefined?currentBands:bands);
    return sampleStops(stops,t);
  }
  function desaturate(c,amount=0.7) {
    const grey=0.62;return c.map(v=>v*(1-amount)+grey*amount);
  }
  // ---- display scales (v0.12, 用户试用反馈 7): a scale is {min,max,log?,diverging?} in raw field units ----
  // Log (speed only): lower end = max(min, max / 200, 1e-3 m/s); diverging scales (through-plane velocity)
  // use the blue-white-red map with a symmetric range.  Only colours change; no value is altered.
  const LOG_SPAN=200, LOG_FLOOR=1e-3;
  function scaleEnds(scale) {
    const s=scale||{};let lo=Number(s.min),hi=Number(s.max);
    if(!Number.isFinite(lo))lo=0;if(!Number.isFinite(hi))hi=lo+1;
    if(s.log){lo=Math.max(lo,hi/LOG_SPAN,LOG_FLOOR);if(!(hi>lo))hi=lo*LOG_SPAN;}
    if(!(hi>lo))hi=lo+1e-6;
    return [lo,hi];
  }
  function scaleT(value,scale) {
    const v=Number(value);if(!Number.isFinite(v))return NaN;
    const [lo,hi]=scaleEnds(scale);
    if(scale&&scale.log){const a=Math.log(lo),b=Math.log(hi);return clamp((Math.log(Math.max(v,lo))-a)/(b-a),0,1);}
    return clamp((v-lo)/(hi-lo),0,1);
  }
  function scaleValueAt(t,scale) {
    const [lo,hi]=scaleEnds(scale),f=clamp(Number(t)||0,0,1);
    return scale&&scale.log?Math.exp(Math.log(lo)+f*(Math.log(hi)-Math.log(lo))):lo+f*(hi-lo);
  }
  function scaleMap(scale) {return scale&&scale.diverging?'bwr':(scale&&COLORMAPS[scale.map]?scale.map:currentMap);}
  function colorAtT(t,scale,bands) {return sampleStops(COLORMAPS[scaleMap(scale)].stops,bandedT(t,bands===undefined?currentBands:bands));}
  function scaleColor(value,scale,bands) {
    const t=scaleT(value,scale);
    return Number.isFinite(t)?colorAtT(t,scale,bands):[0.63,0.68,0.72];
  }
  function quantile(sorted,q) {
    if(!sorted.length)return NaN;const at=clamp(q,0,1)*(sorted.length-1),lo=Math.floor(at),hi=Math.ceil(at);
    return sorted[lo]+(sorted[hi]-sorted[lo])*(at-lo);
  }
  // Robust p2–p98 of the section's own samples; null when there are too few samples (caller falls back).
  function robustRange(values,options={}) {
    const sorted=Array.from(values||[]).map(Number).filter(Number.isFinite).sort((a,b)=>a-b);
    if(sorted.length<(options.minCount||8))return null;
    let lo=quantile(sorted,options.lo===undefined?0.02:options.lo),hi=quantile(sorted,options.hi===undefined?0.98:options.hi);
    if(!(hi>lo)){const pad=Math.max(Math.abs(hi),1e-6)*0.05;lo-=pad;hi+=pad;}
    return {min:lo,max:hi,count:sorted.length};
  }
  // Symmetric ±p98 |v| for signed quantities (through-plane velocity).
  function symmetricRange(values,options={}) {
    const sorted=Array.from(values||[]).map(v=>Math.abs(Number(v))).filter(Number.isFinite).sort((a,b)=>a-b);
    if(sorted.length<(options.minCount||8))return null;
    const m=quantile(sorted,options.q===undefined?0.98:options.q);
    return m>0?{min:-m,max:m,count:sorted.length,diverging:true}:null;
  }
  // Velocity component along the plane normal (positive = along the normal); NaN outside ``indices``.
  function throughPlane(velocity,indices,normal) {
    const n=unit(normal),out=new Float32Array(velocity.length/3).fill(NaN);
    for(const i of indices)out[i]=velocity[3*i]*n[0]+velocity[3*i+1]*n[1]+velocity[3*i+2]*n[2];
    return out;
  }
  function inPlane(vec,plane) {return [dot(vec,plane.u),dot(vec,plane.v)];}
  // One arrow per cell of a ⌊√max⌋² grid over the map bounds (the sample nearest the cell centre), so the count
  // never exceeds ``maxCount`` and the arrows spread evenly; ``ref`` = p95 of the in-plane speed of all samples.
  function arrowSamples(items,bounds,maxCount) {
    const k=Math.max(1,Math.floor(Math.sqrt(Math.max(1,maxCount||120)))),[xmin,xmax,ymin,ymax]=bounds,dx=(xmax-xmin)/k||1,dy=(ymax-ymin)/k||1,best=new Map();
    const mags=[];
    for(const it of items||[]) {
      const m=Math.hypot(it.du,it.dv);if(!Number.isFinite(m))continue;mags.push(m);
      const cx=Math.floor((it.x-xmin)/dx),cy=Math.floor((it.y-ymin)/dy);if(cx<0||cy<0||cx>=k||cy>=k)continue;
      const key=cx+k*cy,d=Math.hypot(it.x-(xmin+(cx+.5)*dx),it.y-(ymin+(cy+.5)*dy)),prev=best.get(key);
      if(!prev||d<prev.d)best.set(key,{...it,mag:m,d});
    }
    mags.sort((a,b)=>a-b);
    return {arrows:Array.from(best.values()),ref:mags.length?quantile(mags,0.95):0};
  }
  // Display units (contract §7): conversions live only in the display layer.
  const UNITS={pressure:{'Pa':1,'mmHg':1/133.322},velocity:{'m/s':1,'cm/s':100}};
  function convertUnit(value,field,unitName) {
    if(!Number.isFinite(value)) return NaN;
    const table=UNITS[field]||{};const factor=table[unitName];
    return factor===undefined ? value : value*factor;
  }
  function unitOptions(field) {return Object.keys(UNITS[field]||{});}
  // Anatomical frame (contract §4): rows of R are the aligned axes in world coordinates.
  function frameFromMeta(ft) {
    if(!ft||!Array.isArray(ft.rotation)||ft.rotation.length!==3||!Array.isArray(ft.origin_mm)||ft.origin_mm.length!==3) return null;
    const R=ft.rotation.map(r=>Array.isArray(r)?r.map(Number):null);
    if(R.some(r=>!r||r.length!==3||!r.every(Number.isFinite))) return null;
    const origin=ft.origin_mm.map(Number);if(!origin.every(Number.isFinite)) return null;
    return {R,origin,direction_source:ft.direction_source||null};
  }
  const dirFromAligned=(d,frame)=>[0,1,2].map(k=>d[0]*frame.R[0][k]+d[1]*frame.R[1][k]+d[2]*frame.R[2][k]);
  const dirToAligned=(d,frame)=>frame.R.map(r=>dot(r,d));
  const worldFromAligned=(a,frame)=>add(frame.origin,dirFromAligned(a,frame));
  const alignedFromWorld=(p,frame)=>dirToAligned(sub(p,frame.origin),frame);
  const STANDARD_VIEWS={
    front:{label:'前',dir:[0,-1,0],up:[0,0,1]}, back:{label:'后',dir:[0,1,0],up:[0,0,1]},
    left:{label:'左',dir:[1,0,0],up:[0,0,1]}, right:{label:'右',dir:[-1,0,0],up:[0,0,1]},
    top:{label:'上',dir:[0,0,1],up:[0,-1,0]}, bottom:{label:'下',dir:[0,0,-1],up:[0,-1,0]},
  };
  function standardCamera(name,frame,target,distance) {
    const view=STANDARD_VIEWS[name];if(!view||!frame) return null;
    const dir=unit(dirFromAligned(view.dir,frame)), up=unit(dirFromAligned(view.up,frame));
    return {position:add(target,mul(dir,distance)),target:target.slice(),up};
  }
  // §19.9 default view: the camera looks from the patient's front (aligned −y) towards the back with aligned +z
  // (towards the inlet) up.  ``dir`` is camera → target, the convention of WssReportCommon.fitView.
  function viewDirections(name,frame) {
    const view=STANDARD_VIEWS[name];if(!view||!frame) return null;
    return {dir:unit(dirFromAligned(mul(view.dir,-1),frame)),up:unit(dirFromAligned(view.up,frame))};
  }
  // Legacy reports without an anatomical frame keep the old oblique direction, now fitted to the viewport.
  const LEGACY_VIEW={dir:unit([-.85,1.25,-.55]),up:[0,0,1]};
  // A standard view framed to fill the viewport (fitView, margin 1.12); without the shared library the old
  // fixed distance (1.5 × bounding diagonal) is kept so the numerical core still works under plain Node.
  function fittedStandardCamera(name,frame,points,options={}) {
    const spec=name==='legacy'?LEGACY_VIEW:viewDirections(name,frame);if(!spec) return null;
    const fit=options.fitView||(root.WssReportCommon&&root.WssReportCommon.fitView);
    if(typeof fit==='function'&&points&&points.length) {
      try{const cam=fit(points,{dir:spec.dir,up:spec.up,fov:options.fov||42,aspect:options.aspect||1,margin:options.margin||1.12});
          // keep the declared up (fitView returns it orthogonalised): OrbitControls spins about camera.up, and the
          // legacy oblique view should still spin about world z; lookAt handles a non-orthogonal up.
          if(cam&&cam.position&&cam.target)return {position:cam.position.slice(),target:cam.target.slice(),up:spec.up.slice()};}catch(_){}
    }
    const target=(options.center||[0,0,0]).slice(),distance=options.distance||1;
    return {position:add(target,mul(spec.dir,-distance)),target,up:spec.up.slice()};
  }
  // Numbers on screen never use scientific notation (§19.1 formatValue; the local copy keeps plain-Node tests
  // and a report opened without the shared library identical).
  function formatNumber(v,opts) {
    const c=root.WssReportCommon;
    if(c&&typeof c.formatValue==='function'){try{return c.formatValue(v,opts);}catch(_){}}
    opts=opts||{};const digits=opts.digits||3,maxDec=opts.maxDecimals===undefined?4:opts.maxDecimals;
    if(v===null||v===undefined||v===''||!Number.isFinite(Number(v)))return opts.missing===undefined?'—':opts.missing;
    v=Number(v);const a=Math.abs(v);
    if(a===0)return '0';
    if(a>=Math.pow(10,digits))return String(Math.round(v));
    if(a<Math.pow(10,-maxDec))return '0';
    return v.toFixed(Math.max(0,Math.min(maxDec,digits-1-Math.floor(Math.log10(a)))));
  }
  // Human wording of the prediction frame (§19.9): "收缩期峰值帧（约 0.21 s）"; the raw label / step live in 技术信息.
  const FRAME_WORDS={peak_systole:['收缩期峰值帧','Peak-systolic frame'],end_diastole:['舒张末期帧','End-diastolic frame'],cycle_mean:['周期平均','Cycle average']};
  function frameText(modelFrame,language) {
    const mf=modelFrame&&typeof modelFrame==='object'?modelFrame:{},en=language==='en';
    const key=String(mf.label||mf.target||''),words=FRAME_WORDS[key],t=Number(mf.time_s);
    const name=words?words[en?1:0]:(en?'Fixed prediction frame':'固定预测时相');
    const secs=String(+t.toFixed(3));
    return Number.isFinite(t)&&mf.time_s!==null&&mf.time_s!==''?(en?`${name} (≈ ${secs} s)`:`${name}（约 ${secs} s）`):name;
  }
  // Short release name, the wall report's rule (W4, §19.9): "PF6_VF6_peak_3seed_20260920" → "PF6_VF6_peak";
  // the full id stays in 技术信息 and the tooltip.
  function releaseShort(id) {
    const text=String(id||'').trim();
    return text.replace(/_\d+seed_\d{6,8}$/,'')||'—';
  }
  // ISO time with an offset when possible: a naive "2026-09-20 02:15:34" (written in the server process' zone)
  // borrows the offset of an ISO stamp written by the same run (the outlet confirmation), else stays naive.
  function withOffset(naive,hintIso) {
    const s=String(naive||'').trim();if(!s) return '';
    if(/[zZ]$|[+-]\d{2}:?\d{2}$/.test(s)) return s;
    const m=String(hintIso||'').match(/([+-]\d{2}:\d{2}|Z)$/);
    const iso=s.replace(' ','T');
    return m?iso+m[1]:iso;
  }
  // B4 banner rows: geometry checks outside the release's declared reference range and a non-good ensemble quality.
  const REF_FIELDS={length_mm:'长度',radius_min_mm:'最小半径',radius_median_mm:'中位半径',radius_max_mm:'最大半径',spacing_mm:'点间距',surface_variation_median:'表面变化度',variation:'表面变化度'};
  function referenceLabel(path) {
    const p=String(path||'');
    if(p.startsWith('cloud.'))return '点云'+(REF_FIELDS[p.slice(6)]||p.slice(6));
    if(p.startsWith('geometry.')){const rest=p.slice(9),at=rest.lastIndexOf('.');if(at>0){const f=rest.slice(at+1);return rest.slice(0,at)+(REF_FIELDS[f]||f);}}
    return p;
  }
  function warningItems(meta) {
    const out=[];const m=meta&&typeof meta==='object'?meta:{};
    const ra=m.reference_assessment&&typeof m.reference_assessment==='object'?m.reference_assessment:null;
    for(const c of (ra&&Array.isArray(ra.checks)?ra.checks:[])) {
      if(!c||c.status!=='review')continue;
      const units=c.units&&c.units!=='1'?' '+c.units:'',what=`${referenceLabel(c.path)} ${formatNumber(c.value)}${units}`,range=`${formatNumber(c.min)}–${formatNumber(c.max)}${units}`;
      out.push({kind:'reference',text:`${what}（参考 ${range}）`,what,range});
    }
    const pop=ra&&ra.population&&typeof ra.population==='object'?ra.population:null;
    if(pop&&pop.status==='review')out.push({kind:'population',text:'人群参照需要复核'});
    const q=m.quality&&typeof m.quality==='object'?m.quality:null;
    if(q&&q.level&&q.level!=='good')out.push({kind:'quality',text:`模型集成质量：${q.label||q.level}`});
    return out;
  }
  // Same sentence frame as the wall report's banner (W4): 「注意：…。预测可信度可能下降，请结合详情复核。」; the first
  // out-of-range measurement is quoted so the doctor sees what is off without opening the menu.
  function warningText(items) {
    const ref=items.filter(x=>x.kind==='reference'),rest=items.filter(x=>x.kind!=='reference'),parts=[];
    if(ref.length)parts.push(`输入几何有 ${ref.length} 项超出发布包参考范围（${ref[0].what}，参考 ${ref[0].range}${ref.length>1?' 等':''}）`);
    for(const x of rest)parts.push(x.text);
    return parts.length?'注意：'+parts.join('；')+'。预测可信度可能下降，请结合详情复核。':'';
  }
  function cameraToAligned(cam,frame) {
    return {position:alignedFromWorld(cam.position,frame),target:alignedFromWorld(cam.target,frame),up:dirToAligned(cam.up,frame)};
  }
  function cameraFromAligned(cam,frame) {
    return {position:worldFromAligned(cam.position,frame),target:worldFromAligned(cam.target,frame),up:dirFromAligned(cam.up,frame)};
  }
  // View state <-> URL-safe text (contract §5).
  function toBase64Url(text) {
    const bytes=new TextEncoder().encode(text);let binary='';for(const b of bytes)binary+=String.fromCharCode(b);
    return btoa(binary).replace(/\+/g,'-').replace(/\//g,'_').replace(/=+$/,'');
  }
  function fromBase64Url(text) {
    const padded=String(text).replace(/-/g,'+').replace(/_/g,'/')+'='.repeat((4-String(text).length%4)%4);
    const binary=atob(padded), bytes=new Uint8Array(binary.length);for(let i=0;i<binary.length;i++)bytes[i]=binary.charCodeAt(i);
    return new TextDecoder().decode(bytes);
  }
  function encodeView(state) {return toBase64Url(JSON.stringify(state));}
  function decodeView(text) {
    const state=JSON.parse(fromBase64Url(text));
    if(!state||typeof state!=='object'||state.schema_version!=='wss-deploy.view/v1') throw new Error('不是有效的视图状态');
    return state;
  }
  function nearestBin(sArray,s) {
    let best=-1,bestD=Infinity;
    for(let i=0;i<sArray.length;i++){const v=Number(sArray[i]);if(!Number.isFinite(v))continue;const d=Math.abs(v-s);if(d<bestD){bestD=d;best=i;}}
    return best;
  }
  function fractionForArc(group,s) {
    const length=group&&group.arc.length?group.arc[group.arc.length-1]:0;
    return length>0 ? clamp(s/length,0,1) : 0;
  }
  // Arc length of each point along its own branch (nearest sample in that branch).
  function arcAlongBranch(groups,pts,segments,indices) {
    const out=new Float32Array(pts.length/3).fill(NaN), byId=new Map(groups.map(g=>[Number(g.segment),g]));
    for(const i of indices) {
      const g=segments?byId.get(Number(segments[i])):groups[0];if(!g||!g.points.length)continue;
      const p=point(pts,i);let best=0,bestD=Infinity;
      for(let j=0;j<g.points.length;j++){const d=norm(sub(g.points[j],p));if(d<bestD){bestD=d;best=j;}}
      out[i]=g.arc[best];
    }
    return out;
  }
  function regionIndices(indices,segments,arc,segment,sLo,sHi) {
    const id=segment===null||segment===undefined||segment===''?null:Number(segment);
    return indices.filter(i=>(id===null||(segments&&Number(segments[i])===id))&&Number.isFinite(arc[i])&&arc[i]>=sLo&&arc[i]<=sHi);
  }
  function regionPressureDrop(indices,arc,pressure,sLo,sHi,fraction=0.1) {
    const span=Math.max(sHi-sLo,1e-9), width=span*fraction;
    const mean=list=>list.length?list.reduce((s,i)=>s+pressure[i],0)/list.length:null;
    const proximal=mean(indices.filter(i=>arc[i]<=sLo+width)), distal=mean(indices.filter(i=>arc[i]>=sHi-width));
    return {proximal,distal,drop:proximal!==null&&distal!==null?proximal-distal:null,
      n_proximal:indices.filter(i=>arc[i]<=sLo+width).length,n_distal:indices.filter(i=>arc[i]>=sHi-width).length};
  }
  function sphereIndices(pts,indices,center,radius) {
    return indices.filter(i=>norm(sub(point(pts,i),center))<=radius);
  }
  function nearestPoint(pts,indices,p) {
    let best=-1,bestD=Infinity;
    for(const i of indices){const d=norm(sub(point(pts,i),p));if(d<bestD){bestD=d;best=i;}}
    return {index:best,distance:bestD};
  }
  function probeRecord(i,data) {
    const rec={index:i,position:point(data.pts,i)};
    if(data.velocity){rec.velocity=point(data.velocity,i);rec.speed=norm(rec.velocity);}
    if(data.pressure&&Number.isFinite(data.pressure[i]))rec.pressure=data.pressure[i];
    if(data.segments)rec.segment=Number(data.segments[i]);
    for(const [key,arr] of [['s',data.s],['radius',data.radius],['distWall',data.distWall],['trust',data.trust]]) if(arr&&Number.isFinite(arr[i]))rec[key]=arr[i];
    return rec;
  }
  const TRUST_LABELS={1:'插值无支撑',2:'表面粗糙',4:'几何越界',8:'采样支撑弱',16:'近切口'};
  function trustLabels(trustMeta) {
    const out=[];const bits=trustMeta&&trustMeta.bits?trustMeta.bits:TRUST_LABELS;
    for(const key of Object.keys(bits).sort((a,b)=>Number(a)-Number(b))) {
      const bit=Number(key);const source=(trustMeta&&Array.isArray(trustMeta.sources)?trustMeta.sources:[]).find(s=>Number(s.bit)===bit);
      const id=typeof bits[key]==='string'?bits[key]:'';
      const fraction=trustMeta&&trustMeta.fractions?trustMeta.fractions[id]:undefined;
      out.push({bit,label:(source&&source.label)||TRUST_LABELS[bit]||id||`位 ${bit}`,rule:source&&source.rule||'',fraction:Number.isFinite(Number(fraction))?Number(fraction):null});
    }
    return out;
  }
  const FINDING_KINDS={high_wss_cluster:'高 WSS 区',low_wss_cluster:'低 WSS 区',max_wss:'WSS 最大值',max_diameter:'最大直径',min_radius:'最小半径',
    max_speed:'速度最大值',min_pressure:'压力最低点',pressure_drop:'分支压降',low_speed_region:'低速区'};
  const SEVERITY_LABELS={attention:'关注',note:'提示',info:'几何'};
  function findingsSorted(items) {
    const order={attention:0,note:1,info:2};
    return (Array.isArray(items)?items:[]).filter(x=>x&&typeof x==='object').slice().sort((a,b)=>(order[a.severity]??3)-(order[b.severity]??3)||(a.rank??99)-(b.rank??99));
  }
  // Conservative local interpolation for a finite-thickness slice.  The
  // convex-hull/radius/angular checks are an explicit extrapolation mask:
  // unsupported regions remain NaN instead of becoming invented pressure.
  function convexHull(points) {
    const sorted=points.map(p=>[p[0],p[1]]).filter(p=>Number.isFinite(p[0])&&Number.isFinite(p[1]))
      .sort((a,b)=>a[0]-b[0]||a[1]-b[1]),unique=[];
    for(const p of sorted){const last=unique[unique.length-1];if(!last||Math.hypot(p[0]-last[0],p[1]-last[1])>1e-10)unique.push(p);}
    if(unique.length<=1)return unique;
    const turn=(o,a,b)=>(a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0]);
    const lower=[];for(const p of unique){while(lower.length>=2&&turn(lower[lower.length-2],lower[lower.length-1],p)<=1e-12)lower.pop();lower.push(p);}
    const upper=[];for(let i=unique.length-1;i>=0;i--){const p=unique[i];while(upper.length>=2&&turn(upper[upper.length-2],upper[upper.length-1],p)<=1e-12)upper.pop();upper.push(p);}
    lower.pop();upper.pop();return lower.concat(upper);
  }
  function inConvexHull(p,hull) {
    if(hull.length<3)return false;let sign=0;
    for(let i=0;i<hull.length;i++){const a=hull[i],b=hull[(i+1)%hull.length],cross=(b[0]-a[0])*(p[1]-a[1])-(b[1]-a[1])*(p[0]-a[0]);if(Math.abs(cross)<=1e-9)continue;const current=Math.sign(cross);if(!sign)sign=current;else if(current!==sign)return false;}
    return Boolean(sign);
  }
  function interpolateIDW(points,values,options={}) {
    if(!Array.isArray(points)||points.length!==values.length)throw new Error('插值点和值的长度必须一致');
    const support=[];for(let i=0;i<points.length;i++){const p=points[i],v=Number(values[i]);if(p&&Number.isFinite(p[0])&&Number.isFinite(p[1])&&Number.isFinite(v))support.push({p:[p[0],p[1]],v});}
    const hull=convexHull(support.map(s=>s.p));
    const bounds=options.bounds||(()=>{if(!support.length)return [0,1,0,1];const xs=support.map(s=>s.p[0]),ys=support.map(s=>s.p[1]);return [Math.min(...xs),Math.max(...xs),Math.min(...ys),Math.max(...ys)];})();
    let [xmin,xmax,ymin,ymax]=bounds.map(Number);if(![xmin,xmax,ymin,ymax].every(Number.isFinite))throw new Error('插值边界必须有限');
    if(xmax<=xmin)xmax=xmin+1e-6;if(ymax<=ymin)ymax=ymin+1e-6;
    const nx=Math.max(2,Math.floor(options.nx||96)),ny=Math.max(2,Math.floor(options.ny||96));
    const minNeighbors=Math.max(3,Math.floor(options.minNeighbors||4)),maxNeighbors=Math.max(minNeighbors,Math.floor(options.maxNeighbors||12));
    const power=Math.max(0.1,Number(options.power||2)),givenRadius=Number(options.maxDistance),radius=Number.isFinite(givenRadius)&&givenRadius>0?givenRadius:Infinity;
    const valuesOut=new Array(nx*ny).fill(NaN),mask=new Uint8Array(nx*ny);let validCount=0;
    for(let iy=0;iy<ny;iy++)for(let ix=0;ix<nx;ix++){
      const q=[xmin+(ix+.5)*(xmax-xmin)/nx,ymin+(iy+.5)*(ymax-ymin)/ny],at=iy*nx+ix;if(!inConvexHull(q,hull))continue;
      const nearest=support.map(s=>({s,d:Math.hypot(s.p[0]-q[0],s.p[1]-q[1])})).sort((a,b)=>a.d-b.d),nearby=nearest.filter(x=>x.d<=radius).slice(0,maxNeighbors);
      if(nearby.length<minNeighbors||nearby[nearby.length-1].d>radius)continue;
      const angles=nearby.map(x=>Math.atan2(x.s.p[1]-q[1],x.s.p[0]-q[0])).sort((a,b)=>a-b);let maxGap=0;
      for(let i=0;i<angles.length;i++)maxGap=Math.max(maxGap,(angles[(i+1)%angles.length]||angles[0]+2*Math.PI)-angles[i]);
      if(maxGap>Number(options.maxAngularGap||Math.PI*1.28))continue;
      const exact=nearby.find(x=>x.d<1e-10);if(exact)valuesOut[at]=exact.s.v;else{let numerator=0,denominator=0;for(const x of nearby){const w=1/Math.pow(x.d,power);numerator+=w*x.s.v;denominator+=w;}valuesOut[at]=numerator/denominator;}
      mask[at]=1;validCount++;
    }
    return {values:valuesOut,mask:Array.from(mask),nx,ny,bounds:[xmin,xmax,ymin,ymax],validCount,supportCount:support.length};
  }
  // ---- Complete cross-section (2026-09-21): the lumen outline is the plane ∩ wall-mesh contour and the wall
  // itself is a boundary condition (no-slip velocity = 0; interpolated wall pressure), so the 2-D map can be
  // filled everywhere inside the vessel instead of only where interior samples happen to be dense.
  function planeContour(vertices,faces,plane,wallValues,maxRadius) {
    const n=unit(plane.normal),o=plane.origin,u=plane.u,v=plane.v,nv=Math.floor(vertices.length/3),d=new Float32Array(nv),segs=[];
    for(let i=0;i<nv;i++)d[i]=(vertices[3*i]-o[0])*n[0]+(vertices[3*i+1]-o[1])*n[1]+(vertices[3*i+2]-o[2])*n[2];
    const crossing=(a,b)=>{
      const da=d[a],db=d[b];if((da>=0)===(db>=0))return null;const t=da/(da-db);const key=a<b?a*nv+b:b*nv+a;
      const q=[vertices[3*a]+(vertices[3*b]-vertices[3*a])*t-o[0],vertices[3*a+1]+(vertices[3*b+1]-vertices[3*a+1])*t-o[1],vertices[3*a+2]+(vertices[3*b+2]-vertices[3*a+2])*t-o[2]];
      let val=NaN;if(wallValues){const wa=wallValues[a],wb=wallValues[b];if(Number.isFinite(wa)&&Number.isFinite(wb))val=wa+(wb-wa)*t;}
      return {x:dot(q,u),y:dot(q,v),val,key};
    };
    const r2=Number.isFinite(maxRadius)&&maxRadius>0?maxRadius*maxRadius:Infinity;
    for(let t=0;t+2<faces.length;t+=3){
      const a=faces[t],b=faces[t+1],c=faces[t+2],p=[crossing(a,b),crossing(b,c),crossing(c,a)].filter(Boolean);
      if(p.length!==2)continue;const mx=(p[0].x+p[1].x)/2,my=(p[0].y+p[1].y)/2;if(mx*mx+my*my>r2)continue;
      segs.push([p[0].x,p[0].y,p[1].x,p[1].y,(p[0].val+p[1].val)/2,p[0].key,p[1].key]);
    }
    return segs;
  }
  // Chain contour segments through their shared mesh edges into loops (open chains where the mesh has openings).
  function contourLoops(segs) {
    const byKey=new Map();segs.forEach((sg,i)=>{for(const k of [sg[5],sg[6]]){let a=byKey.get(k);if(!a){a=[];byKey.set(k,a);}a.push(i);}});
    const used=new Uint8Array(segs.length),loops=[];
    const walk=(from,exitKey,stopKey,out)=>{
      let seg=from,key=exitKey;
      for(let guard=0;guard<=segs.length;guard++){
        if(key===stopKey&&seg!==from)return true;
        const next=(byKey.get(key)||[]).find(j=>!used[j]);if(next===undefined)return key===stopKey&&seg!==from;
        used[next]=1;out.push(next);seg=next;key=segs[next][5]===key?segs[next][6]:segs[next][5];
      }
      return false;
    };
    for(let s0=0;s0<segs.length;s0++){
      if(used[s0])continue;used[s0]=1;const members=[s0],A=segs[s0][5],B=segs[s0][6];
      const closed=walk(s0,B,A,members);if(!closed)walk(s0,A,B,members);
      loops.push({segments:members,closed});
    }
    return loops;
  }
  // The local cross-section. Ranking: a closed loop that encloses the origin (innermost when nested) >
  // an open chain wrapping more than a half turn around the origin (largest wrap, then nearest) > the nearest
  // chain. Open chains never count as "enclosing" (a stray arc crossed once by the test ray is not a lumen).
  function selectLoop(loops,segs,origin,maxDist) {
    const px=origin?origin[0]:0,py=origin?origin[1]:0,limit=Number.isFinite(maxDist)&&maxDist>0?maxDist:Infinity;let best=null;
    const rank=c=>c.inside?3:c.wraps?2:1;
    for(const loop of loops){
      if(loop.segments.length<3&&loops.length>1)continue;
      const ss=loop.segments.map(i=>segs[i]);let crossings=0,dmin=Infinity,minx=Infinity,maxx=-Infinity,miny=Infinity,maxy=-Infinity;
      for(const sg of ss){const [x0,y0,x1,y1]=sg;if((y0<=py)!==(y1<=py)){const xi=x0+(x1-x0)*(py-y0)/(y1-y0);if(xi>px)crossings++;}
        dmin=Math.min(dmin,Math.hypot(x0-px,y0-py),Math.hypot(x1-px,y1-py));minx=Math.min(minx,x0,x1);maxx=Math.max(maxx,x0,x1);miny=Math.min(miny,y0,y1);maxy=Math.max(maxy,y0,y1);}
      let span=2*Math.PI;
      if(!loop.closed){const angs=ss.map(sg=>Math.atan2((sg[1]+sg[3])/2-py,(sg[0]+sg[2])/2-px)).sort((a,b)=>a-b);let gap=0;for(let i=0;i<angs.length;i++){const next=i+1<angs.length?angs[i+1]:angs[0]+2*Math.PI;gap=Math.max(gap,next-angs[i]);}span=2*Math.PI-gap;}
      const inside=loop.closed&&crossings%2===1;
      if(!inside&&dmin>limit)continue;   // the distance limit only prunes chains that do not enclose the origin
      const cand={segs:ss,inside,dmin,area:(maxx-minx)*(maxy-miny),closed:loop.closed,bbox:[minx,maxx,miny,maxy],span,wraps:!loop.closed&&span>Math.PI};
      const better=!best||rank(cand)>rank(best)||(rank(cand)===rank(best)&&(cand.inside?cand.area<best.area:cand.wraps?(cand.span>best.span+1e-6||(Math.abs(cand.span-best.span)<=1e-6&&cand.dmin<best.dmin)):cand.dmin<best.dmin));
      if(better)best=cand;
    }
    return best;
  }
  // An open chain (the plane passes through a vessel opening or a mesh hole) is closed with one synthetic
  // segment between its two free ends when the gap is small next to the chain length; the synthetic segment
  // carries no wall value (key −1) so it never acts as a wall boundary condition.
  function closeChain(ss) {
    const count=new Map();for(const sg of ss)for(const k of [sg[5],sg[6]])count.set(k,(count.get(k)||0)+1);
    const ends=[];for(const sg of ss){if(count.get(sg[5])===1)ends.push([sg[0],sg[1]]);if(count.get(sg[6])===1)ends.push([sg[2],sg[3]]);}
    if(ends.length!==2)return null;
    let length=0;for(const sg of ss)length+=Math.hypot(sg[2]-sg[0],sg[3]-sg[1]);
    const gap=Math.hypot(ends[1][0]-ends[0][0],ends[1][1]-ends[0][1]);
    if(!(gap<=Math.max(0.6*length,1e-6)))return null;
    return ss.concat([[ends[0][0],ends[0][1],ends[1][0],ends[1][1],NaN,-1,-1]]);
  }
  function pointInLoop(segs,x,y) {let inside=false;for(const sg of segs){const [x0,y0,x1,y1]=sg;if((y0<=y)!==(y1<=y)){const xi=x0+(x1-x0)*(y-y0)/(y1-y0);if(xi>x)inside=!inside;}}return inside;}
  function scaleBarLength(mmPerPx,targetPx) {const want=(targetPx||90)*mmPerPx;let best=1;for(const L of [0.5,1,2,5,10,20,50,100,200])if(Math.abs(L-want)<Math.abs(best-want))best=L;return best;}
  // Even-odd scanline test of grid-cell centres against the contour segment soup (loops need not be assembled).
  function scanlineInside(segs,bounds,nx,ny) {
    const [xmin,xmax,ymin,ymax]=bounds,mask=new Uint8Array(nx*ny),dx=(xmax-xmin)/nx,dy=(ymax-ymin)/ny;
    const rows=Array.from({length:ny},()=>[]);
    for(const sg of segs){const lo=Math.min(sg[1],sg[3]),hi=Math.max(sg[1],sg[3]);const r0=Math.max(0,Math.floor((lo-ymin)/dy)),r1=Math.min(ny-1,Math.floor((hi-ymin)/dy));for(let r=r0;r<=r1;r++)rows[r].push(sg);}
    for(let iy=0;iy<ny;iy++){
      const yc=ymin+(iy+.5)*dy,xs=[];
      for(const sg of rows[iy]){const y0=sg[1],y1=sg[3];if((y0<=yc)===(y1<=yc))continue;xs.push(sg[0]+(sg[2]-sg[0])*(yc-y0)/(y1-y0));}
      if(xs.length<2)continue;xs.sort((a,b)=>a-b);
      for(let k=0;k+1<xs.length;k+=2){const i0=Math.max(0,Math.ceil((xs[k]-xmin)/dx-.5)),i1=Math.min(nx-1,Math.floor((xs[k+1]-xmin)/dx-.5));for(let ix=i0;ix<=i1;ix++)mask[iy*nx+ix]=1;}
    }
    return mask;
  }
  // IDW over interior samples plus boundary samples, evaluated on every cell the contour encloses.
  // ``low`` marks cells farther than ``directRadius`` from any interior sample (filled from the wall condition).
  function fillSection(points,values,boundary,bounds,nx,ny,inside,options={}) {
    const support=[];
    for(let i=0;i<points.length;i++){const v=Number(values[i]);if(points[i]&&Number.isFinite(points[i][0])&&Number.isFinite(points[i][1])&&Number.isFinite(v))support.push({x:points[i][0],y:points[i][1],v,wall:false});}
    const direct=support.length;
    for(const b of (boundary||[])){if(b&&Number.isFinite(b[0])&&Number.isFinite(b[1])&&Number.isFinite(b[2]))support.push({x:b[0],y:b[1],v:b[2],wall:true});}
    const [xmin,xmax,ymin,ymax]=bounds,dx=(xmax-xmin)/nx,dy=(ymax-ymin)/ny,out=new Array(nx*ny).fill(NaN),low=new Uint8Array(nx*ny);
    if(!support.length)return {values:out,low,filled:0,direct:0,directCells:0};
    const k=Math.max(3,Math.floor(options.neighbors||8)),directRadius=Number(options.directRadius)>0?Number(options.directRadius):Infinity;
    const h=Math.max(Number(options.cell)||0,Math.max(dx,dy)),gx=Math.max(1,Math.ceil((xmax-xmin)/h)),gy=Math.max(1,Math.ceil((ymax-ymin)/h)),grid=new Map();
    const cellOf=(x,y)=>[Math.max(0,Math.min(gx-1,Math.floor((x-xmin)/h))),Math.max(0,Math.min(gy-1,Math.floor((y-ymin)/h)))];
    support.forEach((sp,i)=>{const [cx,cy]=cellOf(sp.x,sp.y),key=cx+gx*cy;let a=grid.get(key);if(!a){a=[];grid.set(key,a);}a.push(i);});
    const maxRing=Math.max(gx,gy);let filled=0,directCells=0;
    for(let iy=0;iy<ny;iy++)for(let ix=0;ix<nx;ix++){
      const at=iy*nx+ix;if(!inside[at])continue;
      const qx=xmin+(ix+.5)*dx,qy=ymin+(iy+.5)*dy,[cx,cy]=cellOf(qx,qy),found=[];
      for(let r=0;r<=maxRing;r++){
        for(let gxi=cx-r;gxi<=cx+r;gxi++){if(gxi<0||gxi>=gx)continue;for(let gyi=cy-r;gyi<=cy+r;gyi++){if(gyi<0||gyi>=gy)continue;if(Math.max(Math.abs(gxi-cx),Math.abs(gyi-cy))!==r)continue;const a=grid.get(gxi+gx*gyi);if(!a)continue;for(const i of a){const sp=support[i];found.push({sp,d:Math.hypot(sp.x-qx,sp.y-qy)});}}}
        if(found.length>=k){found.sort((a,b)=>a.d-b.d);if(found[k-1].d<=r*h)break;}
      }
      if(!found.length)continue;
      found.sort((a,b)=>a.d-b.d);const near=found.slice(0,k);
      let num=0,den=0;for(const x of near){const w=1/(x.d*x.d+1e-9);num+=w*x.sp.v;den+=w;}
      out[at]=num/den;filled++;
      const nearestDirect=found.find(x=>!x.sp.wall);
      if(nearestDirect&&nearestDirect.d<=directRadius)directCells++;else low[at]=1;
    }
    return {values:out,low,filled,direct,directCells};
  }
  // ---- §15 slice CSV + station series (pure; exported for Node tests) ----
  // Every grid cell the contour mask keeps and that carries a finite value, as
  // [x_mm, y_mm, value (raw field units), units, filled_from_wall].  Coordinates are cell centres
  // in the plane's own (u, v) frame, i.e. the same frame the zoom map draws.
  function sliceGridToRows(grid,bounds,units) {
    const rows=[];
    if(!grid||!Array.isArray(bounds)||bounds.length!==4)return rows;
    const nx=Math.floor(Number(grid.nx)||0),ny=Math.floor(Number(grid.ny)||0);
    if(!(nx>0&&ny>0))return rows;
    const [xmin,xmax,ymin,ymax]=bounds.map(Number);
    if(![xmin,xmax,ymin,ymax].every(Number.isFinite)||xmax<=xmin||ymax<=ymin)return rows;
    const dx=(xmax-xmin)/nx,dy=(ymax-ymin)/ny,values=grid.values||[],mask=grid.mask||null,low=grid.low||null;
    const unit=units===undefined||units===null?'':String(units);
    for(let iy=0;iy<ny;iy++)for(let ix=0;ix<nx;ix++) {
      const at=iy*nx+ix;
      if(mask&&!mask[at])continue;
      const v=Number(values[at]);if(!Number.isFinite(v))continue;
      rows.push([xmin+(ix+0.5)*dx,ymin+(iy+0.5)*dy,v,unit,low&&low[at]?1:0]);
    }
    return rows;
  }
  // Station fractions for the slice series (§15.10): 6 reproduces the automatic 5/20/40/60/80/95 %,
  // any other count is spread evenly between 5 % and 95 %.
  function seriesFractions(count) {
    const n=Math.max(1,Math.min(24,Math.floor(Number(count))||6));
    if(n===6)return [0.05,0.2,0.4,0.6,0.8,0.95];
    if(n===1)return [0.5];
    const out=[];for(let i=0;i<n;i++)out.push(Number((0.05+i*0.9/(n-1)).toFixed(6)));
    return out;
  }
  // ---- C10 / C11 / C12 / C16 / C17 helpers (pure; exported for Node tests) ----
  // Nearest-source label per query point through a uniform hash grid (vertex -> branch for hiding).
  function nearestLabels(srcPts,srcLabels,queries) {
    const n=Math.floor(srcPts.length/3),m=Math.floor(queries.length/3),out=new Int32Array(m).fill(-1);
    if(!n||!m)return out;
    const mn=[Infinity,Infinity,Infinity],mx=[-Infinity,-Infinity,-Infinity];
    for(let i=0;i<n;i++)for(let k=0;k<3;k++){const v=srcPts[3*i+k];if(v<mn[k])mn[k]=v;if(v>mx[k])mx[k]=v;}
    const ext=[0,1,2].map(k=>Math.max(mx[k]-mn[k],1e-6));
    const cell=Math.max(Math.cbrt(ext[0]*ext[1]*ext[2]/n)*2,1e-3);
    const dims=[0,1,2].map(k=>Math.min(1024,Math.floor(ext[k]/cell)+1));
    const cellOf=(x,k)=>Math.max(0,Math.min(dims[k]-1,Math.floor((x-mn[k])/cell)));
    const key=(a,b,c)=>a+dims[0]*(b+dims[1]*c);
    const grid=new Map();
    for(let i=0;i<n;i++){const k=key(cellOf(srcPts[3*i],0),cellOf(srcPts[3*i+1],1),cellOf(srcPts[3*i+2],2));let a=grid.get(k);if(!a){a=[];grid.set(k,a);}a.push(i);}
    const maxRing=Math.max(dims[0],dims[1],dims[2]);
    for(let q=0;q<m;q++){
      const x=queries[3*q],y=queries[3*q+1],z=queries[3*q+2],cx=cellOf(x,0),cy=cellOf(y,1),cz=cellOf(z,2);let best=-1,bd=Infinity;
      for(let r=0;r<=maxRing;r++){
        for(let ix=cx-r;ix<=cx+r;ix++){if(ix<0||ix>=dims[0])continue;for(let iy=cy-r;iy<=cy+r;iy++){if(iy<0||iy>=dims[1])continue;for(let iz=cz-r;iz<=cz+r;iz++){if(iz<0||iz>=dims[2])continue;if(Math.max(Math.abs(ix-cx),Math.abs(iy-cy),Math.abs(iz-cz))!==r)continue;const a=grid.get(key(ix,iy,iz));if(!a)continue;for(const i of a){const dx=srcPts[3*i]-x,dy=srcPts[3*i+1]-y,dz=srcPts[3*i+2]-z,d=dx*dx+dy*dy+dz*dz;if(d<bd){bd=d;best=i;}}}}}
        if(best>=0&&Math.sqrt(bd)<=r*cell)break;
      }
      out[q]=best<0?-1:Number(srcLabels[best]);
    }
    return out;
  }
  // A triangle disappears only when all three vertices belong to hidden branches (C10).
  function filterFaces(faces,vertexLabels,hidden) {
    if(!hidden||!hidden.size||!vertexLabels)return faces;
    const out=[];
    for(let t=0;t<faces.length;t+=3){const a=faces[t],b=faces[t+1],c=faces[t+2];if(hidden.has(vertexLabels[a])&&hidden.has(vertexLabels[b])&&hidden.has(vertexLabels[c]))continue;out.push(a,b,c);}
    return new Uint32Array(out);
  }
  // Probe log serialisation (C11): raw prediction units (m/s, Pa, mm); replaced by WssReportCommon in phase 2 when present.
  const PROBE_COLUMNS=[['id','编号','ID'],['x','x_mm','x_mm'],['y','y_mm','y_mm'],['z','z_mm','z_mm'],['branch','分支','Branch'],['s_from_root_mm','弧长_mm','Arc_mm'],['radius_mm','半径_mm','Radius_mm'],
    ['speed_m_s','速度_m_s','Speed_m_s'],['u','u_m_s','u_m_s'],['v','v_m_s','v_m_s'],['w','w_m_s','w_m_s'],['pressure_pa','压力_Pa','Pressure_Pa'],['wall_pressure_pa','壁面压力_Pa','WallPressure_Pa'],['dist_to_wall_mm','到壁_mm','DistWall_mm'],['created_at','时间','Time']];
  function probeCell(row,key) {
    const numText=(v,digits)=>Number.isFinite(Number(v))?String(+Number(v).toFixed(digits)):'';
    if(key==='x'||key==='y'||key==='z'){const p=Array.isArray(row.xyz_mm)?row.xyz_mm:[];return numText(p[{x:0,y:1,z:2}[key]],3);}
    const values=row.values&&typeof row.values==='object'?row.values:{};
    if(Object.prototype.hasOwnProperty.call(values,key))return numText(values[key],5);
    const v=row[key];if(v===undefined||v===null)return '';
    return typeof v==='number'?numText(v,3):String(v);
  }
  function probeToTSV(rows,lang) {
    const col=lang==='en'?2:1;
    return [PROBE_COLUMNS.map(c=>c[col]).join('\t')].concat((Array.isArray(rows)?rows:[]).map(r=>PROBE_COLUMNS.map(c=>probeCell(r,c[0]).replace(/[\t\r\n]+/g,' ')).join('\t'))).join('\n');
  }
  function probeToCSV(rows,lang) {
    const col=lang==='en'?2:1,esc=s=>/[",\r\n]/.test(s)?'"'+s.replace(/"/g,'""')+'"':s;
    return '\ufeff'+[PROBE_COLUMNS.map(c=>c[col]).join(',')].concat((Array.isArray(rows)?rows:[]).map(r=>PROBE_COLUMNS.map(c=>esc(probeCell(r,c[0]))).join(','))).join('\r\n')+'\r\n';
  }
  // English labels (C12).  Minimal local dictionary; WssReportCommon.englishLabel takes over in phase 2.
  const LABELS_EN={'速度':'Speed','速度大小':'Speed','压力':'Pressure','相对压力':'Relative pressure','主动脉':'Aorta','左髂总':'Left CIA','右髂总':'Right CIA','左髂外':'Left EIA','右髂外':'Right EIA','左髂内':'Left IIA','右髂内':'Right IIA',
    '前':'Front','后':'Back','左':'Left','右':'Right','上':'Top','下':'Bottom','探针':'Probe','发现':'Finding','连续色标':'continuous','段离散色带':'bands','壁面压力':'Wall pressure','流线':'Streamlines','截面':'Slice',
    '半径':'Radius','直径':'Diameter','最大直径':'Max diameter','等效直径':'Equivalent diameter','结论（参考）':'Summary (for reference)'};
  function labelText(text,lang) {
    if(lang!=='en')return text;
    const common=root.WssReportCommon;
    if(common&&typeof common.englishLabel==='function'){try{const v=common.englishLabel(text,lang);if(typeof v==='string'&&v&&v!==text)return v;}catch(_){}}
    return LABELS_EN[text]||text;
  }
  const fmtTick=v=>formatNumber(v);
  // Vector colour bar (C12).  Local fallback; WssReportCommon.colorbarSVG is preferred when present.
  function colorbarSVGLocal(o) {
    o=o||{};const width=o.width||96,height=o.height||260,bands=Math.max(0,Math.floor(o.bands||0)),stops=(COLORMAPS[o.colormap||currentMap]||COLORMAPS.rainbow).stops;
    const min=Number(o.min)||0,max=Number.isFinite(Number(o.max))?Number(o.max):1,n=bands>0?bands:48,top=24,bottom=height-16,span=bottom-top,parts=[],ticks=[];
    const esc=s=>String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    for(let i=0;i<n;i++){const c=sampleStops(stops,bands>0?(i+0.5)/n:i/Math.max(1,n-1)).map(v=>Math.round(v*255));const y=bottom-(i+1)*span/n;parts.push(`<rect x="10" y="${y.toFixed(2)}" width="18" height="${(span/n+0.6).toFixed(2)}" fill="rgb(${c.join(',')})"/>`);}
    const nt=bands>0?bands+1:5;
    for(let i=0;i<nt;i++){const tq=i/(nt-1),y=bottom-tq*span;ticks.push(`<line x1="28" x2="32" y1="${y.toFixed(1)}" y2="${y.toFixed(1)}" stroke="#20374d" stroke-width="1"/><text x="35" y="${(y+4).toFixed(1)}" font-size="11" font-family="Arial,Helvetica,sans-serif" fill="#20374d">${esc(fmtTick(min+(max-min)*tq))}</text>`);}
    const title=esc(o.title||'')+(o.units?' ('+esc(o.units)+')':'');
    return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}"><text x="10" y="15" font-size="12" font-family="Arial,Helvetica,sans-serif" fill="#20374d">${title}</text>${parts.join('')}<rect x="10" y="${top}" width="18" height="${span}" fill="none" stroke="#8093a2" stroke-width="1"/>${ticks.join('')}</svg>`;
  }
  function exportFilenameLocal(o) {
    o=o||{};const safe=s=>String(s||'').replace(/[^\w一-鿿-]+/g,'_').replace(/^_+|_+$/g,'')||'x';
    return `${safe(o.case_id||'volume')}_${safe(o.view||'custom')}_${safe(o.field||'field')}_${Math.max(1,Math.round(Number(o.scale)||1))}x.${o.ext||'png'}`;
  }
  // ---- §21.4 label layout (v0.12.2): one plan for the page overlay and every composited export ----
  // Priority: user labels (measurements, annotations) > attention findings > manual > note > info findings >
  // max diameter > branch names.  User labels are never hidden; automatic ones may be (hideOverflow).
  const LABEL_PRIORITY={meas:100,annot:100,flabel_attention:90,flabel_manual:85,flabel_note:80,flabel_info:70,dlabel:50,blabel:30};
  function labelPriority(item) {return LABEL_PRIORITY[item.kind==='flabel'?'flabel_'+(item.severity||'note'):item.kind]||40;}
  // ``project(xyz)`` → {x,y} in CSS px or null; ``size(item)`` → {w,h} in CSS px.  The label wants to sit just
  // above its (lifted) point; ``declutter`` (WssReportCommon.declutterLabels) moves colliding ones.  Returns one
  // entry per projected item: {item, x, y (label centre), w, h, anchor {x,y}, moved, hidden}.
  function planLabels(items,options={}) {
    const project=options.project,size=options.size,gap=options.gap===undefined?3:options.gap,rows=[];
    for(const item of items||[]) {
      const at=project(item.xyz);if(!at||!item.text)continue;
      const anchor=project(item.anchor)||at,{w,h}=size(item);
      rows.push({item,w,h,anchor,x:at.x,y:at.y-h/2-gap,priority:labelPriority(item)});
    }
    const declutter=typeof options.declutter==='function'?options.declutter:null;
    let placed=null;
    if(declutter&&rows.length){try{placed=declutter(rows.map(r=>({x:r.x,y:r.y,w:r.w,h:r.h,priority:r.priority})),
      {padding:options.padding===undefined?3:options.padding,maxShift:options.maxShift,step:options.step,bounds:options.bounds,hideOverflow:Boolean(options.hideOverflow)});}catch(_){placed=null;}}
    return rows.map((r,i)=>{
      const p=placed&&placed[i]?placed[i]:{x:r.x,y:r.y,moved:false,hidden:false};
      const user=r.item.kind==='meas'||r.item.kind==='annot';
      return {item:r.item,x:p.x,y:p.y,w:r.w,h:r.h,anchor:r.anchor,moved:Boolean(p.moved),hidden:Boolean(p.hidden)&&!user};
    });
  }
  // End of a leader line: the point of the label box nearest to the anchor (the line stops at the box edge).
  function leaderEnd(p) {
    return {x:clamp(p.anchor.x,p.x-p.w/2,p.x+p.w/2),y:clamp(p.anchor.y,p.y-p.h/2,p.y+p.h/2)};
  }
  // Findings review (C16): decisions live in review.items[id]; manual findings in review.added.
  const TRUST_GLOSS={1:'trust_interpolation_uncovered',2:'trust_rough_surface',4:'trust_geometry_out_of_range',8:'trust_low_sample_support',16:'trust_near_opening'};
  function normalizeReview(doc) {
    const out={items:{},added:[]};if(!doc||typeof doc!=='object')return out;
    const items=doc.items&&typeof doc.items==='object'&&!Array.isArray(doc.items)?doc.items:{};
    for(const id of Object.keys(items)){const e=items[id];if(!e||typeof e!=='object')continue;const decision=e.decision==='confirmed'||e.decision==='rejected'?e.decision:null;const note=typeof e.note==='string'?e.note.slice(0,500):'';if(decision||note)out.items[id]={decision,note};}
    out.added=(Array.isArray(doc.added)?doc.added:[]).filter(x=>x&&typeof x==='object'&&typeof x.id==='string'&&Array.isArray(x.xyz_mm)&&x.xyz_mm.length===3).slice(0,30).map(x=>({...x,xyz_mm:x.xyz_mm.map(Number),text:String(x.text||'').slice(0,500),kind:'manual',severity:x.severity==='attention'?'attention':'note'}));
    return out;
  }
  function reviewDecision(review,id) {const e=review&&review.items?review.items[id]:null;return e&&(e.decision==='confirmed'||e.decision==='rejected')?e.decision:null;}
  function reviewNote(review,id) {const e=review&&review.items?review.items[id]:null;return e&&typeof e.note==='string'?e.note:'';}
  function findingsWithReview(items,review) {
    const auto=findingsSorted(items),added=(review&&Array.isArray(review.added)?review.added:[]).map(x=>({...x,manual:true,label:x.text}));
    const all=auto.concat(added),rank=x=>reviewDecision(review,x.id)==='rejected'?1:0;
    return all.map((x,i)=>({x,i})).sort((a,b)=>rank(a.x)-rank(b.x)||a.i-b.i).map(o=>o.x);
  }
  const core={planeBasis,rotatePlane,groupCenterline,centerlinePlane,automaticPlanes,nearestTangent,planeFromPicks,slabIndices,insideIndices,moduleSummary,moduleIndices,sideIndices,interactionDelta,dragAlong,positionStep,planeContour,contourLoops,selectLoop,closeChain,pointInLoop,scaleBarLength,scanlineInside,fillSection,speedField,statistics,
    color,setColormap,setBands,colormapNames,colormapCSS,desaturate,convertUnit,unitOptions,frameFromMeta,dirFromAligned,dirToAligned,worldFromAligned,alignedFromWorld,STANDARD_VIEWS,standardCamera,cameraToAligned,cameraFromAligned,
    viewDirections,fittedStandardCamera,formatNumber,frameText,releaseShort,withOffset,referenceLabel,warningItems,warningText,
    scaleEnds,scaleT,scaleValueAt,scaleColor,colorAtT,quantile,robustRange,symmetricRange,throughPlane,inPlane,arrowSamples,
    LABEL_PRIORITY,labelPriority,planLabels,leaderEnd,
    encodeView,decodeView,nearestBin,fractionForArc,arcAlongBranch,regionIndices,regionPressureDrop,sphereIndices,nearestPoint,probeRecord,trustLabels,findingsSorted,FINDING_KINDS,convexHull,inConvexHull,interpolateIDW,
    nearestLabels,filterFaces,sliceGridToRows,seriesFractions,probeToTSV,probeToCSV,labelText,colorbarSVGLocal,exportFilenameLocal,normalizeReview,reviewDecision,findingsWithReview,TRUST_GLOSS};
  root.VolumeViewerCore=core;
  if(typeof module!=='undefined'&&module.exports) module.exports=core;
  if(typeof document==='undefined') return;

  const $=id=>document.getElementById(id);
  const meta=JSON.parse($('wss-report-meta').textContent), raw=JSON.parse($('volume-arrays').textContent);
  function decode(key,Type=Float32Array) {
    const value=raw[key]; if(value===undefined) return null;
    const str=atob(value), bytes=new Uint8Array(str.length);
    for(let i=0;i<str.length;i++) bytes[i]=str.charCodeAt(i);
    return new Type(bytes.buffer);
  }
  function decodeLine(value) {
    const str=atob(value), bytes=new Uint8Array(str.length);
    for(let i=0;i<str.length;i++) bytes[i]=str.charCodeAt(i);
    return new Float32Array(bytes.buffer);
  }
  const pts=decode('pts'), vertices=decode('vertices'), faces=decode('faces',Uint32Array), walls=decode('is_wall',Uint8Array);
  const segments=decode('segment',Int32Array), pressure=decode('pressure_pa'), velocity=decode('velocity_m_s'), wallPressure=decode('wall_pressure_pa');
  const trust=decode('trust',Uint8Array), wallTrust=decode('wall_trust',Uint8Array);
  const sFromRoot=decode('s_from_root_mm'), distWall=decode('dist_to_wall_mm');
  const centerXyz=decode('center'), centerSeg=decode('center_segment',Int32Array), centerEdges=decode('edges',Uint32Array);
  let centerRadius=decode('center_radius_mm'), pointRadius=decode('radius_mm');
  // Reports built before the two radii were split stored the centreline radius under ``radius_mm``;
  // the array lengths tell them apart (prediction points vs centreline samples).
  if(pointRadius&&centerXyz&&pointRadius.length===centerXyz.length/3&&pointRadius.length!==pts.length/3) {
    if(!centerRadius)centerRadius=pointRadius;
    pointRadius=null;
  }
  const speed=velocity ? speedField(velocity) : null, interior=insideIndices(pts,walls);
  const groups=groupCenterline(centerXyz,centerSeg,decode('tangent'),centerEdges);
  const presets=automaticPlanes(groups), lines=(raw.streamlines||[]).map(l=>({points:decodeLine(l.points),speed:decodeLine(l.speed_m_s)}));
  const modules=raw.modules&&raw.modules.length ? raw.modules : moduleSummary(segments,walls);
  const bounds={min:[Infinity,Infinity,Infinity],max:[-Infinity,-Infinity,-Infinity]};
  for(let i=0;i<vertices.length;i++) {const k=i%3; bounds.min[k]=Math.min(bounds.min[k],vertices[i]);bounds.max[k]=Math.max(bounds.max[k],vertices[i]);}
  const center=mul(add(bounds.min,bounds.max),0.5), diagonal=Math.max(norm(sub(bounds.max,bounds.min)),1);
  const frame=frameFromMeta(meta.frame_transform);
  const branchName=sid=>{const name=(meta.branch_names||{})[String(sid)];return typeof name==='string' ? name : `分支 ${sid}`;};
  // Shared report library (contract §12): embedded by volume_report.py, required since phase 2.
  // The lookups stay guarded so the numerical core keeps working under plain Node tests.
  const common=()=>root.WssReportCommon;
  // Centreline tree for the measurement tools (C7): branch order, arc length and inscribed radius.
  const clGroups=(()=>{
    const c=common();
    if(!c||typeof c.buildCenterlineGroups!=='function'||!centerXyz||!centerXyz.length)return [];
    try{return c.buildCenterlineGroups({xyz:centerXyz,radius:centerRadius,edges:centerEdges,segment:centerSeg},meta.branch_names||{})||[];}catch(_){return [];}
  })();
  const fmt=v=>formatNumber(v);
  function option(select,value,label) {const o=document.createElement('option');o.value=String(value);o.textContent=label;select.appendChild(o);return o;}
  const setText=(id,text)=>{const el=$(id);if(el)el.textContent=text;};
  const setHidden=(id,hidden)=>{const el=$(id);if(el)el.hidden=Boolean(hidden);};
  const num=(id,fallback)=>{const el=$(id);const v=el?Number(el.value):NaN;return el&&el.value!==''&&Number.isFinite(v)?v:fallback;};
  const setVal=(id,value)=>{const el=$(id);if(el&&value!==undefined&&value!==null)el.value=String(value);};
  // display units
  let pressureUnit='Pa', velocityUnit='m/s';
  const unitOf=field=>field==='velocity'?velocityUnit:pressureUnit;
  const shown=(value,field)=>convertUnit(value,field,unitOf(field));
  const fmtField=(value,field)=>fmt(shown(value,field))+' '+unitOf(field);
  if(speed) option($('volume-field'),'velocity','速度大小 / 方向（m/s）');
  if(pressure) option($('volume-field'),'pressure','压力（Pa）');
  groups.forEach(g=>option($('slice-branch'),g.segment,branchName(g.segment)));
  groups.forEach(g=>option($('region-branch'),g.segment,branchName(g.segment)));
  if(raw.has_segments) {
    option($('volume-module'),'','全部模块');
    modules.forEach(m=>option($('volume-module'),m.segment,`${branchName(m.segment)} · ${m.interior} 个体内点`));
  } else {
    $('volume-module').disabled=true;option($('volume-module'),'','报告未提供模块标签');
  }
  presets.forEach((p,i)=>option($('auto-presets'),i,`${branchName(p.segment)} · ${Math.round(p.fraction*100)}%${p.nearEnd?'（端部内侧）':''}`));
  if(!groups.length) {$('slice-basis').value='z';const cl=$('slice-basis').querySelector('option[value="centerline"]');if(cl)cl.disabled=true;}
  if(!presets.length) {$('auto-presets').disabled=true;$('slice-apply-auto').disabled=true;option($('auto-presets'),'','没有可用中心线');}
  // colour map + bands (persisted per browser)
  const colormapSelect=$('colormap');
  for(const item of colormapNames()) option(colormapSelect,item.id,item.label);
  try{const saved=root.localStorage&&root.localStorage.getItem('wss-volume-colormap');if(saved&&COLORMAPS[saved])setColormap(saved);
      const bands=root.localStorage&&root.localStorage.getItem('wss-volume-bands');if(bands)setBands(bands);}catch(_){}
  colormapSelect.value=currentMap;
  for(const n of [0,4,6,8,12]) option($('color-bands'),n,n?`${n} 段`:'连续');
  setVal('color-bands',currentBands);
  for(const u of unitOptions('pressure')) option($('pressure-unit'),u,u);
  for(const u of unitOptions('velocity')) option($('velocity-unit'),u,u);
  setVal('pressure-unit',pressureUnit);setVal('velocity-unit',velocityUnit);
  const releaseId=(meta.model_release&&(meta.model_release.registry_id||meta.model_release.name||meta.model_release.release))||meta.release||meta.release_id||'';
  const releaseName=releaseShort(releaseId);
  setText('volume-subtitle',[meta.case_id||'',releaseId?releaseName:'',frameText(meta.model_frame)].filter(Boolean).join(' · '));
  {const el=$('volume-subtitle');if(el&&el.setAttribute)el.setAttribute('title',[meta.case_id||'',releaseId,frameText(meta.model_frame)].filter(Boolean).join(' · '));}
  setText('volume-protocol','统计基于体内预测点，采用点权重；截面不报告未经体积或面积加权的通量。坐标与厚度单位：mm。');
  const reference=meta.pressure_reference;
  setText('pressure-reference',pressure ? '压力参考：'+(typeof reference==='string'?reference:reference&&(reference.label||reference.description)||'请参阅该发布包的参考压定义；不能直接解释为绝对血压。') : '');
  setText('streamline-note',lines.length ? `已载入 ${lines.length} 条由预测速度向量积分的流线；离开有效支撑区域即停止。` : '本报告没有已积分流线；可查看体内速度点云、方向箭头和截面。');
  // Footer (§19.9): review status · release short name · generation time; hashes and identities live in the
  // 「技术信息」 popover (the ids footer-feature / footer-identity moved there unchanged).
  const localTimeText=(iso,seconds)=>{const c=common();if(!iso)return '';if(c&&typeof c.localTime==='function'){try{return c.localTime(iso,seconds?{seconds:true}:undefined);}catch(_){}}return String(iso).replace('T',' ').replace(/(:\d{2})(\.\d+)?([+-]\d{2}:\d{2}|Z)?$/,seconds?'$1':'');};
  const mappingAt=(()=>{const h=meta.audit&&Array.isArray(meta.audit.mapping_history)?meta.audit.mapping_history:[];for(let i=h.length-1;i>=0;i--)if(h[i]&&typeof h[i].at==='string')return h[i].at;return '';})();
  const createdIso=withOffset(meta.created_at,mappingAt);
  const techRows=[];
  {
    // Same wording as the wall report (W4): 审阅?：<b>状态</b> · 发布包? <b>短名</b> · 生成 时间 · [技术信息].
    const review=meta.review&&typeof meta.review==='object'?meta.review:null;
    const status=review?({reviewed:'已审阅',unreviewed:'待审阅',reopened:'已重新打开'}[review.status]||String(review.status||'待审阅')):'待审阅';
    const at=review&&review.at?localTimeText(review.at):'';
    setText('footer-review',`${status}${review&&review.by?' · '+review.by:''}${at?' · '+at:''}`);
    {const el=$('footer-review');if(el&&el.setAttribute)el.setAttribute('title',review&&review.note?'审阅备注：'+review.note:'审阅状态');}
    setText('footer-release',releaseName);
    {const el=$('footer-release');if(el&&el.setAttribute)el.setAttribute('title',releaseId||'');}
    setText('footer-time',`生成 ${createdIso?localTimeText(createdIso):'—'}`);
    {const el=$('footer-time');if(el&&el.setAttribute)el.setAttribute('title',createdIso?localTimeText(createdIso,true):'');}
    const fc=meta.feature_contract||{},mr=meta.model_release||{},mf=meta.model_frame||{},au=meta.audit||{};
    setText('footer-feature',[fc.version,fc.source_hash].filter(Boolean).join(' · ')||'未记录');
    setText('footer-identity',meta.run_identity?String(meta.run_identity):'未记录');
    const stamp=(iso,raw)=>iso?`${localTimeText(iso,true)}（${raw||iso}）`:'—';
    const nModels=Array.isArray(mr.models)?mr.models.length:Array.isArray(mr.weights)?mr.weights.length:null;
    techRows.push(['审阅状态',`${status}${review&&review.by?' · '+review.by:''}${review&&review.at?' · '+localTimeText(review.at,true):''}${review&&review.note?' · '+review.note:''}`,'review_status'],
      ['发布包',releaseId||'—','release'],
      ['发布包哈希',meta.release_hash||mr.fingerprint||'—',null],
      ['模型族',[mr.model_family||mr.family,nModels!==null?nModels+' 个模型':null].filter(Boolean).join(' · ')||'—',null],
      ['特征合同',null,'feature_contract','footer-feature'],
      ['run_identity',null,'run_identity','footer-identity'],
      ['输入 SHA256',meta.input_sha256||'—',null],
      ['模型帧',[mf.label||mf.target,mf.step!==null&&mf.step!==undefined?'step '+mf.step:'',mf.time_s!==null&&mf.time_s!==undefined?mf.time_s+' s':''].filter(Boolean).join(' · ')||'—',null],
      ['生成时间',createdIso?stamp(createdIso,meta.created_at):(meta.created_at||'—'),null],
      ['报告重建',stamp(au.report_rebuilt_at),null],
      ['摘要版本',meta.schema_version||'—',null],
      ['计算设备',[meta.device,meta.gpu].filter(Boolean).join(' · ')||'—',null],
      ['坐标架方向来源',frame?(frame.direction_source==='unknown_stl'?'按解剖坐标架推断（STL 无患者方向）':frame.direction_source||'—'):'无解剖坐标架',null]);
  }
  function renderTechInfo() {
    const box=$('tech-info-rows');if(!box||!box.replaceChildren)return;
    const keep={};for(const id of ['footer-feature','footer-identity']){const el=$(id);if(el)keep[id]=el;}
    box.replaceChildren();
    for(const [label,value,gloss,id] of techRows) {
      const dt=document.createElement('dt');dt.textContent=label;if(gloss)dt.appendChild(glossButton(gloss));
      let dd=id&&keep[id]?keep[id]:document.createElement('dd');if(!(id&&keep[id]))dd.textContent=value;
      box.append(dt,dd);
    }
  }
  function techInfoText() {return techRows.map(([label,value,,id])=>`${label}\t${id&&$(id)?$(id).textContent:value}`).join('\n');}
  let techOpen=false;
  function setTechInfo(open) {
    techOpen=Boolean(open);setHidden('tech-info',!techOpen);
    for(const id of ['tech-info-toggle','tech-info-menu']){const b=$(id);if(b&&b.setAttribute)b.setAttribute('aria-expanded',String(techOpen));}
    if(techOpen){renderTechInfo();setText('tech-info-status','');}
  }
  const fieldValues=()=> $('volume-field').value==='velocity'?speed:pressure;
  const fieldStats={},fieldRanges={};
  for(const [id,values] of [['velocity',speed],['pressure',pressure]]) if(values) {
    const stats=statistics(values,interior),range={min:stats.min??0,max:stats.max??1};
    if(id==='pressure'&&wallPressure)for(const value of wallPressure)if(Number.isFinite(value)){range.min=Math.min(range.min,value);range.max=Math.max(range.max,value);}
    if(range.max<=range.min)range.max=range.min+1e-6;
    fieldStats[id]=stats;fieldRanges[id]=range;
  }
  // ---- slice readability (用户试用反馈 7): section colour range, velocity log scale, through-plane velocity,
  // in-plane arrows.  The whole-field range (max in an iliac jet) left an aneurysm-sac section at 0.01–0.1 m/s
  // in the bottom 5 % of the colour bar; the section map now adapts to its own samples by default.
  let sliceRangeMode='section',sliceManual={min:null,max:null},sliceQuantity='speed',sliceArrows=true,logScale=false;
  let sliceScaleNow=null,legendScaleNow=null;
  const RANGE_MODES=['section','global','manual'];
  const quantityOf=field=>field==='velocity'?(sliceQuantity==='normal'?'normal':'speed'):'pressure';
  // Whole-field scale of the current display: speed (optionally log), pressure, or ±p99|v| for through-plane velocity.
  function globalScale(field,quantity) {
    if(field!=='velocity'){const r=fieldRanges.pressure||{min:0,max:1};return {min:r.min,max:r.max,source:'global'};}
    if(quantity==='normal'){const m=(fieldStats.velocity&&fieldStats.velocity.p99)||(fieldRanges.velocity||{}).max||1;return {min:-m,max:m,diverging:true,source:'global'};}
    const r=fieldRanges.velocity||{min:0,max:1};return {min:r.min,max:r.max,log:logScale,source:'global'};
  }
  // The slice's own scale: 本截面 = robust p2–p98 of the section samples (±p98 |v_n| when signed), 全局, or 手动.
  function sliceScale(field,quantity,samples) {
    const g=globalScale(field,quantity),signed=quantity==='normal',log=quantity==='speed'&&logScale;
    if(sliceRangeMode==='manual'){
      // Bounds typed for another quantity (field or v·n switched) restart from this section's own range.
      if(sliceManual.field&&(sliceManual.field!==field||sliceManual.quantity!==quantity)){const r=signed?symmetricRange(samples):robustRange(samples),base=r||g;sliceManual={min:base.min,max:base.max,field,quantity};}
      const lo=Number(sliceManual.min),hi=Number(sliceManual.max);
      if(sliceManual.min!==null&&sliceManual.max!==null&&Number.isFinite(lo)&&Number.isFinite(hi)&&hi>lo)return {min:lo,max:hi,log,diverging:signed,source:'manual'};
      return {...g,source:'fallback'};
    }
    if(sliceRangeMode==='global')return g;
    const r=signed?symmetricRange(samples):robustRange(samples);
    return r?{min:r.min,max:r.max,log,diverging:signed,source:'section',count:r.count}:{...g,source:'fallback'};
  }
  const quantityLabel=(field,quantity,language)=>quantity==='normal'?((language||lang)==='en'?'Through-plane velocity':'穿面速度'):fieldLabel(field,language);
  // Which way is positive for the through-plane velocity (the centreline tangent points downstream: verified on
  // LV_GUO_YOU / LIU_YU_MING, every branch's tangent follows increasing arc length and the mean v·t is positive).
  function normalSense(basis) {
    const tilted=Number(($('slice-pitch')||{}).value)||Number(($('slice-yaw')||{}).value);
    if(basis==='centerline')return tilted?'顺流为正、负值=回流（倾斜截面取法向分量）':'顺流为正、负值=回流';
    if(basis==='x'||basis==='y'||basis==='z')return `沿 +${basis.toUpperCase()} 轴为正`;
    return '沿截面法向为正';
  }
  const SCALE_SOURCE={section:['本截面自适应 p2–p98','this section, p2–p98'],global:['全局','whole field'],manual:['手动','manual'],fallback:['本截面样本不足，用全局','too few samples here, whole field']};
  // "色标 0.0079–0.118 m/s（本截面自适应）；对数；顺流为正、负值=回流" — used by the zoom caption, CSV header and legend.
  function scaleCaption(field,scale,quantity,language) {
    const en=language==='en',[lo,hi]=scaleEnds(scale),src=SCALE_SOURCE[scale.source]||SCALE_SOURCE.global;
    let text=`${en?'colour range':'色标'} ${fmt(shown(lo,field))}–${fmt(shown(hi,field))} ${unitOf(field)}（${src[en?1:0]}${scale.shared?(en?', shared by the series':'，系列共用'):''}）`;
    if(scale.log)text+=en?'; log scale':'；对数色标';
    if(quantity==='normal')text+=en?'; positive along the plane normal':'；'+normalSense(($('slice-basis')||{}).value);
    return text;
  }
  function legendNote(field,scale,quantity,inSlice) {
    const parts=[];
    if(field!=='velocity')parts.push('相对压力，非绝对血压');
    if(scale.log)parts.push('对数色标');
    if(inSlice) {
      if(quantity==='normal')parts.push(normalSense(($('slice-basis')||{}).value));
      const g=globalScale(field,quantity),[glo,ghi]=scaleEnds(g);
      if(scale.source==='section')parts.push(`本截面自适应 · 全局 ${fmt(shown(glo,field))}–${fmt(shown(ghi,field))}`);
      else parts.push(scale.source==='manual'?'手动色标':scale.source==='fallback'?'本截面样本不足，用全局色标':'全局色标');
    }
    parts.push(currentBands?`${currentBands} 段离散色带`:'连续色标');
    return parts.join('；');
  }
  let renderer=null,scene=null,camera=null,controls=null,content=null,contextMesh=null,planeMesh=null,markers=null,highlightGroup=null;
  // v0.14 (F5) render on demand: one frame per request (controls change, damping settling, resize, scene or UI
  // change) instead of a continuous loop.  Exports render once themselves before reading pixels.
  let renderQueued=0,drawFrame=null;
  function requestRender() {
    if(renderQueued||!drawFrame||typeof root.requestAnimationFrame!=='function')return;
    renderQueued=root.requestAnimationFrame(()=>{renderQueued=0;if(drawFrame)drawFrame();})||0;
  }
  let sliceSelected=false,gesture=null,cutActive=false,pickMode=false,picks=[],pickPlane=null,pickInfo='';
  // Slice gizmo: what a drag on the blue plane does, whether the pointer hovers it, and its helper objects.
  let dragMode='move',planeHover=false,sliceGizmo=null,planeEdge=null,planeArrow=null;
  let trustOverlay=false, probe=null, probePinned=false, probeEnabled=true, activeFinding=null, findingIndices=[], findingCenter=null, findingExtent=0, regionOn=false, regionSel=[], pointArc=null;
  let applyingRemote=false;
  let shortcuts=null;   // §19.9 keyboard layer (WssReportCommon.installShortcuts), installed at the end of start-up
  let started=false;    // set after the first full refresh (layout switches refresh the view note afterwards)
  // v1.1 view-state extensions (contract §12.2)
  let hiddenBranches=new Set(), vertexSegment=null, lineSegment=null, probeLog=[], measurements=[], annotations=[], presetName=null, lang='zh';
  // §17.3 automatic labels + §17.1 morphology.  ``labels.findings`` = how many findings get a 3-D chip
  // (0 = off), ``labels.branches`` = branch-name chips, ``labels.max_diameter`` = the max-diameter ring.
  let labelState={findings:0,branches:false,max_diameter:true};
  const morphology=meta.morphology&&typeof meta.morphology==='object'?meta.morphology:null;
  const morphMax=(()=>{
    const m=morphology&&morphology.aorta&&morphology.aorta.max;
    if(!m||typeof m!=='object'||!Number.isFinite(Number(m.max_diameter_mm)))return null;
    const xyz=Array.isArray(m.xyz_mm)&&m.xyz_mm.length===3&&m.xyz_mm.every(v=>Number.isFinite(Number(v)))?m.xyz_mm.map(Number):null;
    const poly=Array.isArray(m.polygon_world)?m.polygon_world.filter(p=>Array.isArray(p)&&p.length===3&&p.every(v=>Number.isFinite(Number(v)))).map(p=>p.map(Number)):[];
    return {...m,xyz_mm:xyz,polygon_world:poly.length>=3?poly:null};
  })();
  // Station diameters per branch (report meta keeps only s_from_root / max / equivalent, §17.3).
  const morphStations=(()=>{
    const out={};
    for(const b of (morphology&&Array.isArray(morphology.branches)?morphology.branches:[])) {
      const st=b&&b.stations;if(!st||typeof st!=='object')continue;
      const x=(Array.isArray(st.s_from_root_mm)?st.s_from_root_mm:[]).map(Number);
      if(!x.length)continue;
      const pick=key=>Array.isArray(st[key])&&st[key].length===x.length?st[key].map(v=>v===null?NaN:Number(v)):null;
      out[String(b.segment_id)]={s:x,max:pick('max_diameter_mm'),equiv:pick('equivalent_diameter_mm')};
    }
    return out;
  })();
  // C12 captions: the shared dictionary drives legend / colour bar / measurement text when lang === 'en'.
  const fieldLabel=(field,language)=>{
    const zh=field==='velocity'?'速度':'相对压力';
    if((language||lang)!=='en')return zh;
    const c=common();
    if(c&&typeof c.englishLabel==='function'){const v=c.englishLabel(field==='velocity'?'speed':'pressure','en');if(typeof v==='string'&&v)return v;}
    return labelText(zh,'en');
  };
  const branchLabel=(name,language)=>labelText(String(name||''),language||lang);
  const viewLabel=(name,language)=>{
    const c=common();
    if(c&&typeof c.englishLabel==='function'){const v=c.englishLabel('view_'+name,language||lang);if(typeof v==='string'&&v)return v;}
    return (STANDARD_VIEWS[name]||{}).label||name;
  };
  let exportOptions={scale:1,background:'white',colorbar:'overlay',ui:true};
  // C7 / C8 / C9 interaction state (declared before the renderer so the animation loop can read it).
  let measureMode=null, measurePicks=[], annotMode=false, annotLocked=false, annotSaveTimer=null, presetList=[];
  // §21.4 label layout state (declared before the renderer: the animation loop reads it)
  let labelsDirty=true,labelKey='',labelPlan=[];
  const labelSizeCache=new Map();
  let findingReview=normalizeReview(meta.findings&&meta.findings.review), addFindingMode=false, csrfToken=null;
  // Online = served by the workbench (same origin, http/https): server copies of findings review / annotations / preferences apply.
  const online=(()=>{try{const loc=root.location;if(!loc||!/^https?:$/.test(loc.protocol)||typeof fetch!=='function')return null;const path=String(loc.pathname||'');
    let m=path.match(/^(.*)\/api\/jobs\/([A-Za-z0-9_-]{1,80})\/(?:report|files\/report\.html)$/);if(!m)m=path.match(/^(.*)\/jobs\/([A-Za-z0-9_-]{1,80})\/report\.html$/);if(!m)return null;
    return {job_id:m[2],api:`${m[1]}/api/jobs/${m[2]}/`,root:m[1]+'/'};}catch(_){return null;}})();
  const pageOrigin=(()=>{try{const loc=root.location;return loc&&/^https?:$/.test(loc.protocol)&&loc.origin?loc.origin:null;}catch(_){return null;}})();
  // v0.15: 「← 工作台」 back to this case in the workbench — only on a page the service serves, never inside the compare frame.
  (()=>{const a=$('back-to-workbench');if(!a||!online)return;let framed=false;try{framed=root.parent&&root.parent!==root;}catch(_){framed=true;}if(framed)return;
    a.href=online.root+'#job='+online.job_id;a.hidden=false;const h=a.parentNode;if(h&&h.classList)h.classList.add('has-back');})();
  function postParent(message) {if(!pageOrigin||!root.parent||root.parent===root)return false;try{root.parent.postMessage(message,pageOrigin);return true;}catch(_){return false;}}
  async function csrf() {
    if(csrfToken)return csrfToken;
    const r=await fetch(online.root+'api/session',{credentials:'same-origin'});if(!r.ok)throw new Error('会话失效，请刷新工作台');
    const j=await r.json();csrfToken=j&&j.csrf_token;if(!csrfToken)throw new Error('会话失效，请刷新工作台');return csrfToken;
  }
  async function apiPut(name,body) {
    const token=await csrf();
    const r=await fetch(online.api+name,{method:'PUT',credentials:'same-origin',headers:{'Content-Type':'application/json','X-CSRF-Token':token},body:JSON.stringify(body)});
    let j={};try{j=await r.json();}catch(_){}
    if(!r.ok){const err=new Error(j&&j.error&&j.error.message||`HTTP ${r.status}`);err.status=r.status;throw err;}
    return j;
  }
  async function apiGetFile(name) {
    const r=await fetch(online.api+'files/'+name,{credentials:'same-origin'});if(r.status===404)return null;if(!r.ok)throw new Error(`HTTP ${r.status}`);return r.json();
  }
  // Glossary (C17): embedded JSON rendered by the shared popover; the template element stays as fallback.
  let glossary={};try{const el=$('wss-glossary');const doc=JSON.parse(el&&el.textContent||'{}');glossary=doc&&doc.terms&&typeof doc.terms==='object'?doc.terms:{};}catch(_){glossary={};}
  const glossPop=(()=>{
    const c=common();
    if(!c||typeof c.glossaryPopover!=='function')return null;
    try{return c.glossaryPopover({glossary:{terms:glossary},lang,mount:document.body});}catch(_){return null;}
  })();
  function glossaryEntry(key) {const t=glossary[key];if(!t)return null;return lang==='en'?{title:t.en||t.zh||key,desc:t.en_desc||t.zh_desc||''}:{title:t.zh||t.en||key,desc:t.zh_desc||t.en_desc||''};}
  function showGloss(key,anchor) {
    if(glossPop&&glossary[key]){try{if(glossPop.setLang)glossPop.setLang(lang);if(glossPop.show(key,anchor))return;}catch(_){}}
    const pop=$('gloss-pop');if(!pop)return;const e=glossaryEntry(key);if(!e){pop.hidden=true;return;}
    setText('gloss-title',e.title);setText('gloss-desc',e.desc);pop.hidden=false;
    try{if(anchor&&anchor.getBoundingClientRect&&pop.style){const r=anchor.getBoundingClientRect();pop.style.left=Math.max(8,Math.min(r.left,(root.innerWidth||800)-310))+'px';pop.style.top=Math.min(r.bottom+6,(root.innerHeight||600)-120)+'px';}}catch(_){}
  }
  function hideGloss() {if(glossPop){try{glossPop.hide();}catch(_){}}const pop=$('gloss-pop');if(pop)pop.hidden=true;}
  if(typeof document.addEventListener==='function')document.addEventListener('click',event=>{
    const t=event&&event.target;const btn=t&&t.closest?t.closest('.gloss'):null;
    if(btn){if(event.preventDefault)event.preventDefault();if(event.stopPropagation)event.stopPropagation();showGloss(btn.getAttribute?btn.getAttribute('data-gloss'):null,btn);return;}
    if(t&&t.closest&&t.closest('#gloss-pop'))return;hideGloss();
  });
  {const close=$('gloss-close');if(close)close.addEventListener('click',hideGloss);}
  function glossButton(key) {const b=document.createElement('button');b.type='button';b.className='gloss';b.textContent='?';if(b.setAttribute)b.setAttribute('data-gloss',key);return b;}
  const view=$('volume-view');
  const MODES=['cloud','slice','wall','streamlines'];
  const modeButton=mode=>$('mode-'+mode);
  // A display tab is only unavailable when the report lacks its data; 「壁面压力」 and 「流线」 switch the physical
  // quantity themselves (they used to be greyed out whenever the other quantity was selected, v0.12 audit).
  const MODE_FIELD={wall:'pressure',streamlines:'velocity'};
  function modeAvailable(mode) {
    if(mode==='wall')return Boolean(pressure&&wallPressure);
    if(mode==='streamlines')return Boolean(speed&&lines.length);
    return MODES.includes(mode);
  }
  const MODE_MISSING={wall:'本报告没有壁面压力（发布包未预测压力或未导出壁面插值）',streamlines:'本报告没有流线（未预测速度或未积分流线）'};
  const MODE_TITLES={cloud:'体内预测点云（快捷键 T 循环页签）',slice:'有限厚度截面与平面投影',wall:'壁面压力（自动切到压力）',streamlines:'由预测速度积分的流线（自动切到速度）'};
  function setField(field) {
    const values=field==='velocity'?speed:field==='pressure'?pressure:null;if(!values)return false;
    const select=$('volume-field');if(select.value===field)return true;
    select.value=field;refresh();return true;
  }
  function setMode(mode) {
    if(!MODES.includes(mode)||!modeAvailable(mode))return;
    const select=$('volume-mode'),want=MODE_FIELD[mode];
    if(want&&$('volume-field').value!==want)$('volume-field').value=want;
    select.value=mode;sliceSelected=mode==='slice';refresh();
  }
  function cycleMode() {
    const current=MODES.indexOf($('volume-mode').value);
    for(let k=1;k<=MODES.length;k++){const next=MODES[(current+k)%MODES.length];if(modeAvailable(next)){setMode(next);return next;}}
    return null;
  }
  function refreshModeTabs() {
    const select=$('volume-mode');
    for(const mode of MODES) {
      const button=modeButton(mode);if(!button)continue;
      const available=modeAvailable(mode);
      button.disabled=!available;
      if(button.setAttribute)button.setAttribute('title',available?MODE_TITLES[mode]:MODE_MISSING[mode]||'');
      const active=select.value===mode;
      if(button.classList)button.classList.toggle('on',active);
      if(button.setAttribute)button.setAttribute('aria-pressed',String(active));
    }
  }
  function cameraState() {
    if(!camera||!controls)return null;
    return {position:[camera.position.x,camera.position.y,camera.position.z],target:controls.target.toArray(),up:[camera.up.x,camera.up.y,camera.up.z]};
  }
  // OrbitControls caches camera.up when it is constructed (its azimuth axis); a camera whose up differs from that
  // cached axis would tumble instead of spinning about the vessel, so the controls are rebuilt when up changes.
  let controlsUp=null,autoView=null;
  function makeControls() {
    if(controls&&typeof controls.dispose==='function'){try{controls.dispose();}catch(_){}}
    controls=new THREE.OrbitControls(camera,renderer.domElement);controls.enableDamping=true;
    controlsUp=[camera.up.x,camera.up.y,camera.up.z];
    // Any user orbit / zoom / pan ends the automatic framing (the view is no longer refitted on resize).
    if(typeof controls.addEventListener==='function'){controls.addEventListener('start',()=>{autoView=null;requestRender();});controls.addEventListener('change',requestRender);}
  }
  function setCamera(cam) {
    if(!camera||!controls||!cam)return;
    if(cam.up){
      camera.up.set(...cam.up);
      const u=[camera.up.x,camera.up.y,camera.up.z];
      if(!controlsUp||Math.abs(u[0]-controlsUp[0])+Math.abs(u[1]-controlsUp[1])+Math.abs(u[2]-controlsUp[2])>1e-6)makeControls();
    }
    if(cam.target)controls.target.set(...cam.target);
    if(cam.position)camera.position.set(...cam.position);
    camera.updateProjectionMatrix();controls.update();requestRender();
  }
  // Standard / default views framed to fill the viewport (§19.9); ``autoView`` remembers which one is shown so
  // a resize (compact layout, menu column, window) refits it until the user moves the camera.
  function fittedCamera(name) {
    if(!camera)return null;
    const useName=frame&&STANDARD_VIEWS[name]?name:'legacy';
    if(useName==='legacy'&&name!=='front')return null;
    return fittedStandardCamera(useName,frame,vertices,{fov:camera.fov,aspect:camera.aspect,margin:1.12,center,distance:diagonal*1.5});
  }
  function showStandardView(name) {
    const cam=fittedCamera(name);if(!cam)return false;
    setCamera(cam);autoView=name;return true;
  }
  let hoverAt=0;
  try {
    renderer=new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true});renderer.setPixelRatio(Math.min(root.devicePixelRatio||1,2));
    renderer.setClearColor(0xeef3f7);renderer.domElement.setAttribute('role','img');renderer.domElement.setAttribute('aria-label','三维体场视图，可旋转、缩放并读取体内数值');view.appendChild(renderer.domElement);
    scene=new THREE.Scene();camera=new THREE.PerspectiveCamera(42,1,diagonal/1000,diagonal*100);
    // Up = the anatomical head direction before the controls exist, so orbiting spins about the vessel axis.
    {const up=frame?viewDirections('front',frame).up:LEGACY_VIEW.up;camera.up.set(up[0],up[1],up[2]);}
    makeControls();
    scene.add(new THREE.AmbientLight(0xffffff,0.75));const light=new THREE.DirectionalLight(0xffffff,0.55);light.position.set(1,1,2);scene.add(light);
    const fill=new THREE.DirectionalLight(0xffffff,0.25);fill.position.set(-1,-0.5,-1);scene.add(fill);
    content=new THREE.Group();scene.add(content);markers=new THREE.Group();scene.add(markers);highlightGroup=new THREE.Group();scene.add(highlightGroup);
    const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.BufferAttribute(vertices,3));geometry.setIndex(new THREE.BufferAttribute(faces,1));geometry.computeVertexNormals();
    contextMesh=new THREE.Mesh(geometry,new THREE.MeshPhongMaterial({color:0x849bae,transparent:true,opacity:.1,side:THREE.DoubleSide,depthWrite:false}));scene.add(contextMesh);
    renderer.localClippingEnabled=true;
    // The slice is a grabbable gizmo: a local quad (sized to the vessel), its outline and a normal arrow.
    planeMesh=new THREE.Mesh(new THREE.PlaneGeometry(1,1),new THREE.MeshBasicMaterial({color:0x559cbb,transparent:true,opacity:.16,side:THREE.DoubleSide,depthWrite:false}));planeMesh.renderOrder=4;
    planeEdge=new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints([[-.5,-.5,0],[.5,-.5,0],[.5,.5,0],[-.5,.5,0]].map(q=>new THREE.Vector3(...q))),new THREE.LineBasicMaterial({color:0x1f6f8b,depthTest:false}));planeEdge.renderOrder=6;
    planeArrow=new THREE.ArrowHelper(new THREE.Vector3(0,0,1),new THREE.Vector3(0,0,0),1,0x1f6f8b,.2,.1);
    sliceGizmo=new THREE.Group();sliceGizmo.add(planeMesh);sliceGizmo.add(planeEdge);sliceGizmo.add(planeArrow);scene.add(sliceGizmo);
    function resize() {
      const width=Math.max(view.clientWidth||0,1),height=Math.max(view.clientHeight||0,1);renderer.setSize(width,height);camera.aspect=width/height;camera.updateProjectionMatrix();
      if(autoView)showStandardView(autoView);
      requestRender();
    }
    if(root.ResizeObserver) new ResizeObserver(resize).observe(view);else root.addEventListener('resize',resize);
    resize();
    function setSlider(id,value) {const el=$(id);if(!el)return;const min=Number(el.min),max=Number(el.max);el.value=String(clamp(value,min,max));el.dispatchEvent(new Event('input',{bubbles:true}));}
    function screenDir(origin,axis) {
      const a=new THREE.Vector3(...origin).project(camera),b=new THREE.Vector3(...add(origin,axis)).project(camera);
      return [(b.x-a.x)*renderer.domElement.clientWidth/2,-(b.y-a.y)*renderer.domElement.clientHeight/2];
    }
    function mmPerPixel(origin) {
      const dist=camera.position.distanceTo(new THREE.Vector3(...origin));
      return 2*dist*Math.tan(camera.fov*Math.PI/360)/Math.max(renderer.domElement.clientHeight,1);
    }
    function moveAlongNormal(dMm) {
      if(!dMm)return;const basis=$('slice-basis').value;let arc=0,extent=0;
      if(basis==='centerline'){const g=groups.find(x=>x.segment===Number($('slice-branch').value))||groups[0];arc=g?g.arc[g.arc.length-1]:0;}
      else if(basis!=='pick'){const axis={x:0,y:1,z:2}[basis]??2;extent=bounds.max[axis]-bounds.min[axis];}
      setSlider('slice-position',Number($('slice-position').value)+positionStep(basis,dMm,arc,extent));
    }
    // Drag on the blue plane: move along its normal (default), rotate (Shift or 旋转 mode) or slide in-plane (Alt or 平移 mode).
    function applyPlaneDrag(kind,dx,dy) {
      const plane=currentPlane(),origin=plane.origin;
      if(kind==='rotate'){setSlider('slice-yaw',Number($('slice-yaw').value)+dx*.3);setSlider('slice-pitch',Number($('slice-pitch').value)-dy*.3);return;}
      const px=mmPerPixel(origin);
      if(kind==='offset'){setSlider('slice-offset-u',Number($('slice-offset-u').value)+dragAlong(dx,dy,screenDir(origin,plane.u),px));setSlider('slice-offset-v',Number($('slice-offset-v').value)+dragAlong(dx,dy,screenDir(origin,plane.v),px));return;}
      moveAlongNormal(dragAlong(dx,dy,screenDir(origin,plane.normal),px));
    }
    function updateGizmoStyle() {
      if(!planeMesh)return;requestRender();const active=planeHover||Boolean(gesture);
      planeMesh.material.opacity=active?.3:.16;planeEdge.material.color.setHex(active?0xd97706:0x1f6f8b);planeArrow.setColor(new THREE.Color(active?0xd97706:0x1f6f8b));
      if(renderer.domElement.style)renderer.domElement.style.cursor=gesture?'grabbing':planeHover?'grab':'';
    }
    const gizmoIdle=()=>sliceSelected&&!pickMode&&!measureMode&&!annotMode&&!addFindingMode;
    // Wheel over the plane moves it along its normal (Shift: thickness); elsewhere the wheel zooms the view as usual.
    renderer.domElement.addEventListener('wheel',event=>{
      if(!gizmoIdle())return;
      if(!planeHover&&!event.altKey&&!castAt(event.clientX,event.clientY,[planeMesh]))return;
      event.preventDefault();event.stopImmediatePropagation();
      if(event.shiftKey){setSlider('slice-thickness',Number($('slice-thickness').value)+(event.deltaY>0?.4:-.4));return;}
      moveAlongNormal((event.deltaY>0?-1:1)*(event.ctrlKey||event.metaKey?5:1));
    },{passive:false,capture:true});
    // Keyboard nudges while a slice is shown: arrows move / rotate, PgUp/PgDn tilt, [ ] change thickness, Esc ends picking.
    root.addEventListener('keydown',event=>{
      // A key already consumed by the shortcut layer (help overlay, Esc on a popover) is not a slice nudge.
      if(event.defaultPrevented)return;
      const tag=event.target&&event.target.tagName?String(event.target.tagName).toUpperCase():'';
      if(tag==='INPUT'||tag==='SELECT'||tag==='TEXTAREA')return;
      if(event.key==='Escape'){if(sliceZoomOpen){openSliceZoom(false);event.preventDefault();return;}if(pickMode){setPickMode(false);event.preventDefault();}return;}
      if(!gizmoIdle())return;
      const step=event.shiftKey?5:1,deg=event.shiftKey?10:2;let used=true;
      switch(event.key){
        case 'ArrowUp':moveAlongNormal(step);break;case 'ArrowDown':moveAlongNormal(-step);break;
        case 'ArrowLeft':setSlider('slice-yaw',Number($('slice-yaw').value)-deg);break;case 'ArrowRight':setSlider('slice-yaw',Number($('slice-yaw').value)+deg);break;
        case 'PageUp':setSlider('slice-pitch',Number($('slice-pitch').value)+deg);break;case 'PageDown':setSlider('slice-pitch',Number($('slice-pitch').value)-deg);break;
        case '[':setSlider('slice-thickness',Number($('slice-thickness').value)-.4);break;case ']':setSlider('slice-thickness',Number($('slice-thickness').value)+.4);break;
        default:used=false;
      }
      if(used)event.preventDefault();
    });
    const raycaster=new THREE.Raycaster(),pointer=new THREE.Vector2();
    let pressStart=null;
    function castAt(clientX,clientY,targets) {
      const rect=renderer.domElement.getBoundingClientRect();
      pointer.set((clientX-rect.left)/rect.width*2-1,-((clientY-rect.top)/rect.height)*2+1);
      raycaster.setFromCamera(pointer,camera);raycaster.params.Points={threshold:diagonal/280};
      const hits=raycaster.intersectObjects(targets,false);
      return hits.length?hits[0]:null;
    }
    function pickAt(clientX,clientY) {
      const hit=castAt(clientX,clientY,[contextMesh].concat(content?content.children.filter(o=>o.isPoints||o.isMesh):[]));
      return hit?[hit.point.x,hit.point.y,hit.point.z]:null;
    }
    // Probe: hover over interior points / the coloured wall; click pins.  The
    // transparent context wall answers clicks only, so hovering stays quiet.
    function probeAt(clientX,clientY,includeContext) {
      const targets=content?content.children.filter(o=>o.isPoints||(o.isMesh&&o.userData&&o.userData.kind==='wall')):[];
      if(includeContext)targets.push(contextMesh);
      const hit=castAt(clientX,clientY,targets);if(!hit)return null;
      if(hit.object.isPoints) {const map=hit.object.userData&&hit.object.userData.indices;const i=map?map[hit.index]:hit.index;return {kind:'interior',index:i};}
      if(hit.face) {
        const p=[hit.point.x,hit.point.y,hit.point.z];let best=hit.face.a,bestD=Infinity;
        for(const v of [hit.face.a,hit.face.b,hit.face.c]){const d=norm(sub(point(vertices,v),p));if(d<bestD){bestD=d;best=v;}}
        return {kind:'wall',index:best};
      }
      return null;
    }
    renderer.domElement.addEventListener('pointerdown',event=>{
      if(event.button!==0||!gizmoIdle()||!castAt(event.clientX,event.clientY,[planeMesh]))return;
      gesture={x:event.clientX,y:event.clientY,id:event.pointerId,kind:event.shiftKey?'rotate':event.altKey?'offset':dragMode};pressStart=null;
      if(controls)controls.enabled=false;renderer.domElement.setPointerCapture?.(event.pointerId);
      event.preventDefault();event.stopImmediatePropagation();updateGizmoStyle();
    },{capture:true});
    renderer.domElement.addEventListener('pointerdown',event=>{
      if(event.button!==0)return;
      pressStart={x:event.clientX,y:event.clientY,id:event.pointerId};
    });
    renderer.domElement.addEventListener('pointermove',event=>{
      if(gesture&&gesture.id===event.pointerId) {
        const dx=event.clientX-gesture.x,dy=event.clientY-gesture.y;gesture.x=event.clientX;gesture.y=event.clientY;applyPlaneDrag(gesture.kind,dx,dy);event.preventDefault();return;
      }
      if(!pressStart&&gizmoIdle()){const on=Boolean(castAt(event.clientX,event.clientY,[planeMesh]));if(on!==planeHover){planeHover=on;updateGizmoStyle();}}
      if(probePinned||pressStart||!probeEnabled)return;
      const now=Date.now();if(now-hoverAt<60)return;hoverAt=now;
      const hit=probeAt(event.clientX,event.clientY,false);
      if(hit){probe=hit;renderProbe();}
    });
    const endGesture=event=>{
      if(gesture&&gesture.id===event.pointerId){gesture=null;if(controls)controls.enabled=true;updateGizmoStyle();}
      const click=pressStart&&pressStart.id===event.pointerId&&Math.hypot(event.clientX-pressStart.x,event.clientY-pressStart.y)<6;
      if(click&&addFindingMode) {
        const p=pickAt(event.clientX,event.clientY);
        if(p){addManualFinding(p,($('finding-add-text')||{}).value||'');addFindingMode=false;}else setText('findings-review-status','未点到血管，请点击壁面或体内点。');
      } else if(click&&annotMode) {
        const p=pickAt(event.clientX,event.clientY);
        if(p){if(addAnnotationHere(p))setAnnotMode(false);}else setText('annot-status','未点到血管，请点击壁面或体内点。');
      } else if(click&&measureMode) {
        const p=pickAt(event.clientX,event.clientY);
        if(p)addMeasurePick(p);else setText('measure-note','未点到血管，请点击壁面或体内点。');
      } else if(click&&pickMode) {
        const p=pickAt(event.clientX,event.clientY);
        if(p) addPick(p); else setText('pick-status','未点到血管，请点击半透明壁面或体内点。');
      } else if(click&&probeEnabled) {
        const hit=probeAt(event.clientX,event.clientY,true);
        if(hit){probe=hit;probePinned=true;} else {probePinned=false;}
        renderProbe();
      }
      pressStart=null;
    };
    renderer.domElement.addEventListener('pointerup',endGesture);renderer.domElement.addEventListener('pointercancel',endGesture);
    let lastCam='',lastPost=0;
    function animate() {
      // Keep drawing only while the orbit damping is still settling (update() reports a camera change).
      if(controls.update())requestRender();renderer.render(scene,camera);
      // measurement / annotation labels follow the camera (throttled; no work when there are none)
      // §21.4: labels are laid out again only when the camera / viewport changed or the label set did (once per frame at most)
      if(started&&(labelsDirty||(measurements.length||annotations.length||hasAutoLabels())&&cameraKey()!==labelKey))layoutLabelsNow();
      // camera link (contract §6): post the anatomical-frame camera to the parent page, throttled
      if(frame&&root.parent&&root.parent!==root&&!applyingRemote) {
        const now=Date.now(),cam=cameraState(),key=JSON.stringify(cam);if(key===lastCam)return;
        if(now-lastPost<50){requestRender();return;}   // throttled: draw once more so the final camera still goes out
        lastCam=key;lastPost=now;
        try{root.parent.postMessage({type:'wss-view:camera',family:'volume',camera:cameraToAligned(cam,frame)},'*');}catch(_){}
      }
    }
    drawFrame=animate;
    // Any UI input may change the scene; the requests coalesce into one frame.  Hover over the canvas can move the
    // probe marker or the slice gizmo highlight, so pointer moves there ask for a frame as well.
    if(root.document&&typeof root.document.addEventListener==='function')for(const type of ['input','change','click','keydown','pointerup'])root.document.addEventListener(type,requestRender,true);
    renderer.domElement.addEventListener('pointermove',requestRender);renderer.domElement.addEventListener('wheel',requestRender,{passive:true});
    animate();
  } catch(error) {drawFrame=null;renderer=null;camera=null;controls=null;content=null;$('volume-error').hidden=false;$('volume-error').textContent='三维显示不可用：'+error.message+'。仍可使用截面投影与统计。';}
  root.addEventListener('message',event=>{
    const data=event&&event.data;if(!data||typeof data!=='object'||typeof data.type!=='string'||!data.type.startsWith('wss-view:'))return;
    requestRender();
    if(data.type==='wss-view:set-camera') {
      if(!frame||!data.camera)return;
      applyingRemote=true;try{setCamera(cameraFromAligned(data.camera,frame));autoView=null;}finally{setTimeout(()=>{applyingRemote=false;},120);}
      return;
    }
    // Batch export / apply-state protocol (contract §12.6): same origin only; disabled for file:// reports.
    if(!pageOrigin||event.origin!==pageOrigin)return;
    const id=data.request_id;
    // §15.6: the compare page asks for the full view state so it can forward the display subset.
    if(data.type==='wss-view:get-state') {
      try{postParent({type:'wss-view:state',request_id:id,family:'volume',state:captureView()});}
      catch(err){postParent({type:'wss-view:error',request_id:id,message:err&&err.message||String(err)});}
      return;
    }
    if(data.type==='wss-view:apply-state') {
      try {
        let state=data.state&&typeof data.state==='object'?data.state:{};
        if(typeof state.preset_name==='string'&&state.preset_name) {
          // C9: a named preset is expanded against this case, merged onto the current state; explicit keys win.
          const partial=resolvePreset(state.preset_name);
          if(!partial)throw new Error(common()?`没有该预设：${state.preset_name}`:'预设需要共享库');
          state=mergeState(mergeState(captureView(),partial),state);
        }
        applyView(state);postParent({type:'wss-view:applied',request_id:id});
      } catch(err){postParent({type:'wss-view:error',request_id:id,message:err&&err.message||String(err)});}
      return;
    }
    if(data.type==='wss-view:export') {
      const o={...exportOptions,lang,...(data.options&&typeof data.options==='object'?data.options:{})};
      const r=renderExport(o);
      if(!r||r.error){postParent({type:'wss-view:error',request_id:id,message:r&&r.error||'导出失败'});return;}
      const field=$('volume-field').value==='velocity'?'speed':'pressure';
      const message={type:'wss-view:exported',request_id:id,png_base64:r.dataUrl.replace(/^data:image\/png;base64,/,''),filename:typeof o.filename==='string'&&o.filename?o.filename:exportFilenameLocal({case_id:meta.case_id,view:'custom',field,scale:r.scale}),width:r.width,height:r.height,downgraded:r.downgraded};
      if(o.colorbar==='svg')message.colorbar_svg=colorbarSvgCurrent(o.lang);
      postParent(message);
    }
  });
  // 复位视角 / 0 / R: the anatomical front view filling the viewport; reports without a frame keep the old
  // oblique direction, fitted the same way.
  function fit() {if(!camera)return;showStandardView('front');}
  fit();
  function flyTo(xyz,extent) {
    if(!camera||!controls||!xyz)return;
    const current=cameraState(), dir=unit(sub(current.position,current.target)), distance=Math.max((extent||0)*5,diagonal*.12);
    setCamera({position:add(xyz,mul(dir,distance)),target:xyz.slice(),up:current.up});autoView=null;
  }
  function drawMarkers() {
    requestRender();
    if(!markers){renderLabels();return;}
    while(markers.children.length){const o=markers.children[0];markers.remove(o);o.geometry&&o.geometry.dispose();o.material&&o.material.dispose();}
    const sphere=(p,hex,scale)=>{const s=new THREE.Mesh(new THREE.SphereGeometry(diagonal/(scale||160),14,10),new THREE.MeshBasicMaterial({color:hex,depthTest:false}));s.position.set(...p);s.renderOrder=5;markers.add(s);};
    const segment=(a,b,hex)=>{const g=new THREE.BufferGeometry().setFromPoints([a,b].map(p=>new THREE.Vector3(...p)));const l=new THREE.Line(g,new THREE.LineBasicMaterial({color:hex,depthTest:false}));l.renderOrder=5;markers.add(l);};
    picks.forEach((p,i)=>sphere(p,i===0?0xffb400:0xff4d6d));
    if(picks.length===2)segment(picks[0],picks[1],0xff4d6d);
    // C7: endpoints and the segment between them, plus any pick still waiting for its partner
    measurePicks.forEach(p=>sphere(p,0x1fa33a,200));
    for(const m of measurements) {
      const points=(m.points||[]).filter(p=>Array.isArray(p)&&p.length===3).map(p=>p.map(Number));
      points.forEach(p=>sphere(p,0x0f7d8c,200));
      if(points.length>1)segment(points[0],points[1],0x0f7d8c);
    }
    // C8: a dot at the pin and a leader line to where the text label floats
    for(const a of annotations) {
      if(!Array.isArray(a.xyz_mm)||a.xyz_mm.length!==3)continue;
      const p=a.xyz_mm.map(Number);
      sphere(p,0xd97706,220);segment(p,add(p,[0,0,diagonal*0.05]),0xd97706);
    }
    // §17.3 max-diameter ring: the station's closed wall section, drawn on top of the geometry.
    if(labelState.max_diameter&&morphMax&&morphMax.polygon_world&&typeof THREE.LineLoop==='function') {
      try {
        const geometry=new THREE.BufferGeometry().setFromPoints(morphMax.polygon_world.map(q=>new THREE.Vector3(q[0],q[1],q[2])));
        const ring=new THREE.LineLoop(geometry,new THREE.LineBasicMaterial({color:0x8e44ad,depthTest:false,transparent:true,opacity:.95}));
        ring.renderOrder=6;markers.add(ring);
        if(morphMax.xyz_mm)sphere(morphMax.xyz_mm,0x8e44ad,260);
      } catch(_){}
    }
    renderLabels();
  }
  function gizmoSide(p) {
    let best=Infinity,row=null;
    for(const g of groups)for(let i=0;i<g.points.length;i++){const d=norm(sub(g.points[i],p));if(d<best){best=d;row=g.rows[i];}}
    const r=row!==null&&centerRadius&&Number.isFinite(centerRadius[row])?centerRadius[row]:diagonal/40;
    return clamp(r*8,24,diagonal*.5);
  }
  function setDragMode(mode) {
    dragMode=mode;
    for(const [id,kind] of [['drag-move','move'],['drag-rotate','rotate'],['drag-offset','offset']]){const b=$(id);if(!b)continue;if(b.classList)b.classList.toggle('on',kind===mode);if(b.setAttribute)b.setAttribute('aria-pressed',String(kind===mode));}
  }
  function addPick(p) {
    if(picks.length>=2) picks=[];
    picks.push(p);
    const near=nearestTangent(groups,p)||{tangent:[0,0,1],segment:null};
    pickPlane=planeFromPicks(picks,near.tangent);pickPlane.segment=null;
    pickInfo=picks.length===1 ? `已选 1 点，截面垂直于局部中心线（${near.segment!==null?branchName(near.segment):'—'}）；再点一个点可让截面通过两点，或按 Esc 结束点选后直接拖动截面。`
                              : `已选 2 点，截面通过两点（两点间距 ${pickPlane.chord_mm.toFixed(1)} mm）；点选已结束，可直接拖动蓝色截面微调。`;
    $('slice-basis').value='pick';$('slice-position').value='50';$('slice-pitch').value='0';$('slice-yaw').value='0';$('slice-offset-u').value='0';$('slice-offset-v').value='0';
    $('volume-mode').value='slice';sliceSelected=true;drawMarkers();
    if(picks.length===2)setPickMode(false);else refresh();
  }
  function clearPicks() {picks=[];pickPlane=null;pickInfo='';if($('slice-basis').value==='pick')$('slice-basis').value=groups.length?'centerline':'z';drawMarkers();refresh();}
  function setPickMode(on) {
    pickMode=Boolean(on);
    for(const id of ['pick-toggle','pick-toggle-2','pick-toggle-menu']){const button=$(id);if(!button)continue;if(button.classList)button.classList.toggle('on',pickMode);if(button.setAttribute)button.setAttribute('aria-pressed',String(pickMode));button.textContent=pickMode?'结束点选':(compactOn&&id==='pick-toggle'?'点选':'点选定位截面');}
    planeHover=false;
    refresh();
  }
  function currentPlane() {
    const basis=$('slice-basis').value,fraction=Number($('slice-position').value)/100;
    let result;
    if(basis==='pick'&&pickPlane) result={origin:add(pickPlane.origin,mul(pickPlane.normal,(fraction-0.5)*50)),normal:pickPlane.normal,segment:null};
    else if(basis==='centerline'&&groups.length) result=centerlinePlane(groups.find(g=>g.segment===Number($('slice-branch').value))||groups[0],fraction);
    else {const axis={x:0,y:1,z:2}[basis]??2, origin=center.slice(),normal=[0,0,0];normal[axis]=1;origin[axis]=bounds.min[axis]+fraction*(bounds.max[axis]-bounds.min[axis]);result={origin,normal,segment:null};}
    const plane={...result,...rotatePlane(result.normal,Number($('slice-pitch').value),Number($('slice-yaw').value))};
    const u=Number($('slice-offset-u').value)||0,v=Number($('slice-offset-v').value)||0;
    plane.origin=add(plane.origin,add(mul(plane.u,u),mul(plane.v,v)));return plane;
  }
  function setStats(id,stats,field,extra) {
    const container=$(id);if(!container)return;container.replaceChildren();
    const units=unitOf(field);
    const rows=[['预测点数',String(stats.count)],['均值',fmt(shown(stats.mean,field))+' '+units],['第 99 百分位',fmt(shown(stats.p99,field))+' '+units],['最大值',fmt(shown(stats.max,field))+' '+units]];
    for(const row of (Array.isArray(extra)?extra:[]))rows.push(row);
    for(const [label,value] of rows) {
      const kv=document.createElement('div');kv.className='kv';const span=document.createElement('span'),b=document.createElement('b');span.textContent=label;b.textContent=value;kv.append(span,b);container.append(kv);
    }
  }
  function cutSideFromValue(value) {
    return value==='cut-positive'?1:value==='cut-negative'?-1:null;
  }
  function applyCutPlanes(material,plane,side) {
    if(!material)return;
    if(!cutActive||side===null){material.clippingPlanes=[];return;}
    const n=new THREE.Vector3(...plane.normal), origin=new THREE.Vector3(...plane.origin);
    const signed=side>0?n:n.clone().negate();
    material.clippingPlanes=[new THREE.Plane(signed,-signed.dot(origin))];
  }
  function clearGroup(group) {if(!group)return;while(group.children.length){const object=group.children[0];group.remove(object);if(object.geometry)object.geometry.dispose();if(object.material)object.material.dispose();}}
  function clearContent() {clearGroup(content);}
  const TRUST_INTERIOR=4|8|16, TRUST_WALL_SOFT=2|4;
  // ``range`` is a display scale {min,max,log?,diverging?}; plain {min,max} ranges map linearly as before.
  function pointColor(i,values,range) {
    const c=scaleColor(values[i],range);
    return trustOverlay&&trust&&(trust[i]&TRUST_INTERIOR)?desaturate(c):c;
  }
  function geometryFor(indices,values,range) {
    const positions=new Float32Array(indices.length*3),colors=new Float32Array(indices.length*3);
    indices.forEach((i,j)=>{positions.set(point(pts,i),j*3);colors.set(pointColor(i,values,range),j*3);});
    const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.BufferAttribute(positions,3));geometry.setAttribute('color',new THREE.BufferAttribute(colors,3));return geometry;
  }
  function addVectors(indices,range) {
    if(!velocity||!indices.length)return;
    const positions=[],colors=[],stride=Math.max(1,Math.ceil(indices.length/600));
    for(let j=0;j<indices.length;j+=stride) {
      const i=indices[j],v=point(velocity,i),length=norm(v);if(length<1e-8)continue;
      const start=point(pts,i),direction=unit(v),size=diagonal*.025*Math.sqrt(length/Math.max(range.max,1e-9));
      const end=add(start,mul(direction,size)),side=planeBasis(direction).u,back=sub(end,mul(direction,size*.24)),c=scaleColor(length,range);
      for(const [a,b] of [[start,end],[end,add(back,mul(side,size*.1))],[end,sub(back,mul(side,size*.1))]]) {positions.push(...a,...b);colors.push(...c,...c);}
    }
    const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));geometry.setAttribute('color',new THREE.Float32BufferAttribute(colors,3));content.add(new THREE.LineSegments(geometry,new THREE.LineBasicMaterial({vertexColors:true,transparent:true,opacity:.9})));
  }
  // Streamlines are drawn as lit tubes (WebGL ignores line width), coloured by
  // local speed; thickness/density come from the menu.  Thin lines remain as a
  // low-cost fallback.
  function addStreamlines(range,plane,cutSide) {
    const stride=Math.max(1,Number($('streamline-density').value)||1);
    const width=Number($('streamline-width').value)||1, radius=diagonal/900*width;
    const tubes=!$('streamline-thin').checked;
    const material=tubes?new THREE.MeshPhongMaterial({vertexColors:true,shininess:35,specular:0x333333}):new THREE.LineBasicMaterial({vertexColors:true});
    applyCutPlanes(material,plane,cutSide);
    const hiddenLines=hiddenBranches.size?ensureLineSegment():null;
    for(let li=0;li<lines.length;li+=stride) {
      const line=lines[li],n=line.points.length/3;if(n<2)continue;
      if(hiddenLines&&hiddenLines[li]>=0&&hiddenBranches.has(hiddenLines[li]))continue;
      if(!tubes) {
        const geometry=new THREE.BufferGeometry(),colors=new Float32Array(line.points.length);
        for(let i=0;i<line.speed.length;i++)colors.set(scaleColor(line.speed[i],range),i*3);
        geometry.setAttribute('position',new THREE.BufferAttribute(line.points,3));geometry.setAttribute('color',new THREE.BufferAttribute(colors,3));content.add(new THREE.Line(geometry,material));continue;
      }
      const curvePoints=[];for(let i=0;i<n;i++)curvePoints.push(new THREE.Vector3(line.points[3*i],line.points[3*i+1],line.points[3*i+2]));
      const curve=new THREE.CatmullRomCurve3(curvePoints,false,'centripetal'), segs=Math.max(2,Math.min(n-1,160)), radial=5;
      const geometry=new THREE.TubeGeometry(curve,segs,radius,radial,false);
      const count=geometry.attributes.position.count,colors=new Float32Array(count*3);
      for(let i=0;i<=segs;i++){const c=scaleColor(line.speed[Math.round(i/segs*(n-1))],range);for(let j=0;j<=radial;j++)colors.set(c,3*(i*(radial+1)+j));}
      geometry.setAttribute('color',new THREE.BufferAttribute(colors,3));content.add(new THREE.Mesh(geometry,material));
    }
  }
  // ---- slice map: data (contour, points, bounds) + grid + a renderer shared by the side panel and the zoom view ----
  let sliceLast=null,sliceZoomOpen=false,sliceZoomLast=null,sliceSection=null,sliceSeries=null;
  let compactOn=false,compactOverride=null,sliceBodyUser=null;
  function sliceMapData(indices,values,plane,field,opts={}) {
    const isVelocityField=field==='velocity',fillOn=opts.fill!==undefined?Boolean(opts.fill):Boolean($('slice-fill')&&$('slice-fill').checked);
    const xy=indices.map(i=>{const p=sub(point(pts,i),plane.origin);return [dot(p,plane.u),dot(p,plane.v)];});
    let extent=1;for(const p of xy)extent=Math.max(extent,Math.abs(p[0]),Math.abs(p[1]));
    let finite=xy.map((p,j)=>({p,v:values[indices[j]],i:indices[j]})).filter(x=>Number.isFinite(x.v));
    let contour=[],loop=null,bounds=[-extent,extent,-extent,extent];
    if(fillOn&&vertices&&faces&&faces.length){
      // Every contour of the plane is chained first; a radius cut before chaining would turn far loops into
      // stray arcs. Candidate loops are then limited to a neighbourhood of the origin.
      const all=planeContour(vertices,faces,plane,isVelocityField?null:wallPressure,Infinity);
      loop=all.length?selectLoop(contourLoops(all),all,[0,0],Math.max(gizmoSide(plane.origin)*.5,8)):null;
      if(loop){
        contour=loop.segs;
        if(!loop.closed){const closed=closeChain(contour);if(closed){contour=closed;loop.synthetic=true;}else loop.open=true;}
        // only this cross-section's samples: points cut from a neighbouring vessel would stretch the map
        const kept=finite.filter(x=>pointInLoop(contour,x.p[0],x.p[1]));if(kept.length)finite=kept;
        let [minx,maxx,miny,maxy]=loop.bbox;for(const x of finite){minx=Math.min(minx,x.p[0]);maxx=Math.max(maxx,x.p[0]);miny=Math.min(miny,x.p[1]);maxy=Math.max(maxy,x.p[1]);}
        const cx=(minx+maxx)/2,cy=(miny+maxy)/2,half=Math.max(maxx-minx,maxy-miny,2)/2*1.08;bounds=[cx-half,cx+half,cy-half,cy+half];
      }
    }
    const nearest=[];for(let i=0;i<finite.length;i++){let best=Infinity;for(let j=0;j<finite.length;j++)if(i!==j)best=Math.min(best,Math.hypot(finite[i].p[0]-finite[j].p[0],finite[i].p[1]-finite[j].p[1]));if(Number.isFinite(best))nearest.push(best);}
    nearest.sort((a,b)=>a-b);const median=nearest.length?nearest[Math.floor(nearest.length/2)]:0;
    return {finite,contour,loop,bounds,median,fillOn,isVelocityField,extent,plane};
  }
  function sliceGrid(data,nx,ny) {
    const {finite,contour,bounds,median,fillOn,isVelocityField}=data;
    if(fillOn&&contour.length>=3&&!(data.loop&&data.loop.open)){
      const inside=scanlineInside(contour,bounds,nx,ny);
      const boundary=contour.map(sg=>[(sg[0]+sg[2])/2,(sg[1]+sg[3])/2,sg[5]<0?NaN:(isVelocityField?0:sg[4])]);
      // With no interior samples (a station beyond the sampled region) the median spacing is 0; a hash cell
      // of one grid step would then make the neighbour search walk ~n rings per cell, so scale it to the outline.
      const cell=median>0?median*1.5:Math.max((bounds[1]-bounds[0])/12,1e-6);
      const filled=fillSection(finite.map(x=>x.p),finite.map(x=>x.v),boundary,bounds,nx,ny,inside,{directRadius:median>0?median*3.2:Infinity,cell});
      return {values:filled.values,mask:inside,nx,ny,validCount:filled.filled,low:filled.low,stats:{inside:filled.filled,directCells:filled.directCells,wallSamples:boundary.filter(b=>Number.isFinite(b[2])).length}};
    }
    const g=interpolateIDW(finite.map(x=>x.p),finite.map(x=>x.v),{bounds,nx,ny,minNeighbors:Math.min(4,finite.length),maxDistance:median>0?median*3.2:0});
    g.stats=null;return g;
  }
  function fillSummary(stats,isVelocityField,language) {
    if(!stats)return null;const en=(language||lang)==='en',pct=stats.inside?Math.round(100*stats.directCells/stats.inside):0;
    return en?`${stats.inside} cells inside the wall contour · ${pct}% directly supported by samples · rest filled from the wall ${isVelocityField?'no-slip (0)':'pressure'} condition (faded)`
             :`壁面轮廓内 ${stats.inside} 格 · 邻点直接支撑 ${pct}% · 其余按壁面${isVelocityField?'无滑移（0）':'压力'}边界补全（淡色）`;
  }
  // Draws one slice map into ``ctx``; returns the mapping (for hover read-outs) and the footer text.
  function renderSliceMap(ctx,width,height,data,range,field,opts={}) {
    const {finite,contour,bounds,isVelocityField}=data,language=opts.lang||lang,en=language==='en';
    ctx.clearRect(0,0,width,height);ctx.fillStyle=opts.background||'#f6f8fb';ctx.fillRect(0,0,width,height);
    const pad=opts.pad||{left:20,top:15,right:20,bottom:72},plot={left:pad.left,top:pad.top,width:width-pad.left-pad.right,height:height-pad.top-pad.bottom};
    const [xmin,xmax,ymin,ymax]=bounds,scale=Math.min(plot.width/(xmax-xmin),plot.height/(ymax-ymin)),cx=plot.left+plot.width/2-(xmin+xmax)/2*scale,cy=plot.top+plot.height/2+(ymin+ymax)/2*scale;
    const toX=x=>cx+x*scale,toY=y=>cy-y*scale;
    ctx.strokeStyle='#d7e0e8';ctx.beginPath();ctx.moveTo(plot.left,toY(0));ctx.lineTo(plot.left+plot.width,toY(0));ctx.moveTo(toX(0),plot.top);ctx.lineTo(toX(0),plot.top+plot.height);ctx.stroke();
    const nx=Math.min(opts.gridMax||120,Math.max(48,Math.floor(plot.width/(opts.cellPx||4)))),ny=Math.min(opts.gridMax||120,Math.max(48,Math.floor(plot.height/(opts.cellPx||4))));
    const grid=sliceGrid(data,nx,ny);
    const cellW=(xmax-xmin)/nx*scale,cellH=(ymax-ymin)/ny*scale;
    for(let iy=0;iy<ny;iy++)for(let ix=0;ix<nx;ix++){
      const at=iy*nx+ix,x=toX(xmin+ix*(xmax-xmin)/nx),y=toY(ymin+(iy+1)*(ymax-ymin)/ny);
      if(grid.mask[at]&&Number.isFinite(grid.values[at])){const c=scaleColor(grid.values[at],range);ctx.fillStyle=`rgb(${c.map(v=>Math.round(v*255)).join(',')})`;if(grid.low&&grid.low[at])ctx.globalAlpha=.78;ctx.fillRect(x,y,cellW+1,cellH+1);ctx.globalAlpha=1;}
      else if(!grid.stats){ctx.fillStyle='#e1e7ec';ctx.globalAlpha=.38;ctx.fillRect(x,y,cellW+1,cellH+1);ctx.globalAlpha=1;}
    }
    if(contour.length){ctx.strokeStyle='#33475b';ctx.lineWidth=opts.lineWidth||1.4;ctx.beginPath();for(const sg of contour)if(sg[5]>=0){ctx.moveTo(toX(sg[0]),toY(sg[1]));ctx.lineTo(toX(sg[2]),toY(sg[3]));}ctx.stroke();
      if(ctx.setLineDash){ctx.setLineDash([6,4]);ctx.beginPath();for(const sg of contour)if(sg[5]<0){ctx.moveTo(toX(sg[0]),toY(sg[1]));ctx.lineTo(toX(sg[2]),toY(sg[3]));}ctx.stroke();ctx.setLineDash([]);}ctx.lineWidth=1;}
    if(opts.points!==false&&opts.values){const r=opts.pointRadius||2.7;for(const x of finite){const c=pointColor(x.i,opts.values,range);ctx.fillStyle=`rgb(${c.map(v=>Math.round(v*255)).join(',')})`;ctx.strokeStyle='#ffffff';ctx.lineWidth=1;ctx.beginPath();ctx.arc(toX(x.p[0]),toY(x.p[1]),r,0,Math.PI*2);ctx.fill();ctx.stroke();}}
    // In-plane flow arrows: the samples' velocity projected on (u, v), one per grid cell, length ∝ in-plane speed /
    // the section's p95 in-plane speed (capped at 6 % of the plot width); dark stroke over a white halo.
    let arrowCount=0;
    if(opts.arrows&&velocity&&data.plane&&finite.length) {
      const items=finite.map(x=>{const [du,dv]=inPlane(point(velocity,x.i),data.plane);return {x:x.p[0],y:x.p[1],du,dv};});
      const {arrows,ref}=arrowSamples(items,bounds,opts.maxArrows||120),maxLen=plot.width*.06,lw=opts.arrowWidth||1.4;
      if(ref>0) {
        const segsOf=a=>{const len=Math.min(1,a.mag/ref)*maxLen;if(len<1.5)return null;const ux=a.du/a.mag,uy=-a.dv/a.mag,x0=toX(a.x)-ux*len/2,y0=toY(a.y)-uy*len/2,x1=x0+ux*len,y1=y0+uy*len,h=Math.max(3,len*.35),c=Math.cos(.45),s=Math.sin(.45);
          return [[x0,y0,x1,y1],[x1,y1,x1-h*(ux*c-uy*s),y1-h*(uy*c+ux*s)],[x1,y1,x1-h*(ux*c+uy*s),y1-h*(uy*c-ux*s)]];};
        const all=arrows.map(segsOf).filter(Boolean);arrowCount=all.length;
        for(const [color2,width2] of [['#ffffff',lw+2],['#16283a',lw]]){ctx.strokeStyle=color2;ctx.lineWidth=width2;if(ctx.lineCap!==undefined)ctx.lineCap='round';ctx.beginPath();for(const segs of all)for(const [x0,y0,x1,y1] of segs){ctx.moveTo(x0,y0);ctx.lineTo(x1,y1);}ctx.stroke();}
        ctx.lineWidth=1;
      }
    }
    const font=opts.font||12;ctx.font=`${font}px sans-serif`;ctx.fillStyle='#33475b';ctx.strokeStyle='#33475b';
    {const L=scaleBarLength(1/scale,Math.min(120,plot.width*.22)),px=L*scale,x0=plot.left+8,y0=plot.top+plot.height-10;ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(x0,y0);ctx.lineTo(x0+px,y0);ctx.moveTo(x0,y0-4);ctx.lineTo(x0,y0+4);ctx.moveTo(x0+px,y0-4);ctx.lineTo(x0+px,y0+4);ctx.stroke();ctx.lineWidth=1;ctx.textAlign='left';ctx.fillText(`${L} mm`,x0,y0-6);}
    const barW=Math.max(120,Math.round(width*.26)),barH=Math.max(10,Math.round(font*.9)),barX=width-pad.right-barW,barY=height-pad.bottom+Math.round(font*2.6);
    if(opts.colorbar!==false) {
      for(let i=0;i<barW;i++){const c=colorAtT(i/Math.max(1,barW-1),range);ctx.fillStyle=`rgb(${c.map(v=>Math.round(v*255)).join(',')})`;ctx.fillRect(barX+i,barY,1,barH);}
      ctx.strokeStyle='#8093a2';ctx.strokeRect(barX,barY,barW,barH);ctx.fillStyle='#536a80';
      const [lo,hi]=scaleEnds(range);
      ctx.textAlign='left';ctx.fillText(fmt(shown(lo,field)),barX,barY-3);ctx.textAlign='right';ctx.fillText(fmt(shown(hi,field)),barX+barW,barY-3);
      if(range.log||range.diverging){ctx.textAlign='center';ctx.fillText(fmt(shown(scaleValueAt(.5,range),field)),barX+barW/2,barY-3);}
      ctx.textAlign='center';ctx.fillText(`${quantityLabel(field,opts.quantity,language)} · ${unitOf(field)}${range.log?(en?' · log':' · 对数'):''}`,barX+barW/2,barY+barH+font+2);ctx.textAlign='left';
    }
    const footer=(grid.stats?fillSummary(grid.stats,isVelocityField,language):(en?`local IDW · ${grid.validCount}/${nx*ny} cells supported`:`局部 IDW · 支持 ${grid.validCount}/${nx*ny} 格`))
      +(arrowCount?(en?' · arrows = in-plane flow direction (length normalised to this section)':' · 箭头 = 面内速度方向（长度按本截面归一化）'):'');
    if(opts.footer!==false){ctx.fillStyle='#536a80';ctx.fillText(footer,pad.left,height-pad.bottom+Math.round(font*1.4),Math.max(80,barX-pad.left-12));}
    if(!finite.length){ctx.textAlign='center';ctx.fillStyle='#536a80';ctx.fillText(en?'no valid samples within this thickness':'此厚度内没有有效预测点',width/2,toY(0)-20);ctx.textAlign='left';}
    return {plot,bounds,scale,cx,cy,grid,nx,ny,footer,arrows:arrowCount};
  }
  // The displayed quantity on a slice: speed / pressure, or the through-plane velocity (v · n, signed).
  function sliceValuesFor(field,quantity,plane,indices) {
    return quantity==='normal'&&velocity?throughPlane(velocity,indices,plane.normal):field==='velocity'?speed:pressure;
  }
  const arrowsOn=field=>field==='velocity'&&sliceArrows&&Boolean(velocity);
  function drawSlice(indices,plane,field) {
    const quantity=quantityOf(field),values=sliceValuesFor(field,quantity,plane,indices);
    const data=sliceMapData(indices,values,plane,field);
    // 本截面: robust range of this section's own samples (wall-boundary fills are never samples).
    const range=sliceScale(field,quantity,data.finite.map(x=>x.v));
    sliceScaleNow=range;
    sliceLast={data,range,field,plane,values,indices,quantity};computeSliceSection();
    const canvas=$('slice-canvas'),ctx=canvas&&canvas.getContext?canvas.getContext('2d'):null;if(!ctx)return;
    const map=renderSliceMap(ctx,canvas.width,canvas.height,data,range,field,{values,gridMax:120,footer:false,quantity,arrows:arrowsOn(field),maxArrows:120});
    sliceLast.arrows=map.arrows;
    setText('slice-fill-note',map.footer+(data.fillOn&&!data.loop?(lang==='en'?' · no wall contour found here (plane beyond the vessel?)':' · 此处未找到壁面轮廓（截面可能已越过血管末端），按严格模式绘制'):data.loop&&data.loop.synthetic?(lang==='en'?' · plane crosses an opening; outline closed by a straight edge':' · 截面经过血管开口，轮廓缺口以直线封闭'):data.loop&&data.loop.open?(lang==='en'?' · open outline, strict map':' · 轮廓不闭合，按严格模式绘制'):''));
    if(sliceZoomOpen)renderSliceZoom();
  }
  // Zoom view: a large, figure-quality map with hover read-out and PNG export.
  function renderSliceZoom() {
    const canvas=$('slice-zoom-canvas');if(!canvas||!sliceLast||!canvas.getContext)return;const ctx=canvas.getContext('2d');if(!ctx)return;
    const showPoints=!($('slice-zoom-points')&&$('slice-zoom-points').checked===false);
    const {data,range,field,plane,values,quantity}=sliceLast;
    const map=renderSliceMap(ctx,canvas.width,canvas.height,data,range,field,{values,points:showPoints,gridMax:240,cellPx:5,font:22,lineWidth:2.2,pointRadius:4.5,pad:{left:40,top:30,right:40,bottom:120},background:'#ffffff',
      quantity,arrows:arrowsOn(field),maxArrows:300,arrowWidth:2.4});
    sliceZoomLast=map;
    const en=lang==='en',thickness=Number($('slice-thickness').value);
    setText('slice-zoom-title',`${quantityLabel(field,quantity,lang)} · ${unitOf(field)}${meta.case_id?' · '+meta.case_id:''}`);
    setText('slice-zoom-caption',(en?`centre [${plane.origin.map(v=>v.toFixed(1)).join(', ')}] mm · normal [${plane.normal.map(v=>v.toFixed(3)).join(', ')}] · thickness ${thickness.toFixed(1)} mm`
                                   :`中心 [${plane.origin.map(v=>v.toFixed(1)).join(', ')}] mm · 法向 [${plane.normal.map(v=>v.toFixed(3)).join(', ')}] · 厚度 ${thickness.toFixed(1)} mm`)+sectionCaption(lang)
                                   +' · '+scaleCaption(field,range,quantity,en?'en':'zh'));
  }
  // ---- slice colour controls (panel + zoom view mirror each other; manual bounds are typed in display units) ----
  const displayFactor=field=>{const f=convertUnit(1,field,unitOf(field));return Number.isFinite(f)&&f!==0?f:1;};
  function writeSliceControls(field) {
    field=field||currentFieldKey();
    const isVel=field==='velocity'&&Boolean(velocity),manual=sliceRangeMode==='manual';
    for(const id of ['slice-range-mode','slice-zoom-range'])setVal(id,sliceRangeMode);
    for(const id of ['slice-quantity','slice-zoom-quantity']){const el=$(id);if(el){el.value=sliceQuantity;el.disabled=!isVel;}}
    for(const id of ['slice-arrows','slice-zoom-arrows']){const el=$(id);if(el){el.checked=sliceArrows;el.disabled=!isVel;}}
    for(const id of ['slice-quantity-row','slice-zoom-quantity-row','slice-arrows-row','slice-zoom-arrows-row'])setHidden(id,!isVel);
    for(const id of ['slice-manual','slice-zoom-manual'])setHidden(id,!manual);
    if(manual&&sliceManual.min!==null&&sliceManual.max!==null) {
      const k=displayFactor(field),active=typeof document!=='undefined'?document.activeElement:null;
      for(const [id,v] of [['slice-min',sliceManual.min],['slice-max',sliceManual.max],['slice-zoom-min',sliceManual.min],['slice-zoom-max',sliceManual.max]]){const el=$(id);if(el&&el!==active)el.value=String(+(Number(v)*k).toPrecision(4));}
    }
    const log=$('velocity-log');if(log){log.checked=logScale;log.disabled=!velocity;}
  }
  function setSliceDisplay(partial) {
    const p=partial&&typeof partial==='object'?partial:{},field=currentFieldKey();
    // quantity / arrows / log first: manual bounds are tagged with the quantity they are typed for
    if(p.quantity==='speed'||p.quantity==='normal')sliceQuantity=p.quantity;
    if(p.arrows!==undefined)sliceArrows=Boolean(p.arrows);
    if(p.log!==undefined)logScale=Boolean(p.log);
    if(RANGE_MODES.includes(p.mode)){
      sliceRangeMode=p.mode;
      // entering 手动 starts from what is on screen, so the two boxes are never empty
      if(p.mode==='manual'&&(sliceManual.min===null||sliceManual.max===null)&&sliceScaleNow){const [lo,hi]=scaleEnds(sliceScaleNow);sliceManual={min:lo,max:hi,field,quantity:quantityOf(field)};}
    }
    if(p.min!==undefined||p.max!==undefined){
      const k=displayFactor(field),lo=p.min===undefined?sliceManual.min:Number(p.min)/k,hi=p.max===undefined?sliceManual.max:Number(p.max)/k;
      sliceManual={min:Number.isFinite(lo)?lo:null,max:Number.isFinite(hi)?hi:null,field,quantity:quantityOf(field)};
    }
    refresh();
    return {mode:sliceRangeMode,quantity:sliceQuantity,arrows:sliceArrows,log:logScale,scale:sliceScaleNow&&{...sliceScaleNow}};
  }
  function openSliceZoom(open) {
    sliceZoomOpen=Boolean(open);setHidden('slice-zoom',!sliceZoomOpen);
    if(sliceZoomOpen){const f=$('slice-zoom-fill');if(f&&$('slice-fill'))f.checked=$('slice-fill').checked;setText('slice-zoom-status','');renderSliceZoom();}
  }
  // §15.14 tail: when the plane cuts a closed lumen outline, report its real geometry next to the
  // point statistics.  Synthetic closures (the plane crosses an opening) are flagged, open chains give nothing.
  function computeSliceSection() {
    sliceSection=null;
    const c=common();
    if(!c||typeof c.loopPolygon!=='function'||typeof c.sectionMetrics!=='function')return null;
    const data=sliceLast&&sliceLast.data;
    if(!data||!data.loop||data.loop.open||!Array.isArray(data.contour)||data.contour.length<3)return null;
    try {
      const poly=c.loopPolygon(data.contour);
      const m=poly&&poly.length>=3?c.sectionMetrics(poly):null;
      if(m&&Number.isFinite(m.area_mm2)&&m.area_mm2>0)sliceSection={...m,synthetic:Boolean(data.loop.synthetic)};
    } catch(_){sliceSection=null;}
    return sliceSection;
  }
  function sectionRows() {
    const m=sliceSection;if(!m)return [];
    return [['轮廓面积',fmt(m.area_mm2)+' mm²'],['最大直径',fmt(m.max_diameter_mm)+' mm'],['等效直径',fmt(m.equivalent_diameter_mm)+' mm']];
  }
  function sectionCaption(language) {
    const m=sliceSection;if(!m)return '';
    const en=(language||lang)==='en';
    return (en?` · outline area ${fmt(m.area_mm2)} mm² · max diameter ${fmt(m.max_diameter_mm)} mm · equivalent diameter ${fmt(m.equivalent_diameter_mm)} mm`
              :` · 轮廓面积 ${fmt(m.area_mm2)} mm² · 最大直径 ${fmt(m.max_diameter_mm)} mm · 等效直径 ${fmt(m.equivalent_diameter_mm)} mm`)
         +(m.synthetic?(en?' (outline closed by a straight edge)':'（轮廓缺口以直线封闭）'):'');
  }
  function sliceZoomReadout(clientX,clientY) {
    const canvas=$('slice-zoom-canvas');if(!canvas||!sliceZoomLast||!canvas.getBoundingClientRect)return;
    const rect=canvas.getBoundingClientRect(),x=(clientX-rect.left)*canvas.width/Math.max(rect.width,1),y=(clientY-rect.top)*canvas.height/Math.max(rect.height,1);
    const m=sliceZoomLast,wx=(x-m.cx)/m.scale,wy=(m.cy-y)/m.scale,[xmin,xmax,ymin,ymax]=m.bounds;
    const ix=Math.floor((wx-xmin)/(xmax-xmin)*m.nx),iy=Math.floor((wy-ymin)/(ymax-ymin)*m.ny);
    if(ix<0||iy<0||ix>=m.nx||iy>=m.ny){setText('slice-zoom-readout','');return;}
    const at=iy*m.nx+ix,v=m.grid.mask[at]?m.grid.values[at]:NaN,field=sliceLast.field;
    setText('slice-zoom-readout',Number.isFinite(v)?`${lang==='en'?'in-plane':'面内坐标'} (${wx.toFixed(1)}, ${wy.toFixed(1)}) mm · ${fmtField(v,field)}${m.grid.low&&m.grid.low[at]?(lang==='en'?' · filled from the wall condition':' · 壁面边界补全值'):''}`:'');
  }
  // ---- probe card ----
  function renderProbe() {
    const body=$('probe-body');if(!body)return;
    if(!probe){setHidden('probe-card',true);return;}
    const rows=[];
    if(probe.kind==='interior') {
      const rec=probeRecord(probe.index,{pts,velocity,pressure,segments,s:sFromRoot,radius:pointRadius,distWall,trust});
      rows.push(['位置 (mm)',rec.position.map(v=>v.toFixed(1)).join(', ')]);
      if(rec.speed!==undefined)rows.push(['速度大小',fmtField(rec.speed,'velocity')],['速度分量',rec.velocity.map(v=>fmt(shown(v,'velocity'))).join(', ')]);
      if(rec.pressure!==undefined)rows.push(['相对压力',fmtField(rec.pressure,'pressure')]);
      if(rec.segment!==undefined)rows.push(['分支',branchName(rec.segment)]);
      if(rec.s!==undefined)rows.push(['弧长 s',rec.s.toFixed(1)+' mm']);
      if(rec.radius!==undefined)rows.push(['局部半径',rec.radius.toFixed(2)+' mm']);
      if(rec.distWall!==undefined)rows.push(['到壁距离',rec.distWall.toFixed(2)+' mm']);
      if(rec.trust)rows.push(['可信标记',trustLabels(meta.trust).filter(t=>rec.trust&t.bit).map(t=>t.label).join('、')||'—']);
    } else {
      const v=probe.index;rows.push(['壁面顶点 (mm)',point(vertices,v).map(x=>x.toFixed(1)).join(', ')]);
      if(wallPressure&&Number.isFinite(wallPressure[v]))rows.push(['壁面压力',fmtField(wallPressure[v],'pressure')]);else if(wallPressure)rows.push(['壁面压力','无插值支撑']);
      const near=nearestTangent(groups,point(vertices,v));if(near)rows.push(['最近分支',branchName(near.segment)],['弧长 s',near.arc.toFixed(1)+' mm']);
      if(wallTrust&&wallTrust[v])rows.push(['可信标记',trustLabels(meta.trust).filter(t=>wallTrust[v]&t.bit).map(t=>t.label).join('、')]);
    }
    body.replaceChildren();
    for(const [label,value] of rows){const kv=document.createElement('div');kv.className='kv';const span=document.createElement('span'),b=document.createElement('b');span.textContent=label;b.textContent=value;kv.append(span,b);body.append(kv);}
    setText('probe-note',probePinned?'已固定；点击空白处或「清除」解除；「记录」加入探针记录表。':'悬停读数；单击可固定；「记录」加入探针记录表。');
    setHidden('probe-card',false);
  }
  // ---- probe log (C11) ----
  function probeToRow() {
    if(!probe)return null;
    const id='P'+(probeLog.reduce((m,r)=>Math.max(m,Number(String(r.id||'').replace(/^P/,''))||0),0)+1);
    if(probe.kind==='interior') {
      const rec=probeRecord(probe.index,{pts,velocity,pressure,segments,s:sFromRoot,radius:pointRadius,distWall,trust}),values={};
      if(rec.speed!==undefined){values.speed_m_s=rec.speed;values.u=rec.velocity[0];values.v=rec.velocity[1];values.w=rec.velocity[2];}
      if(rec.pressure!==undefined)values.pressure_pa=rec.pressure;
      if(rec.distWall!==undefined)values.dist_to_wall_mm=rec.distWall;
      return {id,kind:'interior',index:probe.index,xyz_mm:rec.position.map(v=>+v.toFixed(3)),branch:rec.segment!==undefined?branchName(rec.segment):'',segment_id:rec.segment!==undefined?rec.segment:null,
        s_from_root_mm:rec.s!==undefined?+rec.s.toFixed(2):null,radius_mm:rec.radius!==undefined?+rec.radius.toFixed(3):null,values,created_at:new Date().toISOString()};
    }
    const v=probe.index,p=point(vertices,v),near=nearestTangent(groups,p),values={};
    if(wallPressure&&Number.isFinite(wallPressure[v]))values.wall_pressure_pa=wallPressure[v];
    return {id,kind:'wall',index:v,xyz_mm:p.map(x=>+x.toFixed(3)),branch:near?branchName(near.segment):'',segment_id:near?near.segment:null,s_from_root_mm:near?+near.arc.toFixed(2):null,radius_mm:null,values,created_at:new Date().toISOString()};
  }
  function summarizeValues(values) {
    values=values||{};const parts=[];
    if(values.speed_m_s!==undefined)parts.push(fmtField(Number(values.speed_m_s),'velocity'));
    if(values.pressure_pa!==undefined)parts.push(fmtField(Number(values.pressure_pa),'pressure'));
    if(values.wall_pressure_pa!==undefined)parts.push('壁 '+fmtField(Number(values.wall_pressure_pa),'pressure'));
    if(values.dist_to_wall_mm!==undefined)parts.push('壁距 '+Number(values.dist_to_wall_mm).toFixed(2)+' mm');
    return parts.join(' · ')||'—';
  }
  function renderProbeLog() {
    const body=$('probe-log-body');if(!body)return;body.replaceChildren();
    for(const row of probeLog) {
      const tr=document.createElement('tr');
      const cells=[row.id,(row.xyz_mm||[]).map(v=>Number(v).toFixed(1)).join(', '),row.branch||'—',row.s_from_root_mm===null||row.s_from_root_mm===undefined?'—':Number(row.s_from_root_mm).toFixed(1),
        row.radius_mm===null||row.radius_mm===undefined?'—':Number(row.radius_mm).toFixed(2),summarizeValues(row.values)];
      for(const c of cells){const td=document.createElement('td');td.textContent=c;tr.appendChild(td);}
      const td=document.createElement('td'),del=document.createElement('button');del.type='button';del.textContent='删';del.addEventListener('click',()=>{probeLog=probeLog.filter(r=>r!==row);renderProbeLog();});td.appendChild(del);tr.appendChild(td);
      body.appendChild(tr);
    }
    setText('probe-log-count',String(probeLog.length));
    setText('probe-log-note',probeLog.length?`${probeLog.length} 条记录；导出数值为预测原始单位（m/s、Pa、mm），随视图状态保存。`:'点选或悬停探针后点「记录」，把读数攒成表；可复制 TSV 或导出 CSV。');
  }
  function probeSerializer(kind) {const common=root.WssReportCommon;const fn=common&&(kind==='csv'?common.probeToCSV:common.probeToTSV);return typeof fn==='function'?fn:(kind==='csv'?probeToCSV:probeToTSV);}
  // ---- measurements (C7): picks on the wall or interior, values from the shared centreline helpers ----
  const MEASURE_MODES={distance:{prefix:'D',label:'距离',picks:2},arc:{prefix:'A',label:'弧长',picks:2},
    diameter:{prefix:'R',label:'管径',picks:1},segment:{prefix:'S',label:'分段',picks:2}};
  function measureLabel(m,language) {
    const c=common();
    if(c&&typeof c.measurementLabel==='function'){try{const t=c.measurementLabel(m,language||lang);if(typeof t==='string'&&t)return t;}catch(_){}}
    const mode=MEASURE_MODES[m.kind];
    return `${mode?mode.label:m.kind} ${Number(m.value_mm).toFixed(1)} mm${m.branch?'（'+branchLabel(m.branch,language)+'）':''}`;
  }
  function measureValue(kind,points) {
    const c=common();
    if(!c)return {error:'测量需要共享库 report_common.js。'};
    if(kind==='distance') {
      const pr=clGroups.length?c.projectToCenterline(clGroups,points[0]):null;
      return {value_mm:c.straightDistance(points[0],points[1]),branch:pr?pr.name:'',segment_id:pr?pr.segment_id:null};
    }
    if(!clGroups.length)return {error:'本报告没有可用中心线，只能测直线距离。'};
    if(kind==='arc') {
      const r=c.arcDistance(clGroups,points[0],points[1]);
      if(!r||!Number.isFinite(r.value_mm))return {error:'两点不在同一条中心线树上。'};
      return {value_mm:r.value_mm,branch:r.a?r.a.name:'',segment_id:r.a?r.a.segment_id:null,path:r.path};
    }
    if(kind==='diameter') {
      const d=c.localDiameter(clGroups,points[0]);
      if(!d||!(Number(d.diameter_mm)>0))return {error:'该处中心线没有半径，无法给出管径。'};
      return {value_mm:d.diameter_mm,branch:d.name,segment_id:d.segment_id,dist_to_junction_mm:d.dist_to_junction_mm};
    }
    const a=c.projectToCenterline(clGroups,points[0]),b=c.projectToCenterline(clGroups,points[1]);
    if(!a||!b)return {error:'无法把选点投影到中心线。'};
    if(Number(a.segment_id)!==Number(b.segment_id))return {error:`分段长度要求两点落在同一分支（本次为 ${a.name} 与 ${b.name}）。`};
    return {value_mm:Math.abs(a.s-b.s),branch:a.name,segment_id:a.segment_id};
  }
  function addMeasurePick(p) {
    const mode=MEASURE_MODES[measureMode];
    if(!mode||!Array.isArray(p)||p.length!==3)return null;
    measurePicks=measurePicks.concat([p.map(Number)]);
    if(measurePicks.length<mode.picks){setText('measure-note',`${mode.label}：已选 ${measurePicks.length}/${mode.picks} 点。`);drawMarkers();return null;}
    const points=measurePicks.slice(0,mode.picks);measurePicks=[];
    const value=measureValue(measureMode,points);
    if(!value||value.error||!Number.isFinite(Number(value.value_mm))){setText('measure-note',(value&&value.error)||'无法计算该测量。');drawMarkers();return null;}
    const c=common();
    const item={id:c?c.newId(mode.prefix,measurements):mode.prefix+(measurements.length+1),kind:measureMode,
      points:points.map(q=>q.map(v=>+Number(v).toFixed(3))),value_mm:+Number(value.value_mm).toFixed(3),
      branch:value.branch||'',segment_id:value.segment_id===undefined?null:value.segment_id,created_at:new Date().toISOString()};
    if(value.dist_to_junction_mm!==undefined&&value.dist_to_junction_mm!==null)item.dist_to_junction_mm=+Number(value.dist_to_junction_mm).toFixed(2);
    if(Array.isArray(value.path))item.path=value.path.slice();
    item.label=measureLabel(item);
    measurements=measurements.concat([item]);
    setText('measure-note',`已记录 ${item.id}：${item.label}`);
    renderMeasurements();drawMarkers();
    return item;
  }
  function syncMeasureButtons() {
    for(const key of Object.keys(MEASURE_MODES)) {
      const b=$('measure-'+key);if(!b)continue;
      if(b.classList)b.classList.toggle('on',measureMode===key);
      if(b.setAttribute)b.setAttribute('aria-pressed',String(measureMode===key));
    }
  }
  function setMeasureMode(mode) {
    measureMode=(measureMode===mode||!MEASURE_MODES[mode])?null:mode;
    measurePicks=[];
    if(measureMode&&annotMode)setAnnotMode(false);
    syncMeasureButtons();
    setText('measure-note',measureMode
      ? `${MEASURE_MODES[measureMode].label}模式：在三维视图中点 ${MEASURE_MODES[measureMode].picks} 个点${renderer?'':'（三维视图不可用，无法点选）'}。`
      : (measurements.length?`共 ${measurements.length} 条测量；随视图状态、导图与六视角输出。`:'已退出测量模式。'));
    drawMarkers();
  }
  function measureTSV(language) {
    const head=['id','kind','value_mm','branch','x1','y1','z1','x2','y2','z2'].join('\t');
    return [head].concat(measurements.map(m=>{
      const a=(m.points||[])[0]||[],b=(m.points||[])[1]||[];
      return [m.id,m.kind,Number(m.value_mm).toFixed(3),branchLabel(m.branch,language),a[0],a[1],a[2],b[0],b[1],b[2]]
        .map(v=>v===undefined||v===null?'':String(v)).join('\t');
    })).join('\n');
  }
  function renderMeasurements() {
    const list=$('measure-list');if(!list)return;list.replaceChildren();
    for(const m of measurements) {
      const li=document.createElement('li');li.className='item';
      const text=document.createElement('span');text.textContent=`${m.id} · ${measureLabel(m)}`;
      const tools=document.createElement('span');tools.className='tools';
      const go=document.createElement('button');go.type='button';go.textContent='定位';
      go.addEventListener('click',()=>{const p=(m.points||[])[0];if(p)flyTo(p.map(Number),Math.max(Number(m.value_mm)||0,diagonal*0.03));});
      const del=document.createElement('button');del.type='button';del.textContent='删';
      del.addEventListener('click',()=>{measurements=measurements.filter(x=>x!==m);renderMeasurements();drawMarkers();});
      tools.append(go,del);li.append(text,tools);list.appendChild(li);
    }
  }
  // ---- annotations (C8): pinned text labels; server copy when online, embedded copy offline ----
  function normalizeAnnotations(list) {
    return (Array.isArray(list)?list:[])
      .filter(a=>a&&typeof a==='object'&&Array.isArray(a.xyz_mm)&&a.xyz_mm.length===3&&a.xyz_mm.every(v=>Number.isFinite(Number(v))))
      .slice(0,50)
      .map(a=>({...a,xyz_mm:a.xyz_mm.map(Number),text:String(a.text||'').slice(0,200),color:typeof a.color==='string'?a.color:'#d97706'}));
  }
  function setAnnotMode(on) {
    annotMode=Boolean(on);
    if(annotMode&&measureMode){measureMode=null;measurePicks=[];syncMeasureButtons();}
    const b=$('annot-add');
    if(b){if(b.classList)b.classList.toggle('on',annotMode);if(b.setAttribute)b.setAttribute('aria-pressed',String(annotMode));}
    if(annotMode)setText('annot-status','在三维视图中点击一处钉下标注（再点本按钮取消）。');
    drawMarkers();
  }
  function addAnnotationHere(p) {
    const input=$('annot-text');
    let text=String((input&&input.value)||'').trim();
    if(!text&&typeof root.prompt==='function'){try{text=String(root.prompt('标注文字（≤ 200 字）')||'').trim();}catch(_){text='';}}
    return addAnnotation(p,text);
  }
  function addAnnotation(p,text) {
    text=String(text||'').trim();
    if(!Array.isArray(p)||p.length!==3||!p.every(v=>Number.isFinite(Number(v)))){setText('annot-status','标注位置无效。');return null;}
    if(!text){setText('annot-status','标注文字不能为空：先在输入框写好文字，或在弹窗里输入。');return null;}
    if(annotLocked){setText('annot-status','已审阅锁定，标注只读。');return null;}
    if(annotations.length>=50){setText('annot-status','标注最多 50 条。');return null;}
    const c=common();
    const pr=c&&clGroups.length?c.projectToCenterline(clGroups,p.map(Number)):null;
    const near=nearestTangent(groups,p.map(Number));
    const np=nearestPoint(pts,interior,p.map(Number));
    let s=sFromRoot&&np.index>=0&&Number.isFinite(sFromRoot[np.index])?sFromRoot[np.index]:null;
    if(s===null&&c&&clGroups.length){const fr=c.arcFromRoot(clGroups,p.map(Number));if(fr&&Number.isFinite(fr.s_from_root_mm))s=fr.s_from_root_mm;}
    if(s===null&&near)s=near.arc;
    const item={id:c?c.newId('A',annotations):'A'+(annotations.length+1),xyz_mm:p.map(v=>+Number(v).toFixed(2)),
      text:text.slice(0,200),color:'#d97706',branch:pr?pr.name:(near?branchName(near.segment):''),
      segment_id:pr?pr.segment_id:(near?Number(near.segment):null),
      s_from_root_mm:Number.isFinite(Number(s))?+Number(s).toFixed(1):null,created_at:new Date().toISOString()};
    annotations=annotations.concat([item]);
    const input=$('annot-text');if(input)input.value='';
    renderAnnotations();drawMarkers();scheduleAnnotSave();
    setText('annot-status',`已新增标注 ${item.id}${online?'。':'（离线只读：只随视图状态保存）。'}`);
    return item;
  }
  function renderAnnotations() {
    const list=$('annot-list');if(!list)return;list.replaceChildren();
    for(const a of annotations) {
      const li=document.createElement('li');li.className='item';
      const box=document.createElement('span');
      const head=document.createElement('b');
      head.textContent=`${a.id} · ${branchLabel(a.branch)||'—'}${a.s_from_root_mm===null||a.s_from_root_mm===undefined?'':' · s '+Number(a.s_from_root_mm).toFixed(1)+' mm'}`;
      const edit=document.createElement('input');edit.type='text';edit.maxLength=200;edit.value=a.text||'';
      edit.addEventListener('change',()=>{a.text=String(edit.value||'').slice(0,200);renderLabels();scheduleAnnotSave();});
      box.append(head,edit);
      const tools=document.createElement('span');tools.className='tools';
      const go=document.createElement('button');go.type='button';go.textContent='定位';
      go.addEventListener('click',()=>flyTo(a.xyz_mm.map(Number),diagonal*0.03));
      const del=document.createElement('button');del.type='button';del.textContent='删';
      del.addEventListener('click',()=>{annotations=annotations.filter(x=>x!==a);renderAnnotations();drawMarkers();scheduleAnnotSave();});
      tools.append(go,del);li.append(box,tools);list.appendChild(li);
    }
    const count=annotations.length;
    if(!online)setText('annot-status',`离线只读（内嵌副本）：${count} 条标注只随视图状态保存。`);
  }
  function scheduleAnnotSave() {
    if(!online){renderLabels();setText('annot-status',`离线只读（内嵌副本）：${annotations.length} 条标注只随视图状态保存。`);return;}
    if(annotSaveTimer)clearTimeout(annotSaveTimer);
    annotSaveTimer=setTimeout(saveAnnotations,400);
    renderLabels();
  }
  async function saveAnnotations() {
    annotSaveTimer=null;
    if(!online)return;
    try {
      const out=await apiPut('annotations',{items:annotations});
      if(out&&out.annotations&&Array.isArray(out.annotations.items))annotations=normalizeAnnotations(out.annotations.items);
      annotLocked=false;renderAnnotations();renderLabels();
      setText('annot-status',`标注已保存到服务（${annotations.length} 条）。`);
    } catch(err) {
      annotLocked=Boolean(err&&err.status===409);
      setText('annot-status',(annotLocked?'已审阅锁定，标注只读：':'保存失败：')+(err&&err.message||err));
    }
  }
  async function loadAnnotations() {
    if(!online) {
      const doc=meta.annotations;
      if(doc&&Array.isArray(doc.items)&&!annotations.length)annotations=normalizeAnnotations(doc.items);
      renderAnnotations();drawMarkers();return;
    }
    try {
      const doc=await apiGetFile('annotations.json');
      if(doc&&Array.isArray(doc.items)){annotations=normalizeAnnotations(doc.items);renderAnnotations();drawMarkers();setText('annot-status',`已载入服务端标注 ${annotations.length} 条。`);}
      else setText('annot-status','本任务还没有标注；点「钉标注」后在视图中点选位置。');
    } catch(_){}
  }
  // ---- presets (C9): builtin generators from the shared library plus user presets in preferences ----
  function builtinPresetList() {
    const c=common();
    if(!c||typeof c.builtinPresets!=='function')return [];
    try{return c.builtinPresets('volume',meta)||[];}catch(_){return [];}
  }
  function presetContext() {
    const views={};
    if(frame)for(const name of Object.keys(STANDARD_VIEWS)){const cam=fittedCamera(name)||standardCamera(name,frame,center,diagonal*1.5);if(cam)views[name]=cam;}
    return {meta,branchNames:meta.branch_names||{},standardViews:views,findings:currentFindings(),
      profiles:meta.profiles||null,branchIds:branchIds.slice(),groups:clGroups};
  }
  function resolvePreset(name) {
    const hit=builtinPresetList().find(p=>p.name===name);
    if(hit){try{const state=hit.build(presetContext());return state&&typeof state==='object'?state:null;}catch(_){return null;}}
    const user=presetList.find(p=>p&&p.name===name);
    return user&&user.state&&typeof user.state==='object'?user.state:null;
  }
  // Partial preset states are merged onto the current state (nested objects merged one level).
  function mergeState(base,partial) {
    const out={...(base||{}),...(partial||{})};
    for(const key of ['slice','highlight','overlay','export','ui','streamlines','range','units_by_field']) {
      const a=base&&base[key],b=partial&&partial[key];
      if(a&&b&&typeof a==='object'&&typeof b==='object'&&!Array.isArray(a)&&!Array.isArray(b))out[key]={...a,...b};
    }
    return out;
  }
  function applyPresetByName(name) {
    const partial=resolvePreset(name);
    if(!partial){setText('preset-note',common()?`没有该预设：${name}`:'预设需要共享库 report_common.js。');return null;}
    // Set the status first: a preset that needs a more specific note (截面系列) overwrites it while applying.
    setText('preset-note',`已应用预设「${name}」。`);
    applyView(mergeState(captureView(),{...partial,preset_name:name}));
    return partial;
  }
  function loadLocalPresets() {
    try{const text=root.localStorage&&root.localStorage.getItem('wss-report-presets:volume');const doc=text?JSON.parse(text):null;
        return Array.isArray(doc)?doc:(doc&&Array.isArray(doc.items)?doc.items:[]);}catch(_){return [];}
  }
  function saveLocalPresets() {
    try{if(root.localStorage)root.localStorage.setItem('wss-report-presets:volume',JSON.stringify(presetList));}catch(_){}
  }
  async function loadServerPresets() {
    if(!online)return;
    try {
      const r=await fetch(online.root+'api/preferences',{credentials:'same-origin'});
      if(!r.ok)return;
      const prefs=(await r.json()).preferences||{};
      if(prefs.presets&&Array.isArray(prefs.presets.volume)){presetList=prefs.presets.volume.filter(p=>p&&typeof p==='object'&&typeof p.name==='string');renderPresets();}
    } catch(_){}
  }
  async function persistPresets() {
    saveLocalPresets();
    if(!online){setText('preset-note',`已保存到本浏览器（${presetList.length} 个用户预设）。`);return;}
    try {
      const r=await fetch(online.root+'api/preferences',{credentials:'same-origin'});
      const prefs=r.ok?((await r.json()).preferences||{}):{};
      prefs.schema_version=prefs.schema_version||'wss-deploy.preferences/v1';
      // Only the volume slot is replaced; every other preference section is written back untouched.
      prefs.presets=Object.assign({},prefs.presets||{},{volume:presetList});
      const token=await csrf();
      const put=await fetch(online.root+'api/preferences',{method:'PUT',credentials:'same-origin',
        headers:{'Content-Type':'application/json','X-CSRF-Token':token},body:JSON.stringify(prefs)});
      setText('preset-note',put.ok?`已保存到账户偏好（${presetList.length} 个用户预设）。`:'服务器保存失败，已保存到本浏览器。');
    } catch(_){setText('preset-note','服务器保存失败，已保存到本浏览器。');}
  }
  function savePreset(name) {
    name=String(name||'').trim().slice(0,40);
    if(!name){setText('preset-note','请先填写预设名称。');return null;}
    const state=captureView();state.preset_name=name;
    presetList=presetList.filter(p=>p&&p.name!==name).concat([{name,state,created_at:new Date().toISOString()}]);
    presetName=name;renderPresets();persistPresets();
    return name;
  }
  function renderPresets() {
    const builtin=$('preset-builtin');
    if(builtin&&builtin.replaceChildren) {
      builtin.replaceChildren();
      for(const p of builtinPresetList()) {
        const b=document.createElement('button');b.type='button';b.textContent=p.name;
        if(b.setAttribute)b.setAttribute('title',p.description||'');
        b.addEventListener('click',()=>applyPresetByName(p.name));
        builtin.appendChild(b);
      }
    }
    const list=$('preset-user');
    if(list&&list.replaceChildren) {
      list.replaceChildren();
      for(const p of presetList) {
        const li=document.createElement('li');li.className='item';
        const text=document.createElement('span');text.textContent=p.name+(presetName===p.name?' · 当前':'');
        const tools=document.createElement('span');tools.className='tools';
        const use=document.createElement('button');use.type='button';use.textContent='应用';
        use.addEventListener('click',()=>applyPresetByName(p.name));
        const ren=document.createElement('button');ren.type='button';ren.textContent='改名';
        ren.addEventListener('click',()=>{
          const input=$('preset-name');
          let next=String((input&&input.value)||'').trim();
          if(!next&&typeof root.prompt==='function'){try{next=String(root.prompt('新的预设名称',p.name)||'').trim();}catch(_){next='';}}
          if(!next){setText('preset-note','改名需要一个新名称（填写「名称」输入框或在弹窗输入）。');return;}
          p.name=next.slice(0,40);if(p.state)p.state.preset_name=p.name;
          renderPresets();persistPresets();
        });
        const del=document.createElement('button');del.type='button';del.textContent='删';
        del.addEventListener('click',()=>{presetList=presetList.filter(x=>x!==p);renderPresets();persistPresets();});
        tools.append(use,ren,del);li.append(text,tools);list.appendChild(li);
      }
    }
  }
  // ---- 3-D labels for measurements and annotations (drawn in the page and composited into exports) ----
  function projectPoint(p) {
    if(!camera||!view||typeof THREE==='undefined'||!THREE.Vector3)return null;
    try {
      const v=new THREE.Vector3(Number(p[0]),Number(p[1]),Number(p[2])).project(camera);
      if(!Number.isFinite(v.x)||!Number.isFinite(v.y)||v.z>1)return null;
      return {x:(v.x*0.5+0.5)*Math.max(view.clientWidth||0,1),y:(-v.y*0.5+0.5)*Math.max(view.clientHeight||0,1)};
    } catch(_){return null;}
  }
  // §17.3 chips: compact kind names that the shared English dictionary can translate verbatim.
  const FINDING_SHORT={high_wss_cluster:'高 WSS 区',low_wss_cluster:'低 WSS 区',max_wss:'全场最大 WSS',max_diameter:'最大直径',
    min_radius:'最小半径',max_speed:'最大速度',min_pressure:'最低压力',pressure_drop:'压降',low_speed_region:'低速区'};
  function findingChipText(item,language) {
    if(item.manual)return `${item.id} ${String(item.text||'').slice(0,40)}`.trim();
    const field=item.kind&&/speed/.test(item.kind)?'velocity':/pressure/.test(item.kind||'')?'pressure':null;
    const raw=Number(item.value);
    const value=!Number.isFinite(raw)?'':field?`${fmtTick(shown(raw,field))} ${unitOf(field)}`:`${fmtTick(raw)} ${item.units||''}`.trim();
    const name=labelText(FINDING_SHORT[item.kind]||item.label||item.kind||'',language);
    return `${item.id} ${name} ${value}`.replace(/\s+/g,' ').trim();
  }
  // A finding / branch chip is dropped when its branch is hidden (C10) so the overlay matches the scene.
  function branchHidden(sid){return sid!==null&&sid!==undefined&&Number.isFinite(Number(sid))&&hiddenBranches.has(Number(sid));}
  function labelItems(language) {
    const out=[];
    for(const m of measurements) {
      const points=(m.points||[]).filter(p=>Array.isArray(p)&&p.length===3).map(p=>p.map(Number));
      if(!points.length)continue;
      const anchor=points.length>1?mul(add(points[0],points[1]),0.5):points[0];
      out.push({kind:'meas',anchor,xyz:anchor,text:measureLabel(m,language)});
    }
    for(const a of annotations) {
      if(!Array.isArray(a.xyz_mm)||a.xyz_mm.length!==3)continue;
      const anchor=a.xyz_mm.map(Number);
      out.push({kind:'annot',anchor,xyz:add(anchor,[0,0,diagonal*0.05]),text:String(a.text||''),color:a.color||'#d97706'});
    }
    // §17.3 automatic findings labels: the first N of the list order, rejected items never labelled.
    const wanted=Math.max(0,Math.round(Number(labelState.findings)||0));
    if(wanted>0) {
      let pinned=0;
      for(const item of currentFindings()) {
        if(pinned>=wanted)break;
        if(reviewDecision(findingReview,item.id)==='rejected')continue;
        if(!Array.isArray(item.xyz_mm)||item.xyz_mm.length!==3)continue;
        if(branchHidden(item.segment_id))continue;
        const anchor=item.xyz_mm.map(Number);
        if(!anchor.every(Number.isFinite))continue;
        out.push({kind:'flabel',severity:item.manual?'manual':(item.severity||'note'),finding:item.id,
          anchor,xyz:add(anchor,[0,0,diagonal*0.04]),text:findingChipText(item,language)});
        pinned++;
      }
    }
    // Branch names at the mid-arc point of each centreline branch.
    if(labelState.branches) {
      for(const g of groups) {
        if(!g.points.length||branchHidden(g.segment))continue;
        let origin=null;
        try{origin=centerlinePlane(g,0.5).origin;}catch(_){origin=g.points[Math.floor(g.points.length/2)]||null;}
        if(!origin||!origin.every(Number.isFinite))continue;
        out.push({kind:'blabel',anchor:origin,xyz:origin,segment_id:Number(g.segment),text:branchLabel(branchName(g.segment),language)});
      }
    }
    // §17.3 max-diameter station label, pinned with the ring.
    if(labelState.max_diameter&&morphMax&&morphMax.xyz_mm) {
      out.push({kind:'dlabel',anchor:morphMax.xyz_mm,xyz:add(morphMax.xyz_mm,[0,0,diagonal*0.04]),
        text:`${labelText('最大直径',language)} ${fmtTick(Number(morphMax.max_diameter_mm))} mm`});
    }
    return out;
  }
  // ---- §21.4 page overlay: laid out once per camera / viewport change (the animation loop compares a camera key),
  // or when the label set changes (renderLabels marks it dirty).  Chips and leader lines are pooled divs.
  const labelClass=item=>item.kind==='flabel'?`flabel ${item.severity||'note'}`:item.kind;
  function labelsHideOverflow() {return compactOn||Math.max(view&&view.clientWidth||0,0)<900;}
  function declutterFn() {const c=common();return c&&typeof c.declutterLabels==='function'?c.declutterLabels:null;}
  function renderLabels() {
    labelsDirty=true;requestRender();
    if(!camera||!renderer){const box=$('labels');if(box&&box.replaceChildren)box.replaceChildren();labelPlan=[];}
  }
  function cameraKey() {
    if(!camera||!controls)return '';
    const q=v=>Number(v).toFixed(3);
    return [camera.position.x,camera.position.y,camera.position.z,controls.target.x,controls.target.y,controls.target.z,camera.up.x,camera.up.y,camera.up.z].map(q).join(',')
      +'|'+(view&&view.clientWidth)+'x'+(view&&view.clientHeight)+'|'+(labelsHideOverflow()?1:0);
  }
  // Size of a chip as the page draws it (measured once per class + text; estimated where there is no layout engine).
  function measureChip(box,item) {
    const key=labelClass(item)+'|'+item.text;
    if(labelSizeCache.has(key))return labelSizeCache.get(key);
    let w=0,h=0;
    try{const probe=document.createElement('div');probe.className=labelClass(item);probe.textContent=item.text;if(probe.style)probe.style.visibility='hidden';box.appendChild(probe);
        w=Number(probe.offsetWidth)||0;h=Number(probe.offsetHeight)||0;if(probe.remove)probe.remove();else if(box.removeChild)box.removeChild(probe);}catch(_){}
    const size={w:w>0?w:String(item.text).length*7+14,h:h>0?h:20};
    labelSizeCache.set(key,size);return size;
  }
  function layoutLabelsNow() {
    labelsDirty=false;labelKey=cameraKey();
    const box=$('labels');if(!box||!box.replaceChildren)return [];
    if(!camera||!renderer){box.replaceChildren();labelPlan=[];return [];}
    const items=labelItems(lang);
    const plan=planLabels(items,{project:projectPoint,size:item=>measureChip(box,item),declutter:declutterFn(),
      bounds:{width:Math.max(view.clientWidth||0,1),height:Math.max(view.clientHeight||0,1)},hideOverflow:labelsHideOverflow()});
    box.replaceChildren();
    // leaders first so the chips cover their inner ends
    for(const p of plan) {
      if(p.hidden||!p.moved)continue;
      const end=leaderEnd(p),dx=end.x-p.anchor.x,dy=end.y-p.anchor.y,len=Math.hypot(dx,dy);if(len<2)continue;
      const style=OVERLAY_STYLE[p.item.kind==='flabel'?'flabel_'+(p.item.severity||'note'):p.item.kind]||OVERLAY_STYLE.meas;
      const line=document.createElement('div');line.className='leader';
      if(line.style){line.style.left=p.anchor.x.toFixed(1)+'px';line.style.top=p.anchor.y.toFixed(1)+'px';line.style.width=len.toFixed(1)+'px';
        line.style.transform=`rotate(${Math.atan2(dy,dx).toFixed(4)}rad)`;line.style.background=style.leader;}
      box.appendChild(line);
    }
    for(const p of plan) {
      if(p.hidden)continue;
      const div=document.createElement('div');
      div.className=labelClass(p.item)+(p.moved?' moved':'');div.textContent=p.item.text;
      if(div.style){div.style.left=p.x.toFixed(1)+'px';div.style.top=p.y.toFixed(1)+'px';}
      // A findings chip is a shortcut into the list: clicking it flies to the finding, same as the row.
      if(p.item.finding&&div.addEventListener){const id=p.item.finding;div.addEventListener('click',()=>{
        const hit=currentFindings().find(x=>x.id===id);
        if(hit)activateFinding(activeFinding===hit.id?null:hit);
      });}
      box.appendChild(div);
    }
    labelPlan=plan;
    return plan;
  }
  // Chip palette shared by the page overlay (CSS) and the composited exports (canvas).
  const OVERLAY_STYLE={
    meas:{fill:'rgba(232,246,248,0.94)',border:'#5bbcc9',leader:'#3f8f9c'},
    annot:{fill:'rgba(255,247,230,0.94)',border:'#e0a84a',leader:'#b9791d'},
    // finding chips follow the page palette (v0.11.2 restyle: attention red, note amber, info blue)
    flabel_attention:{fill:'rgba(253,231,227,0.96)',border:'#e2a294',leader:'#c0392b',ink:'#8d3223'},
    flabel_note:{fill:'rgba(255,242,213,0.96)',border:'#e6c47a',leader:'#b7791f',ink:'#7a5814'},
    flabel_info:{fill:'rgba(229,240,250,0.96)',border:'#9dc3e6',leader:'#176caa',ink:'#175b8c'},
    flabel_manual:{fill:'rgba(243,236,250,0.96)',border:'#7d5ba6',leader:'#7d5ba6',ink:'#5b3f7d'},
    blabel:{fill:'rgba(247,251,249,0.96)',border:'#3f8f6b',leader:'#3f8f6b',ink:'#2c6b50'},
    dlabel:{fill:'rgba(246,238,250,0.96)',border:'#8e44ad',leader:'#8e44ad',ink:'#6d2f88'}};
  // Exports (PNG, six views, montages, one-pager figures) redo the same layout at the export's own label sizes
  // (text measured with the export font, divided by the scale) so the composited chips never overlap either.
  function drawOverlayLabels(ctx,W,H,k,language) {
    const items=labelItems(language);
    if(!items.length||!camera||!ctx||typeof ctx.fillText!=='function')return [];
    ctx.save();ctx.font=`${12*k}px Arial,Helvetica,sans-serif`;ctx.textAlign='left';ctx.textBaseline='middle';
    const pad=5,boxH=18;
    const size=item=>{let width=String(item.text).length*7*k;try{const m=ctx.measureText(item.text);if(m&&Number.isFinite(m.width)&&m.width>0)width=m.width;}catch(_){}return {w:width/k+2*pad,h:boxH};};
    const plan=planLabels(items,{project:projectPoint,size,declutter:declutterFn(),bounds:{width:W/k,height:H/k},hideOverflow:labelsHideOverflow()});
    for(const p of plan) {
      if(p.hidden)continue;
      const item=p.item,style=OVERLAY_STYLE[item.kind==='flabel'?'flabel_'+(item.severity||'note'):item.kind]||OVERLAY_STYLE.meas;
      const x=p.x*k,y=p.y*k,boxW=p.w*k,bh=p.h*k;
      // leader: a moved chip always gets one; a lifted chip (annotation / finding) keeps its short stem
      const lifted=Math.abs(p.anchor.x-p.x)>0.5||Math.abs(p.anchor.y-(p.y+p.h/2))>0.5;
      if(p.moved||lifted) {
        const end=leaderEnd(p);
        ctx.save();if(p.moved)ctx.globalAlpha=.6;ctx.strokeStyle=style.leader;ctx.lineWidth=Math.max(1,k*(p.moved?1:0.8));
        ctx.beginPath();ctx.moveTo(p.anchor.x*k,p.anchor.y*k);ctx.lineTo(end.x*k,end.y*k);ctx.stroke();ctx.restore();
      }
      ctx.fillStyle=style.fill;ctx.fillRect(x-boxW/2,y-bh/2,boxW,bh);
      ctx.strokeStyle=style.border;ctx.lineWidth=Math.max(1,k*0.8);ctx.strokeRect(x-boxW/2,y-bh/2,boxW,bh);
      ctx.fillStyle=style.ink||'#20374d';ctx.fillText(item.text,x-boxW/2+pad*k,y);
    }
    ctx.restore();
    return plan;
  }
  // ---- §17.3 automatic-label state, the max-diameter row and the §17.2 narrative ----
  function hasAutoLabels(){return labelState.findings>0||labelState.branches||Boolean(labelState.max_diameter&&morphMax);}
  function writeLabelControls() {
    const select=$('labels-findings');
    if(select) {
      const value=String(labelState.findings);
      // Tolerant apply: a count the menu does not offer gets its own option instead of being dropped.
      if(select.querySelector&&!select.querySelector(`option[value="${value}"]`))option(select,value,`前 ${value}`);
      select.value=value;
    }
    const branches=$('labels-branches');if(branches)branches.checked=Boolean(labelState.branches);
    const ring=$('labels-max-diameter');if(ring)ring.checked=Boolean(labelState.max_diameter);
  }
  function setLabels(partial) {
    const p=partial&&typeof partial==='object'?partial:{};
    if(p.findings!==undefined){const n=Number(p.findings);labelState.findings=Number.isFinite(n)?Math.max(0,Math.min(50,Math.round(n))):0;}
    if(p.branches!==undefined)labelState.branches=Boolean(p.branches);
    if(p.max_diameter!==undefined)labelState.max_diameter=Boolean(p.max_diameter);
    writeLabelControls();
    return {...labelState};
  }
  function renderMorphologyRow() {
    setHidden('profile-morphology',!morphMax);
    if(!morphMax)return;
    const d=Number(morphMax.max_diameter_mm),eq=Number(morphMax.equivalent_diameter_mm);
    const at=Number.isFinite(Number(morphMax.distance_from_inlet_mm))?Number(morphMax.distance_from_inlet_mm)
      :Number.isFinite(Number(morphMax.s_from_root_mm))?Number(morphMax.s_from_root_mm):null;
    setText('max-diameter-text',`最大直径 ${fmtTick(d)} mm${Number.isFinite(eq)?`（等效 ${fmtTick(eq)} mm）`:''}`
      +`${at===null?'':` · 入口下 ${fmtTick(at)} mm`}`);
    const fly=$('max-diameter-fly');if(fly)fly.disabled=!morphMax.xyz_mm;
    const ring=$('labels-max-diameter');if(ring)ring.disabled=!morphMax.polygon_world;
    // The profiles menu also opens for a case that only carries morphology (no along-branch curves).
    if(!profileBranches.length)setHidden('menu-profiles',false);
  }
  let narrativeLang=null,narrativeFilled=false;
  function renderNarrative() {
    const card=$('narrative-card');if(!card)return;
    const doc=meta.narrative&&typeof meta.narrative==='object'?meta.narrative:null;
    const edited=doc&&typeof doc.edited==='string'&&doc.edited.trim()?doc.edited.trim():null;
    const auto=doc?(lang==='en'&&Array.isArray(doc.en)&&doc.en.length?doc.en:Array.isArray(doc.zh)?doc.zh:[]):[];
    const lines=edited?String(edited).split(/\n+/).map(x=>x.trim()).filter(Boolean):auto.filter(x=>typeof x==='string'&&x.trim());
    setHidden('narrative-card',!lines.length);
    setHidden('narrative-edited',!edited);
    if(!lines.length){narrativeLang=lang;return;}
    if(narrativeLang===lang&&narrativeFilled)return;
    narrativeLang=lang;narrativeFilled=true;
    const body=$('narrative-body');if(!body||!body.replaceChildren)return;
    body.replaceChildren();
    for(const line of lines){const p=document.createElement('p');p.textContent=line;body.appendChild(p);}
  }
  // ---- branch visibility (C10) ----
  const branchIds=(()=>{const ids=new Set();groups.forEach(g=>ids.add(Number(g.segment)));if(raw.has_segments)modules.forEach(m=>ids.add(Number(m.segment)));return Array.from(ids).filter(Number.isFinite).sort((a,b)=>a-b);})();
  function renderBranchVisibility() {
    const box=$('branch-visibility');if(!box)return;box.replaceChildren();
    for(const sid of branchIds) {
      const label=document.createElement('label');label.className='inline';
      const cb=document.createElement('input');cb.type='checkbox';cb.checked=!hiddenBranches.has(sid);
      cb.addEventListener('change',()=>{if(cb.checked)hiddenBranches.delete(sid);else hiddenBranches.add(sid);refresh();drawMarkers();});
      const name=document.createElement('span');name.textContent=branchName(sid);label.append(cb,name);box.appendChild(label);
    }
    setHidden('branch-visibility-row',!branchIds.length);
  }
  function visibleIndices(indices) {if(!hiddenBranches.size||!segments)return indices;return indices.filter(i=>!hiddenBranches.has(Number(segments[i])));}
  function ensureVertexSegment() {
    if(vertexSegment)return vertexSegment;
    const wallIdx=[];for(let i=0;i<walls.length;i++)if(walls[i])wallIdx.push(i);
    if(wallIdx.length&&segments) {
      const src=new Float32Array(wallIdx.length*3),labels=new Int32Array(wallIdx.length);
      wallIdx.forEach((i,j)=>{src.set(point(pts,i),j*3);labels[j]=segments[i];});
      vertexSegment=nearestLabels(src,labels,vertices);
    } else {
      const cxyz=decode('center'),cseg=decode('center_segment',Int32Array);
      vertexSegment=cxyz&&cxyz.length&&cseg?nearestLabels(cxyz,cseg,vertices):new Int32Array(vertices.length/3).fill(-1);
    }
    return vertexSegment;
  }
  let faceCache={key:'',faces};
  function visibleFaces() {
    if(!hiddenBranches.size)return faces;
    const key=Array.from(hiddenBranches).sort((a,b)=>a-b).join(',');
    if(faceCache.key!==key)faceCache={key,faces:filterFaces(faces,ensureVertexSegment(),hiddenBranches)};
    return faceCache.faces;
  }
  function ensureLineSegment() {
    if(lineSegment)return lineSegment;
    lineSegment=lines.map(l=>{const n=Math.floor(l.points.length/3);if(!n||!segments)return -1;const near=nearestPoint(pts,interior,point(l.points,Math.floor(n/2)));return near.index>=0?Number(segments[near.index]):-1;});
    return lineSegment;
  }
  // ---- findings (list + review decisions, C16) ----
  const findingItems=findingsSorted(meta.findings&&meta.findings.items);
  const stop=ev=>{if(ev&&ev.stopPropagation)ev.stopPropagation();};
  function currentFindings() {return findingsWithReview(findingItems,findingReview);}
  function renderFindings() {
    const list=$('findings-list');if(!list)return;
    list.replaceChildren();
    const items=currentFindings();
    for(const item of items) {
      const decision=reviewDecision(findingReview,item.id);
      const li=document.createElement('li');li.className='finding'+(activeFinding===item.id?' on':'')+(decision==='rejected'?' rejected':decision==='confirmed'?' confirmed':'');
      const badge=document.createElement('span');
      if(item.manual){badge.className='sev manual';badge.textContent='人工';}
      else{badge.className='sev '+(item.severity||'note');badge.textContent=SEVERITY_LABELS[item.severity]||item.severity||'';}
      const text=document.createElement('span');
      const field=item.kind&&/speed/.test(item.kind)?'velocity':/pressure/.test(item.kind||'')?'pressure':null;
      const value=!item.manual&&Number.isFinite(Number(item.value))?(field?fmtField(Number(item.value),field):`${fmt(Number(item.value))} ${item.units||''}`):'';
      text.textContent=item.manual?`${item.text||''} · ${item.branch||''}`.trim():`${item.label||FINDING_KINDS[item.kind]||item.kind} · ${item.branch||''} ${value}`.trim();
      li.title=item.definition||(item.manual?'人工新增的发现':'');
      const tools=document.createElement('div');tools.className='finding-review';
      for(const [value2,label] of [['confirmed','确认'],['rejected','驳回'],[null,'未判定']]) {
        const b=document.createElement('button');b.type='button';b.textContent=label;b.className='rv'+(decision===value2?' on':'');
        b.addEventListener('click',ev=>{stop(ev);setDecision(item.id,value2);});tools.appendChild(b);
      }
      const note=document.createElement('input');note.type='text';note.maxLength=500;note.placeholder='备注';note.className='rv-note';note.value=reviewNote(findingReview,item.id);
      note.addEventListener('click',stop);note.addEventListener('change',ev=>{stop(ev);setNote(item.id,note.value);});tools.appendChild(note);
      if(item.manual){const del=document.createElement('button');del.type='button';del.textContent='删除';del.addEventListener('click',ev=>{stop(ev);removeManualFinding(item.id);});tools.appendChild(del);}
      li.append(badge,text,tools);
      li.addEventListener('click',()=>activateFinding(activeFinding===item.id?null:item));
      list.appendChild(li);
    }
    setText('findings-note',items.length?'点击一项飞到该处并高亮；再点取消。确认 / 驳回 / 备注'+(online?'保存到服务器':'离线时只随视图状态保存')+'；驳回项排到末尾且不进一页纸正文。':'本报告没有发现列表。');
    setHidden('menu-findings',!items.length&&!findingItems.length);
    renderLabels();   // §17.3: a rejected finding loses its chip immediately
  }
  let reviewSaveTimer=null;
  function scheduleReviewSave() {
    if(!online){setText('findings-review-status','离线只读：判定只保存在视图状态（导出 JSON 或复现链接）。');return;}
    if(reviewSaveTimer)clearTimeout(reviewSaveTimer);reviewSaveTimer=setTimeout(saveFindingsReview,400);
  }
  async function saveFindingsReview() {
    reviewSaveTimer=null;
    try {
      const out=await apiPut('findings_review',{items:findingReview.items,added:findingReview.added});
      if(out&&out.findings_review){findingReview=normalizeReview(out.findings_review);renderFindings();}
      setText('findings-review-status','判定已保存到服务器。');
    } catch(err){setText('findings-review-status',(err&&err.status===409?'结果已审阅锁定，判定不能修改：':'保存失败：')+(err&&err.message||err));}
  }
  async function loadFindingsReview() {
    if(!online)return;
    try{const doc=await apiGetFile('findings_review.json');if(doc){findingReview=normalizeReview(doc);renderFindings();refresh();}}catch(_){}
  }
  function setDecision(id,decision) {
    const entry={...(findingReview.items[id]||{decision:null,note:''}),decision};
    if(entry.decision||entry.note)findingReview.items[id]=entry;else delete findingReview.items[id];
    renderFindings();scheduleReviewSave();
  }
  function setNote(id,note) {
    const entry={...(findingReview.items[id]||{decision:null,note:''}),note:String(note||'').slice(0,500)};
    if(entry.decision||entry.note)findingReview.items[id]=entry;else delete findingReview.items[id];
    scheduleReviewSave();
  }
  function addManualFinding(p,text) {
    text=String(text||'').trim();if(!text){setText('findings-review-status','请先输入发现文字。');return null;}
    if(findingReview.added.length>=30){setText('findings-review-status','人工发现最多 30 条。');return null;}
    const ids=new Set(currentFindings().map(x=>x.id));let k=1;while(ids.has('M'+k))k++;
    const near=nearestTangent(groups,p),np=nearestPoint(pts,interior,p);
    const s=sFromRoot&&np.index>=0&&Number.isFinite(sFromRoot[np.index])?sFromRoot[np.index]:near?near.arc:null;
    const item={id:'M'+k,xyz_mm:p.map(v=>+Number(v).toFixed(2)),branch:near?branchName(near.segment):'',segment_id:near?near.segment:null,s_from_root_mm:s===null?null:+Number(s).toFixed(1),
      text:text.slice(0,500),kind:'manual',severity:'note',created_at:new Date().toISOString()};
    findingReview.added=findingReview.added.concat([item]);
    const input=$('finding-add-text');if(input)input.value='';
    setText('findings-review-status',`已新增人工发现 ${item.id}。`);renderFindings();refresh();scheduleReviewSave();return item;
  }
  function removeManualFinding(id) {
    findingReview.added=findingReview.added.filter(x=>x.id!==id);delete findingReview.items[id];
    if(activeFinding===id)activeFinding=null;renderFindings();refresh();scheduleReviewSave();
  }
  function activateFinding(item) {
    activeFinding=item?item.id:null;findingIndices=[];findingCenter=null;findingExtent=0;
    if(item) {
      const xyz=Array.isArray(item.xyz_mm)&&item.xyz_mm.length===3?item.xyz_mm.map(Number):null;
      findingCenter=xyz;findingExtent=Number(item.extent_mm)||diagonal*.03;
      if(Array.isArray(item.point_indices)&&item.point_indices.length)findingIndices=item.point_indices.map(Number).filter(i=>i>=0&&i<pts.length/3);
      else if(xyz)findingIndices=sphereIndices(pts,interior,xyz,findingExtent);
      if(item.kind==='pressure_drop'&&item.segment_id!==undefined&&groups.some(g=>Number(g.segment)===Number(item.segment_id))) {
        $('slice-basis').value='centerline';$('slice-branch').value=String(item.segment_id);$('slice-position').value='10';$('slice-pitch').value='0';$('slice-yaw').value='0';$('slice-offset-u').value='0';$('slice-offset-v').value='0';
        $('volume-mode').value='slice';sliceSelected=true;
        setText('findings-hint',`压降定义：${item.branch||branchName(item.segment_id)} 近端 10% 与远端 10% 弧长段的平均压差。截面已放在近端 10%；把位置滑到 90% 查看远端。`);
      } else setText('findings-hint',item.definition||'');
      if(xyz)flyTo(xyz,findingExtent);
    } else setText('findings-hint','');
    renderFindings();refresh();
  }
  // ---- along-branch profiles ----
  const profileBranches=(meta.profiles&&Array.isArray(meta.profiles.branches)?meta.profiles.branches:[]).filter(b=>b&&b.volume&&Array.isArray(b.s_local_mm));
  const profileSelect=$('profile-branch');
  profileBranches.forEach((b,i)=>option(profileSelect,i,b.name||branchName(b.segment_id)));
  {
    const first=profileBranches[0]&&profileBranches[0].volume||{};
    if(first.speed_mean_m_s)option($('profile-quantity'),'speed','截面平均 / 最大速度');
    if(first.pressure_mean_pa)option($('profile-quantity'),'pressure','截面平均 / 最低压力');
  }
  setHidden('menu-profiles',!profileBranches.length);
  function currentProfile() {return profileBranches[Number($('profile-branch').value)||0]||null;}
  function profileGroup(branch) {return branch?groups.find(g=>Number(g.segment)===Number(branch.segment_id))||null:null;}
  // Morphology stations are indexed by arc length from the inlet, the profile plot by the branch's own
  // s_local axis; shift by the branch offset so both curves share one x axis.
  function morphSeriesFor(branch) {
    if(!branch)return null;
    const st=morphStations[String(branch.segment_id)];
    if(!st||!(st.max||st.equiv))return null;
    const root=Array.isArray(branch.s_from_root_mm)?branch.s_from_root_mm.map(Number):null;
    const local=Array.isArray(branch.s_local_mm)?branch.s_local_mm.map(Number):null;
    const off=root&&local&&root.length&&local.length&&Number.isFinite(root[0])&&Number.isFinite(local[0])?root[0]-local[0]:0;
    return {x:st.s.map(v=>v-off),xRoot:st.s.slice(),max:st.max,equiv:st.equiv,offset_mm:off};
  }
  let profileLayout=null;
  function drawProfile() {
    const canvas=$('profile-canvas');if(!canvas)return;const ctx=canvas.getContext('2d');if(!ctx)return;
    const branch=currentProfile();const width=canvas.width,height=canvas.height;
    ctx.clearRect(0,0,width,height);ctx.fillStyle='#ffffff';ctx.fillRect(0,0,width,height);profileLayout=null;
    if(!branch)return;
    const quantity=$('profile-quantity').value||'speed', vol=branch.volume||{};
    const s=branch.s_local_mm.map(Number), field=quantity==='pressure'?'pressure':'velocity';
    const main=quantity==='pressure'?[['平均',vol.pressure_mean_pa,false],['最低',vol.pressure_min_pa,true]]:[['平均',vol.speed_mean_m_s,false],['最大',vol.speed_max_m_s,true]];
    const series=main.filter(x=>Array.isArray(x[1])).map(([label,values,dashed])=>({label,dashed,values:values.map(v=>Number.isFinite(Number(v))&&v!==null?shown(Number(v),field):NaN)}));
    const radius=Array.isArray(branch.radius_mm)?branch.radius_mm.map(v=>v===null?NaN:Number(v)):null;
    const sMin=0,sMax=Math.max(...s.filter(Number.isFinite),1);
    const top={left:44,top:14,width:width-56,height:Math.round(height*.58)-22}, bottom={left:44,top:Math.round(height*.58)+12,width:width-56,height:height-Math.round(height*.58)-40};
    function range(arrays){let lo=Infinity,hi=-Infinity;for(const a of arrays)for(const v of a)if(Number.isFinite(v)){lo=Math.min(lo,v);hi=Math.max(hi,v);}if(!Number.isFinite(lo)){lo=0;hi=1;}if(hi<=lo)hi=lo+1e-6;return {lo,hi};}
    function panel(box,arrays,labelText) {
      const r=range(arrays);ctx.strokeStyle='#d7e0e8';ctx.strokeRect(box.left,box.top,box.width,box.height);
      ctx.fillStyle='#536a80';ctx.font='11px sans-serif';ctx.textAlign='left';ctx.fillText(labelText,box.left+4,box.top+12);
      ctx.textAlign='right';ctx.fillText(fmt(r.hi),box.left-4,box.top+10);ctx.fillText(fmt(r.lo),box.left-4,box.top+box.height);ctx.textAlign='left';
      return r;
    }
    function polyline(box,r,values,dashed,colorText,xs) {
      const X=xs||s;
      ctx.strokeStyle=colorText;ctx.lineWidth=1.6;ctx.setLineDash(dashed?[4,3]:[]);ctx.beginPath();let pen=false;
      for(let i=0;i<values.length;i++){const v=values[i];if(!Number.isFinite(v)||!Number.isFinite(X[i])){pen=false;continue;}const x=box.left+(X[i]-sMin)/(sMax-sMin)*box.width,y=box.top+box.height-(v-r.lo)/(r.hi-r.lo)*box.height;if(pen)ctx.lineTo(x,y);else ctx.moveTo(x,y);pen=true;}
      ctx.stroke();ctx.setLineDash([]);
    }
    const dim=hiddenBranches.has(Number(branch.segment_id));
    const r1=panel(top,series.map(x=>x.values),`${quantity==='pressure'?'压力':'速度'} · ${unitOf(field)}${dim?'（分支已隐藏）':''}`);
    series.forEach((x,i)=>polyline(top,r1,x.values,x.dashed,dim?'#b0bcc6':i?'#c0392b':'#176caa'));
    // §17.3: the radius sub-plot also carries the morphology station diameters when the case has them.
    const morph=morphSeriesFor(branch);
    const bottomArrays=[];if(radius)bottomArrays.push(radius);
    if(morph&&morph.max)bottomArrays.push(morph.max);
    if(morph&&morph.equiv)bottomArrays.push(morph.equiv);
    let r2=null;
    if(bottomArrays.length) {
      r2=panel(bottom,bottomArrays,morph?'半径 · 直径（最大 实线 / 等效 虚线）· mm':'半径 · mm');
      if(radius)polyline(bottom,r2,radius,false,dim?'#b0bcc6':'#3f8f6b');
      if(morph&&morph.max)polyline(bottom,r2,morph.max,false,dim?'#c4c9cf':'#8e44ad',morph.x);
      if(morph&&morph.equiv)polyline(bottom,r2,morph.equiv,true,dim?'#c4c9cf':'#b07cc6',morph.x);
    }
    ctx.fillStyle='#536a80';ctx.font='11px sans-serif';ctx.textAlign='center';ctx.fillText(`弧长 s（mm）· 0 → ${sMax.toFixed(0)}`,width/2,height-8);ctx.textAlign='left';
    // marker for the current centreline slice on this branch
    const g=profileGroup(branch);
    if(g&&$('slice-basis').value==='centerline'&&Number($('slice-branch').value)===Number(branch.segment_id)) {
      const sNow=Number($('slice-position').value)/100*g.arc[g.arc.length-1];const x=top.left+(sNow-sMin)/(sMax-sMin)*top.width;
      ctx.strokeStyle='#a3324d';ctx.lineWidth=1.2;ctx.beginPath();ctx.moveTo(x,top.top);ctx.lineTo(x,bottom.top+bottom.height);ctx.stroke();
    }
    profileLayout={left:top.left,width:top.width,sMin,sMax,branch};
  }
  function sliceAtArc(branch,sValue) {
    const g=profileGroup(branch);if(!g)return false;
    $('slice-basis').value='centerline';$('slice-branch').value=String(branch.segment_id);$('slice-position').value=String((fractionForArc(g,sValue)*100).toFixed(1));
    $('slice-pitch').value='0';$('slice-yaw').value='0';$('slice-offset-u').value='0';$('slice-offset-v').value='0';$('volume-mode').value='slice';sliceSelected=true;refresh();return true;
  }
  // ---- region statistics (between two arc-length stations) ----
  function ensureArc() {if(!pointArc)pointArc=arcAlongBranch(groups,pts,segments,interior);return pointArc;}
  function updateRegion() {
    const container=$('region-stats');if(!container)return;
    const g=groups.find(x=>Number(x.segment)===num('region-branch',NaN));
    regionSel=[];
    if(!g||!interior.length){container.replaceChildren();setText('region-note','没有可用的中心线分支。');return;}
    const length=g.arc[g.arc.length-1], lo=Math.min(num('region-smin',0),num('region-smax',100))/100*length, hi=Math.max(num('region-smin',0),num('region-smax',100))/100*length;
    const arc=ensureArc();regionSel=regionIndices(interior,segments,arc,g.segment,lo,hi);
    const rows=[['体内点数',String(regionSel.length)],['弧长范围',`${lo.toFixed(1)} – ${hi.toFixed(1)} mm`]];
    if(speed){const st=statistics(speed,regionSel);rows.push(['速度均值',fmtField(st.mean,'velocity')],['速度最大',fmtField(st.max,'velocity')]);}
    if(pressure){const st=statistics(pressure,regionSel);rows.push(['压力均值',fmtField(st.mean,'pressure')],['压力最低',fmtField(st.min,'pressure')]);
      const drop=regionPressureDrop(regionSel,arc,pressure,lo,hi);rows.push(['近端−远端压差',drop.drop===null?'—':fmtField(drop.drop,'pressure')]);}
    container.replaceChildren();
    for(const [label,value] of rows){const span=document.createElement('span'),b=document.createElement('b');span.textContent=label;b.textContent=value;container.append(span,b);}
    setText('region-smin-value',num('region-smin',0).toFixed(0)+'%');setText('region-smax-value',num('region-smax',100).toFixed(0)+'%');
    setText('region-note','压差 = 区间近端 10% 与远端 10% 弧长段的点平均相对压力之差；统计按采样点等权。');
  }
  function drawHighlight() {
    requestRender();
    if(!highlightGroup)return;clearGroup(highlightGroup);
    const add3=(indices,hex,size)=>{if(!indices.length)return;const positions=new Float32Array(indices.length*3);indices.forEach((i,j)=>positions.set(point(pts,i),j*3));const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.BufferAttribute(positions,3));highlightGroup.add(new THREE.Points(geometry,new THREE.PointsMaterial({color:hex,size,sizeAttenuation:true,depthTest:false,transparent:true,opacity:.95})));};
    if(activeFinding){add3(findingIndices,0xff2fa8,diagonal/260);if(findingCenter){const s=new THREE.Mesh(new THREE.SphereGeometry(Math.max(findingExtent,diagonal/200),18,12),new THREE.MeshBasicMaterial({color:0xff2fa8,wireframe:true,transparent:true,opacity:.35,depthTest:false}));s.position.set(...findingCenter);highlightGroup.add(s);}}
    if(regionOn)add3(regionSel,0x1fa33a,diagonal/300);
  }
  function renderTrustLegend() {
    const list=$('trust-legend-list');if(!list)return;
    const available=Boolean(trust||wallTrust);setHidden('trust-row',!available);setHidden('trust-legend',!(available&&trustOverlay));
    if(!available)return;list.replaceChildren();
    for(const t of trustLabels(meta.trust)){const li=document.createElement('li');li.textContent=`${t.label}${t.fraction!==null?` · ${(t.fraction*100).toFixed(1)}%`:''}${t.rule?'：'+t.rule:''}`;if(TRUST_GLOSS[t.bit])li.appendChild(glossButton(TRUST_GLOSS[t.bit]));list.appendChild(li);}
  }
  // ---- view state (contract §5) ----
  function captureView() {
    const field=$('volume-field').value, range=fieldRanges[field]||{min:0,max:1};
    return {schema_version:'wss-deploy.view/v1',family:'volume',run_identity:meta.run_identity||null,camera:cameraState(),
      field,mode:$('volume-mode').value,colormap:currentMap,range:{mode:'case',min:range.min,max:range.max},bands:currentBands,log:logScale,
      units:unitOf(field),units_by_field:{pressure:pressureUnit,velocity:velocityUnit},pressure_units:pressureUnit,speed_units:velocityUnit,thresholds_pa:null,opacity:num('volume-opacity',.1),
      overlay:{trust:trustOverlay,contours:false},
      slice:{basis:$('slice-basis').value,branch:$('slice-branch').value,position:num('slice-position',50),pitch:num('slice-pitch',0),yaw:num('slice-yaw',0),thickness:num('slice-thickness',2),offset_u:num('slice-offset-u',0),offset_v:num('slice-offset-v',0),picks:picks.map(p=>p.slice()),cut:cutActive,module:$('volume-module').value,fill:!($('slice-fill')&&$('slice-fill').checked===false),series:sliceSeries?{segment_id:sliceSeries.segment_id,fractions:sliceSeries.fractions.slice()}:null,
        // v0.12 slice readability: colour range (min / max in raw units, used by 手动), coloured quantity, arrows
        color_range:{mode:sliceRangeMode,min:sliceManual.min,max:sliceManual.max},quantity:sliceQuantity,arrows:sliceArrows},
      highlight:{finding:activeFinding,region:regionOn?{branch:$('region-branch').value,smin:num('region-smin',0),smax:num('region-smax',100)}:null},
      streamlines:{width:num('streamline-width',1),density:$('streamline-density').value||'1',thin:Boolean($('streamline-thin').checked)},vectors:Boolean($('volume-vectors').checked),
      montage:{views:montageSelection(),columns:Math.max(2,Math.min(4,Math.round(num('montage-columns',2))))},
      // v1.1 (contract §12.2)
      measurements:measurements.map(m=>({...m})),annotations:{items:annotations.map(a=>({...a}))},
      probe_log:probeLog.map(r=>({...r,xyz_mm:(r.xyz_mm||[]).slice(),values:{...(r.values||{})}})),
      branches_hidden:Array.from(hiddenBranches).sort((a,b)=>a-b),preset_name:presetName,lang,export:{...exportOptions},
      // §17.3 automatic labels
      labels:{findings:labelState.findings,branches:labelState.branches,max_diameter:labelState.max_diameter},
      findings_review:{items:{...findingReview.items},added:findingReview.added.map(x=>({...x}))}};
  }
  function pickExport(e) {
    const out={};if(!e||typeof e!=='object')return out;
    if([1,2,4].includes(Number(e.scale)))out.scale=Number(e.scale);
    if(['white','transparent','current'].includes(e.background))out.background=e.background;
    if(['overlay','none','svg'].includes(e.colorbar))out.colorbar=e.colorbar;
    if(e.ui!==undefined)out.ui=Boolean(e.ui);
    return out;
  }
  function writeExportOptions() {setVal('export-scale',exportOptions.scale);setVal('export-background',exportOptions.background);setVal('export-colorbar',exportOptions.colorbar);const ui=$('export-hide-ui');if(ui)ui.checked=!exportOptions.ui;setVal('export-lang',lang);}
  function readExportOptions() {
    exportOptions={...exportOptions,...pickExport({scale:num('export-scale',exportOptions.scale),background:($('export-background')||{}).value,colorbar:($('export-colorbar')||{}).value,ui:!(($('export-hide-ui')||{}).checked)})};
    const l=($('export-lang')||{}).value;if(l==='en'||l==='zh')lang=l;
    return exportOptions;
  }
  function applyView(state) {
    if(!state||typeof state!=='object')return;
    if(state.field&&(state.field==='velocity'?speed:pressure))setVal('volume-field',state.field);
    if(state.colormap){setColormap(state.colormap);setVal('colormap',currentMap);}
    if(state.bands!==undefined){setBands(state.bands);setVal('color-bands',currentBands);}
    // ``log`` = velocity log colour scale (older states carry log:false; pressure ignores it).
    if(state.log!==undefined)logScale=Boolean(state.log);
    // §15.6: the compare page sends the display subset, where the two units are top-level keys.
    const ubf=state.units_by_field||{},pu=state.pressure_units||ubf.pressure,su=state.speed_units||ubf.velocity;
    if(UNITS.pressure[pu]){pressureUnit=pu;setVal('pressure-unit',pressureUnit);}if(UNITS.velocity[su]){velocityUnit=su;setVal('velocity-unit',velocityUnit);}
    if(Number.isFinite(Number(state.opacity)))setVal('volume-opacity',state.opacity);
    if(state.overlay){trustOverlay=Boolean(state.overlay.trust);const el=$('trust-overlay');if(el)el.checked=trustOverlay;}
    // A state without a ``slice`` key leaves the plane, the picks and the cut exactly as they are
    // (§15.6: the compare page may send only part of the display subset).
    const sl=state.slice&&typeof state.slice==='object'?state.slice:null;
    if(sl) {
      picks=Array.isArray(sl.picks)?sl.picks.filter(p=>Array.isArray(p)&&p.length===3).map(p=>p.map(Number)):[];
      if(picks.length){const near=nearestTangent(groups,picks[0])||{tangent:[0,0,1]};pickPlane=planeFromPicks(picks,near.tangent);pickPlane.segment=null;}else pickPlane=null;
      if(sl.basis&&!(sl.basis==='pick'&&!pickPlane))setVal('slice-basis',sl.basis);
      if(sl.fill!==undefined&&$('slice-fill'))$('slice-fill').checked=sl.fill!==false;
      for(const [key,id] of [['branch','slice-branch'],['position','slice-position'],['pitch','slice-pitch'],['yaw','slice-yaw'],['thickness','slice-thickness'],['offset_u','slice-offset-u'],['offset_v','slice-offset-v']]) if(sl[key]!==undefined)setVal(id,sl[key]);
      // C9「瘤囊截面系列」: an automatic series along one branch; the first station is shown, the rest via the slider.
      if(sl.kind==='auto') {
        const fractions=(Array.isArray(sl.fractions)?sl.fractions:[0.4,0.6,0.8]).map(Number).filter(f=>Number.isFinite(f)&&f>=0&&f<=1);
        const series=fractions.length?fractions:[0.4,0.6,0.8];
        if(groups.length) {
          setVal('slice-basis','centerline');
          if(sl.branch!==undefined&&groups.some(g=>Number(g.segment)===Number(sl.branch)))setVal('slice-branch',String(sl.branch));
          setVal('slice-position',String(+(series[0]*100).toFixed(1)));
          for(const id of ['slice-pitch','slice-yaw','slice-offset-u','slice-offset-v'])setVal(id,'0');
          setVal('volume-mode','slice');
        }
        setText('preset-note',`截面系列：${series.map(f=>Math.round(f*100)+'%').join(' / ')}；已放在第一处，用位置滑块查看其余。`);
      }
      // §15.10: a recorded station series is remembered (branch + count) but never moves the current plane.
      if(sl.series&&typeof sl.series==='object') {
        const fr=(Array.isArray(sl.series.fractions)?sl.series.fractions:[]).map(Number).filter(f=>Number.isFinite(f)&&f>=0&&f<=1);
        const sid=Number(sl.series.segment_id);
        sliceSeries={segment_id:Number.isFinite(sid)?sid:null,fractions:fr};
        if(Number.isFinite(sid)&&groups.some(g=>Number(g.segment)===sid))setVal('slice-series-branch',String(sid));
        if(fr.length)setVal('slice-series-count',String(fr.length));
      }
      // v0.12 slice colour keys.  A complete older state (schema_version, no color_range) was drawn on the whole-field
      // range without arrows, so a replay keeps that look; partial states (compare page subsets) change nothing.
      const legacy=Boolean(state.schema_version);
      const cr=sl.color_range&&typeof sl.color_range==='object'?sl.color_range:null;
      if(cr){if(RANGE_MODES.includes(cr.mode))sliceRangeMode=cr.mode;const lo=Number(cr.min),hi=Number(cr.max);
        if(cr.min!==null&&cr.max!==null&&cr.min!==undefined&&cr.max!==undefined&&Number.isFinite(lo)&&Number.isFinite(hi))sliceManual={min:lo,max:hi,field:$('volume-field').value,quantity:null};}
      else if(legacy)sliceRangeMode='global';
      if(sl.quantity==='speed'||sl.quantity==='normal')sliceQuantity=sl.quantity;else if(legacy)sliceQuantity='speed';
      if(sl.arrows!==undefined)sliceArrows=Boolean(sl.arrows);else if(legacy)sliceArrows=false;
      if(sliceManual.quantity===null)sliceManual.quantity=quantityOf($('volume-field').value);
      cutActive=Boolean(sl.cut);const moduleSelect=$('volume-module');
      if(cutActive&&moduleSelect.querySelector&&!moduleSelect.querySelector('option[value="cut-positive"]')){option(moduleSelect,'cut-positive','截面 A 侧');option(moduleSelect,'cut-negative','截面 B 侧');}
      if(sl.module!==undefined)setVal('volume-module',sl.module);
    }
    const st=state.streamlines||{};if(st.width!==undefined)setVal('streamline-width',st.width);if(st.density!==undefined)setVal('streamline-density',st.density);const thin=$('streamline-thin');if(thin&&st.thin!==undefined)thin.checked=Boolean(st.thin);
    const vec=$('volume-vectors');if(vec&&state.vectors!==undefined)vec.checked=Boolean(state.vectors);
    if(state.mode&&MODES.includes(state.mode))setVal('volume-mode',state.mode);sliceSelected=$('volume-mode').value==='slice';
    const hl=state.highlight||{};
    if(hl.region){regionOn=true;setVal('region-branch',hl.region.branch);setVal('region-smin',hl.region.smin);setVal('region-smax',hl.region.smax);const rh=$('region-highlight');if(rh)rh.checked=true;}
    if(Array.isArray(state.branches_hidden)){hiddenBranches=new Set(state.branches_hidden.map(Number).filter(Number.isFinite));renderBranchVisibility();}
    // §17.3: a state without ``labels`` leaves the current chips alone; partial objects are merged.
    if(state.labels&&typeof state.labels==='object')setLabels(state.labels);
    if(Array.isArray(state.probe_log)){probeLog=state.probe_log.filter(r=>r&&typeof r==='object'&&Array.isArray(r.xyz_mm)).map(r=>({...r,xyz_mm:r.xyz_mm.map(Number),values:r.values&&typeof r.values==='object'?{...r.values}:{}}));renderProbeLog();}
    if(Array.isArray(state.measurements)) {
      measurements=state.measurements.filter(m=>m&&typeof m==='object'&&Array.isArray(m.points)).map(m=>({...m,points:m.points.map(p=>Array.isArray(p)?p.map(Number):p)}));
      renderMeasurements();
    }
    if(state.annotations&&Array.isArray(state.annotations.items)) {
      // Online the server copy stays authoritative; offline the view state carries them.
      if(!online||!annotations.length)annotations=normalizeAnnotations(state.annotations.items);
      renderAnnotations();
    }
    if(state.preset_name!==undefined)presetName=typeof state.preset_name==='string'?state.preset_name:null;
    // Presets may ask for a menu to be open and a profile to be selected (contract §12.4).
    const ui=state.ui&&typeof state.ui==='object'?state.ui:null;
    if(ui) {
      if(typeof ui.menu==='string'){const card=$('menu-'+ui.menu);if(card)card.open=true;}
      const pf=ui.profile&&typeof ui.profile==='object'?ui.profile:null;
      if(pf&&profileBranches.length) {
        if(pf.branch!==undefined){const at=profileBranches.findIndex(b=>Number(b.segment_id)===Number(pf.branch));if(at>=0)setVal('profile-branch',String(at));}
        if(pf.field==='pressure'||pf.field==='speed') {
          const select=$('profile-quantity'),option=select&&select.querySelector?select.querySelector(`option[value="${pf.field}"]`):null;
          if(option)setVal('profile-quantity',pf.field);
        }
        const branch=profileBranches[Number(($('profile-branch')||{}).value)||0];
        if(branch&&Number.isFinite(Number(pf.s)))sliceAtArc(branch,Number(pf.s));
      }
    }
    if(state.lang==='en'||state.lang==='zh')lang=state.lang;
    if(state.export&&typeof state.export==='object')exportOptions={...exportOptions,...pickExport(state.export)};
    if(state.montage&&typeof state.montage==='object') {
      const views=Array.isArray(state.montage.views)?state.montage.views.filter(v=>MONTAGE_VIEWS.includes(v)):null;
      if(views)for(const name of MONTAGE_VIEWS){const box=$('mv-'+name);if(box&&!box.disabled)box.checked=views.includes(name);}
      const cols=Number(state.montage.columns);if([2,3,4].includes(cols))setVal('montage-columns',String(cols));
    }
    writeExportOptions();
    // Offline reports carry the decisions in the view state; online the server copy is authoritative.
    if(!online&&state.findings_review&&typeof state.findings_review==='object')findingReview=normalizeReview(state.findings_review);
    if(hl.finding){const item=currentFindings().find(x=>x.id===hl.finding);if(item){activeFinding=item.id;findingCenter=Array.isArray(item.xyz_mm)?item.xyz_mm.map(Number):null;findingExtent=Number(item.extent_mm)||diagonal*.03;findingIndices=Array.isArray(item.point_indices)&&item.point_indices.length?item.point_indices.map(Number):(findingCenter?sphereIndices(pts,interior,findingCenter,findingExtent):[]);}}
    drawMarkers();renderFindings();refresh();
    // An explicit camera (view state, #view= link, preset, camera link) always wins over the automatic framing.
    if(state.camera){setCamera(state.camera);autoView=null;}
  }
  const viewKey=()=>'wss-volume-view:'+(meta.run_identity||meta.case_id||'report');
  function viewFromHash() {
    try{const hash=root.location&&root.location.hash||'';const m=hash.match(/[#&]view=([A-Za-z0-9_-]+)/);return m?decodeView(m[1]):null;}catch(_){return null;}
  }
  function download(name,href) {if(!document.body)return;const a=document.createElement('a');a.download=name;a.href=href;document.body.appendChild(a);a.click();a.remove&&a.remove();}
  // ---- publication export (C12): off-screen render at 1x/2x/4x, white / transparent / current background ----
  // The scale on screen (the legend's): the slice's own scale on the 截面 page, the whole-field one elsewhere.
  function currentScale() {
    const field=currentFieldKey();
    return legendScaleNow||{...globalScale(field,field==='velocity'?'speed':'pressure'),quantity:field==='velocity'?'speed':'pressure'};
  }
  function drawColorbarOverlay(ctx,W,H,k,language) {
    const field=currentFieldKey(),range=currentScale();
    const pad=14*k,barW=18*k,barH=Math.min(220*k,H*0.4),x=W-pad-84*k,y=pad+22*k;
    ctx.save();ctx.fillStyle='rgba(255,255,255,0.86)';ctx.fillRect(x-8*k,pad-4*k,pad+92*k,barH+40*k);
    const n=currentBands>0?currentBands:64;
    for(let i=0;i<n;i++){const c=colorAtT((i+0.5)/n,range).map(v=>Math.round(v*255));ctx.fillStyle=`rgb(${c.join(',')})`;ctx.fillRect(x,y+barH-(i+1)*barH/n,barW,barH/n+1);}
    ctx.strokeStyle='#8093a2';ctx.lineWidth=Math.max(1,k*0.8);ctx.strokeRect(x,y,barW,barH);
    ctx.fillStyle='#20374d';ctx.font=`${12*k}px Arial,Helvetica,sans-serif`;ctx.textAlign='left';
    ctx.fillText(quantityLabel(field,range.quantity,language)+' ('+unitOf(field)+')'+(range.log?(language==='en'?' log':' 对数'):''),x-4*k,pad+13*k);
    const nt=currentBands>0?currentBands+1:5;
    for(let i=0;i<nt;i++){const tq=i/(nt-1);ctx.fillText(fmt(shown(scaleValueAt(tq,range),field)),x+barW+5*k,y+barH-tq*barH+4*k);}
    ctx.restore();
  }
  // SVG colour bar of a scale (default: the one on screen); ``note`` is appended to the title (e.g. 系列共用).
  function colorbarSvgCurrent(language,scaleOverride,note) {
    const field=currentFieldKey(),range=scaleOverride||currentScale(),language2=language||lang,[lo,hi]=scaleEnds(range),map=scaleMap(range);
    const title=quantityLabel(field,range.quantity,language2)+(range.log?(language2==='en'?' · log':' · 对数'):'')+(note?' · '+note:'');
    const opts={colormap:map,min:shown(lo,field),max:shown(hi,field),bands:currentBands,log:false,units:unitOf(field),title,lang:language2,width:96,height:260,orientation:'vertical'};
    const common=root.WssReportCommon;
    if(common&&typeof common.colorbarSVG==='function'){try{const svg=common.colorbarSVG({...opts,stops:(COLORMAPS[map]||COLORMAPS.rainbow).stops.map((c,i,a)=>[i/(a.length-1),'#'+c.map(v=>v.toString(16).padStart(2,'0')).join('')])});if(typeof svg==='string'&&svg.startsWith('<svg'))return formatSvgTicks(svg,range,field,opts.bands);}catch(_){}}
    return colorbarSVGLocal(opts);
  }
  // The shared colour bar writes its ticks with fixed decimals (0.00177 → "0.00") and knows nothing of our log
  // mapping; the ticks sit at even fractions (bands + 1, or 5 for a continuous bar) in document order, so they are
  // rewritten from the scale itself with formatValue (log ticks are geometric).
  function formatSvgTicks(svg,range,field,bands) {
    const n=bands>0?Math.floor(bands):4;let i=0;
    return svg.replace(/(<text class="tick"[^>]*>)([^<]*)(<\/text>)/g,(all,open,_old,close)=>open+formatNumber(shown(scaleValueAt(i++/n,range),field))+close);
  }
  function renderExport(options) {
    if(!renderer||!camera||!scene)return {error:'三维视图不可用，无法导出。'};
    const o={scale:1,background:'white',colorbar:'overlay',ui:true,lang,...(options&&typeof options==='object'?options:{})};
    const width=Math.max(view.clientWidth,1),height=Math.max(view.clientHeight,1);
    const ratio=typeof renderer.getPixelRatio==='function'?renderer.getPixelRatio():1;
    const prevColor=new THREE.Color();let prevAlpha=1;try{renderer.getClearColor(prevColor);prevAlpha=renderer.getClearAlpha();}catch(_){}
    const attempt=k=>{
      renderer.setPixelRatio(1);renderer.setSize(width*k,height*k,false);
      if(o.background==='transparent')renderer.setClearColor(0x000000,0);else if(o.background==='white')renderer.setClearColor(0xffffff,1);else renderer.setClearColor(prevColor,prevAlpha);
      camera.aspect=width/height;camera.updateProjectionMatrix();renderer.render(scene,camera);
      const gl=typeof renderer.getContext==='function'?renderer.getContext():null;
      if(gl&&gl.drawingBufferWidth&&gl.drawingBufferWidth<width*k-1)throw new Error('显卡不支持该分辨率');
      const canvas=document.createElement('canvas');canvas.width=width*k;canvas.height=height*k;
      const ctx=canvas.getContext('2d');if(!ctx)throw new Error('2D 画布不可用');
      ctx.drawImage(renderer.domElement,0,0,canvas.width,canvas.height);
      if(o.ui!==false){
        if(o.colorbar==='overlay')drawColorbarOverlay(ctx,canvas.width,canvas.height,k,o.lang);
        drawOverlayLabels(ctx,canvas.width,canvas.height,k,o.lang);   // C7 / C8 labels ride along with the colour bar
      }
      return {dataUrl:canvas.toDataURL('image/png'),width:canvas.width,height:canvas.height,scale:k,downgraded:false};
    };
    let k=[1,2,4].includes(Number(o.scale))?Number(o.scale):1,result=null;
    try {
      try{result=attempt(k);}
      catch(err){if(k===1)return {error:'导出失败：'+(err&&err.message||err)};try{result=attempt(1);result.downgraded=true;}catch(err2){return {error:'导出失败：'+(err2&&err2.message||err2)};}}
    } finally {
      try{renderer.setPixelRatio(ratio);renderer.setSize(width,height,false);renderer.setClearColor(prevColor,prevAlpha);camera.aspect=width/height;camera.updateProjectionMatrix();}catch(_){}
    }
    return result;
  }
  function exportPNG(viewName) {
    readExportOptions();
    const r=renderExport({...exportOptions,lang});
    if(!r||r.error){setText('export-status',r&&r.error||'导出失败。');return null;}
    const field=$('volume-field').value==='velocity'?'speed':'pressure';
    const filename=exportFilenameLocal({case_id:meta.case_id,view:viewName||'custom',field,scale:r.scale});
    download(filename,r.dataUrl);
    if(exportOptions.colorbar==='svg')download(exportFilenameLocal({case_id:meta.case_id,view:viewName||'custom',field,scale:r.scale,ext:'svg'}).replace(/_\d+x\.svg$/,'_colorbar.svg'),'data:image/svg+xml;charset=utf-8,'+encodeURIComponent(colorbarSvgCurrent(lang)));
    setText('export-status',`已导出 ${filename}（${r.width}×${r.height}）${r.downgraded?'；显卡不支持所选分辨率，已降到 1×':''}${exportOptions.colorbar==='svg'?'，色标已另存 SVG':''}。`);
    return r;
  }
  function exportSixViews() {
    if(!camera||!frame){setText('export-status','没有解剖坐标架或三维视图，无法导出标准视角。');return;}
    const saved=cameraState(),done=[];
    for(const name of Object.keys(STANDARD_VIEWS)){setCamera(fittedCamera(name));if(exportPNG(name))done.push(viewLabel(name,lang));}
    setCamera(saved);
    setText('export-status',`已导出 ${done.length} 个标准视角（${done.join(' / ')}）${measurements.length||annotations.length?'，含测量与标注标签':''}。`);
  }
  // ---------------------------------------------------------------- §15 figures (2026-09-21)
  // Slice CSV, station-series montage, multi-view montage, profile SVG and the one-pager snapshots.
  // Everything shares the current colour range, units and language; nothing recomputes a prediction.
  const safeFile=x=>{const c=common();if(c&&typeof c.safeName==='function'){try{return c.safeName(x);}catch(_){}}
    return String(x==null?'':x).replace(/[^\w㐀-鿿-]+/g,'_').replace(/^_+|_+$/g,'')||'case';};
  const currentFieldKey=()=>$('volume-field').value==='velocity'?'velocity':'pressure';
  const fileFieldKey=field=>(field||currentFieldKey())==='velocity'?'speed':'pressure';
  const rawUnitOf=field=>field==='velocity'?'m/s':'Pa';
  const caseName=()=>meta.case_id||'volume';
  async function apiPost(name,body) {
    const token=await csrf();
    const r=await fetch(online.api+name,{method:'POST',credentials:'same-origin',headers:{'Content-Type':'application/json','X-CSRF-Token':token},body:JSON.stringify(body)});
    let j={};try{j=await r.json();}catch(_){}
    if(!r.ok){const err=new Error(j&&j.error&&j.error.message||`HTTP ${r.status}`);err.status=r.status;throw err;}
    return j;
  }
  // ---- §15.9 slice CSV: the zoom map's grid, in the plane's own coordinates and the field's raw units ----
  function sliceCSV() {
    const c=common();
    if(!c||typeof c.tableToCSV!=='function')return null;
    const map=sliceZoomLast;
    if(!sliceLast||!map||!map.grid||!Array.isArray(map.bounds))return null;
    const field=sliceLast.field,plane=sliceLast.plane,units=rawUnitOf(field),quantity=sliceLast.quantity;
    const rows=sliceGridToRows(map.grid,map.bounds,units);
    if(!rows.length)return null;
    const filled=Boolean(sliceLast.data&&sliceLast.data.fillOn&&sliceLast.data.loop&&!sliceLast.data.loop.open);
    const comments=[
      `病例 ${caseName()}`,
      `物理量 ${quantityLabel(field,quantity,'zh')}（数值为原始单位 ${units}，与显示单位无关）${quantity==='normal'?'；速度 · 截面法向，'+normalSense(($('slice-basis')||{}).value):''}`,
      `显示${scaleCaption(field,sliceLast.range,quantity,'zh')}（只影响颜色，不改数值）`,
      `中心 xyz [${plane.origin.map(v=>v.toFixed(3)).join(', ')}] mm`,
      `法向 [${plane.normal.map(v=>v.toFixed(4)).join(', ')}]`,
      `厚度 ${num('slice-thickness',2).toFixed(1)} mm`,
      `网格 ${map.grid.nx}×${map.grid.ny}`,
      `补全模式 ${filled?'壁面边界补全（filled_from_wall=1 为主要由壁面边界补出的格子）':'严格模式（只有预测点支撑的格子，filled_from_wall 恒为 0）'}`,
      'x_mm / y_mm 为截面平面内以截面中心为原点的格心坐标（u 轴、v 轴）。'];
    return {csv:c.tableToCSV(['x_mm','y_mm','value','units','filled_from_wall'],rows,comments),rows:rows.length,
      name:`${safeFile(caseName())}_slice_${sliceFileKey(field,quantity)}.csv`};
  }
  function exportSliceCSV() {
    const out=sliceCSV();
    if(!out){setText('slice-zoom-status','当前截面没有可导出的格点（先打开「补全截面」或换一处位置）。');return null;}
    download(out.name,'data:text/csv;charset=utf-8,'+encodeURIComponent(out.csv));
    setText('slice-zoom-status',`已导出 ${out.rows} 个格点：${out.name}。`);
    return out;
  }
  // ---- §15.10 station series along one branch ----
  function seriesGroup() {
    const el=$('slice-series-branch'),id=el&&el.value!==''&&el.value!==undefined?Number(el.value):NaN;
    return groups.find(g=>Number(g.segment)===id)||groups[0]||null;
  }
  const sliceFileKey=(field,quantity)=>quantity==='normal'?'through_plane':fileFieldKey(field);
  function seriesPanelURL(station,field,range,quantity,width,height) {
    const canvas=document.createElement('canvas');canvas.width=width;canvas.height=height;
    const ctx=canvas.getContext&&canvas.getContext('2d');
    if(!ctx)throw new Error('2D 画布不可用');
    renderSliceMap(ctx,width,height,station.data,range,field,{values:station.values,gridMax:160,footer:false,points:false,colorbar:false,lang,
      quantity,arrows:arrowsOn(field),maxArrows:120,pad:{left:20,top:15,right:20,bottom:20}});
    if(typeof canvas.toDataURL!=='function')throw new Error('画布导出不可用');
    return canvas.toDataURL('image/png');
  }
  async function colorbarImage(scale,note) {
    const c=common();
    if(!c||typeof c.loadImage!=='function')return null;
    try{return {image:await c.loadImage('data:image/svg+xml;charset=utf-8,'+encodeURIComponent(colorbarSvgCurrent(lang,scale,note)),document)};}catch(_){return null;}
  }
  // Stations along one branch; 本截面 mode gives the whole series one shared colour bar spanning every station's
  // own robust range (±max for through-plane velocity), so the panels stay comparable.
  function seriesStations(group,fractions,field,quantity) {
    const thickness=num('slice-thickness',2);
    return fractions.map(fraction=>{
      const base=centerlinePlane(group,fraction),plane={...base,...planeBasis(base.normal)};
      const indices=slabIndices(pts,walls,segments,plane,thickness,raw.has_segments?group.segment:null);
      const values=sliceValuesFor(field,quantity,plane,indices);
      return {fraction,plane,values,data:sliceMapData(indices,values,plane,field)};
    });
  }
  function seriesScale(stations,field,quantity) {
    if(sliceRangeMode!=='section')return sliceScale(field,quantity,[]);
    const parts=stations.map(st=>sliceScale(field,quantity,st.data.finite.map(x=>x.v))).filter(r=>r.source==='section');
    if(!parts.length)return {...globalScale(field,quantity),source:'fallback'};
    let lo=Math.min(...parts.map(r=>r.min)),hi=Math.max(...parts.map(r=>r.max));
    if(quantity==='normal'){const m=Math.max(Math.abs(lo),Math.abs(hi));lo=-m;hi=m;}
    return {min:lo,max:hi,log:quantity==='speed'&&logScale,diverging:quantity==='normal',source:'section',shared:true};
  }
  async function exportSliceSeries() {
    const note=text=>setText('slice-series-note',text);
    const c=common();
    if(!c||typeof c.composeMontage!=='function'||typeof c.loadImage!=='function'){note('拼图需要共享库。');return null;}
    const group=seriesGroup();
    if(!group||group.points.length<2||!(group.arc[group.arc.length-1]>0)){note('没有可用的中心线分支。');return null;}
    if(!fieldValues()){note('当前物理量不可用。');return null;}
    const field=currentFieldKey(),quantity=quantityOf(field);
    const fractions=seriesFractions(num('slice-series-count',6));
    sliceSeries={segment_id:Number(group.segment),fractions:fractions.slice()};
    const length=group.arc[group.arc.length-1];
    note(`正在渲染 ${fractions.length} 个站位…`);
    try {
      const stations=seriesStations(group,fractions,field,quantity),range=seriesScale(stations,field,quantity);
      const panels=[];
      for(let i=0;i<stations.length;i++) {
        const url=seriesPanelURL(stations[i],field,range,quantity,700,620);
        panels.push({image:await c.loadImage(url,document),label:String.fromCharCode(97+i),
          caption:`s = ${(fractions[i]*length).toFixed(1)} mm (${Math.round(fractions[i]*100)}%)`});
      }
      const branch=branchLabel(branchName(group.segment),lang),shared=range.shared?(lang==='en'?'shared by the series':'系列共用'):'';
      const canvas=c.composeMontage({columns:3,panels,colorbar:await colorbarImage(range,shared),document,background:'#ffffff',
        title:[caseName(),branch,`${quantityLabel(field,quantity,lang)} · ${unitOf(field)}`,shared?(lang==='en'?'one colour bar for all stations':'系列共用色标'):''].filter(Boolean).join(' · ')});
      if(!canvas||typeof canvas.toDataURL!=='function')throw new Error('拼图画布不可用');
      const name=`${safeFile(caseName())}_slices_${safeFile(branchName(group.segment))}_${sliceFileKey(field,quantity)}.png`;
      download(name,canvas.toDataURL('image/png'));
      note(`已导出 ${panels.length} 站截面系列拼图：${name}（${scaleCaption(field,range,quantity,'zh')}）。`);
      return {name,range:{...range}};
    } catch(err){note('导出失败：'+(err&&err.message||err));return null;}
  }
  // ---- §15.11 multi-view montage ----
  const MONTAGE_VIEWS=['front','back','left','right','top','bottom','current','slice'];
  const MONTAGE_DEFAULT=['front','left','current','slice'];
  function montageSelection() {
    const out=[];for(const name of MONTAGE_VIEWS){const box=$('mv-'+name);if(box&&box.checked)out.push(name);}return out;
  }
  function montageLabel(name,language) {
    const en=(language||lang)==='en';
    if(name==='current')return en?'Current view':'当前视角';
    if(name==='slice')return en?'Section':'截面';
    return en?viewLabel(name,'en')+' view':viewLabel(name,'zh')+'视';
  }
  // The zoom map on its own canvas: used as the montage's "section" panel and as the one-pager's slice figure.
  // ``withBar`` draws the slice's own colour bar (one-pager figure); the montage panel shares the montage's bar,
  // so outside the 截面 page (legend = whole field) the panel is drawn on that same scale and quantity.
  function sliceCanvasURL(width,height,withBar) {
    if(!sliceLast)return null;
    const canvas=document.createElement('canvas');canvas.width=width;canvas.height=height;
    const ctx=canvas.getContext&&canvas.getContext('2d');
    if(!ctx||typeof canvas.toDataURL!=='function')return null;
    let {data,range,field,values,quantity}=sliceLast;
    if(!withBar&&$('volume-mode').value!=='slice'){
      const legend=currentScale();quantity=legend.quantity;range=legend;
      if(quantity!==sliceLast.quantity){values=sliceValuesFor(field,quantity,sliceLast.plane,sliceLast.indices);data=sliceMapData(sliceLast.indices,values,sliceLast.plane,field);}
    }
    renderSliceMap(ctx,width,height,data,range,field,{values,points:false,gridMax:240,cellPx:5,font:22,lineWidth:2.2,
      colorbar:Boolean(withBar),footer:false,background:'#ffffff',lang,pad:{left:40,top:30,right:40,bottom:withBar?120:60},quantity,arrows:arrowsOn(field),maxArrows:300,arrowWidth:2.4});
    return canvas.toDataURL('image/png');
  }
  async function exportMontage() {
    const note=text=>setText('montage-status',text);
    const c=common();
    if(!c||typeof c.composeMontage!=='function'||typeof c.loadImage!=='function'){note('拼图需要共享库。');return null;}
    const wanted=montageSelection();
    if(!wanted.length){note('请至少勾选一个视角。');return null;}
    readExportOptions();
    const scale=[1,2,4].includes(Number(exportOptions.scale))?Number(exportOptions.scale):1;
    const saved=cameraState(),shots=[],skipped=[];
    try {
      for(const name of wanted) {
        if(name==='slice') {
          const url=sliceCanvasURL(1400,1200);
          if(!url){skipped.push(montageLabel(name,lang));continue;}
          shots.push({url,caption:montageLabel(name,lang)});continue;
        }
        if(name!=='current') {
          const cam=frame&&camera?fittedCamera(name):null;
          if(!cam){skipped.push(montageLabel(name,lang));continue;}
          setCamera(cam);
        }
        const r=renderExport({...exportOptions,scale,colorbar:'none',lang});
        if(!r||r.error){skipped.push(montageLabel(name,lang));continue;}
        shots.push({url:r.dataUrl,caption:montageLabel(name,lang)});
      }
    } finally {if(saved)setCamera(saved);}
    if(!shots.length){note('没有可用的视角'+(skipped.length?`（跳过 ${skipped.join(' / ')}）`:'')+'。');return null;}
    try {
      const panels=[];
      for(let i=0;i<shots.length;i++)panels.push({image:await c.loadImage(shots[i].url,document),label:String.fromCharCode(97+i),caption:shots[i].caption});
      const field=currentFieldKey(),columns=Math.max(2,Math.min(4,Math.round(num('montage-columns',2))));
      const canvas=c.composeMontage({columns,panels,colorbar:await colorbarImage(),document,background:'#ffffff',
        title:[caseName(),`${quantityLabel(field,currentScale().quantity,lang)} · ${unitOf(field)}`].join(' · ')});
      if(!canvas||typeof canvas.toDataURL!=='function')throw new Error('拼图画布不可用');
      const name=`${safeFile(caseName())}_montage_${fileFieldKey(field)}_${scale}x.png`;
      download(name,canvas.toDataURL('image/png'));
      note(`已导出 ${panels.length} 幅视角拼图：${name}${skipped.length?`；跳过 ${skipped.join(' / ')}`:''}。`);
      return name;
    } catch(err){note('导出失败：'+(err&&err.message||err));return null;}
  }
  // ---- §15.12 profile curves as SVG (one file per quantity, radius separately) ----
  function buildProfileSVGs() {
    const c=common();
    if(!c||typeof c.profileSVG!=='function')return null;
    const branch=currentProfile();
    if(!branch)return null;
    const quantity=$('profile-quantity').value||'speed',field=quantity==='pressure'?'pressure':'velocity',vol=branch.volume||{};
    const en=lang==='en';
    const xs=(Array.isArray(branch.s_from_root_mm)&&branch.s_from_root_mm.length?branch.s_from_root_mm:branch.s_local_mm||[]).map(Number);
    const name=branchLabel(branch.name||branchName(branch.segment_id),lang);
    const title=[caseName(),name].join(' · ');
    const xLabel=en?'Arc length from inlet (mm)':'距入口弧长 (mm)';
    const conv=list=>(list||[]).map(v=>v===null||!Number.isFinite(Number(v))?NaN:shown(Number(v),field));
    const wanted=quantity==='pressure'
      ? [[en?'Mean':'均值',vol.pressure_mean_pa,null],[en?'Minimum':'最低',vol.pressure_min_pa,'6 3']]
      : [[en?'Mean':'均值',vol.speed_mean_m_s,null],[en?'Maximum':'最大',vol.speed_max_m_s,'6 3']];
    const series=wanted.filter(item=>Array.isArray(item[1])).map(([label,list,dash])=>({name:label,x:xs,y:conv(list),dash:dash||undefined}));
    const out={};
    if(series.length&&xs.length)out.field={
      name:`${safeFile(caseName())}_profile_${safeFile(name)}_${fileFieldKey(field)}.svg`,
      svg:c.profileSVG({series,xLabel,yLabel:`${fieldLabel(field,lang)} · ${unitOf(field)}`,title})};
    // The radius file also carries the §17.3 station diameters; x stays the profile's own axis.
    const radius=Array.isArray(branch.radius_mm)?branch.radius_mm.map(v=>v===null?NaN:Number(v)):null;
    const usedRoot=Array.isArray(branch.s_from_root_mm)&&branch.s_from_root_mm.length>0;
    const morph=morphSeriesFor(branch),mx=morph?(usedRoot?morph.xRoot:morph.x):null;
    const radiusSeries=[];
    if(radius&&xs.length&&radius.some(Number.isFinite))radiusSeries.push({name:labelText('半径',lang),x:xs,y:radius,color:'#3f8f6b'});
    if(morph&&morph.max&&morph.max.some(Number.isFinite))radiusSeries.push({name:labelText('最大直径',lang),x:mx,y:morph.max,color:'#8e44ad'});
    if(morph&&morph.equiv&&morph.equiv.some(Number.isFinite))radiusSeries.push({name:labelText('等效直径',lang),x:mx,y:morph.equiv,color:'#b07cc6',dash:'6 3'});
    if(radiusSeries.length)out.radius={
      name:`${safeFile(caseName())}_profile_${safeFile(name)}_radius.svg`,
      svg:c.profileSVG({series:radiusSeries,xLabel,
        yLabel:radiusSeries.length>1?`${labelText('半径',lang)} / ${labelText('直径',lang)} · mm`:`${labelText('半径',lang)} · mm`,title})};
    return out;
  }
  function exportProfileSVG() {
    const out=buildProfileSVGs();
    if(!out||!(out.field||out.radius)){setText('profile-svg-note','没有可导出的沿程曲线（本报告没有 profiles 或当前分支为空）。');return null;}
    const done=[];
    for(const key of ['field','radius'])if(out[key]) {
      download(out[key].name,'data:image/svg+xml;charset=utf-8,'+encodeURIComponent(out[key].svg));
      done.push(out[key].name);
    }
    setText('profile-svg-note',`已导出 ${done.join(' 与 ')}。`);
    return done;
  }
  // ---- §15.8 one-pager snapshots (online only) ----
  function setOnepageStatus(text,withLink) {
    const el=$('onepage-status');if(!el)return;
    if(el.replaceChildren)el.replaceChildren();
    el.textContent=withLink?text+' ':text;
    if(!withLink||!online||!document.createElement)return;
    const a=document.createElement('a');a.textContent='打开一页纸';a.href=online.api+'onepage';
    if(a.setAttribute){a.setAttribute('target','_blank');a.setAttribute('rel','noopener');}
    if(el.appendChild)el.appendChild(a);
  }
  function snapshotCaption(view,language) {
    const field=currentFieldKey(),quantity=view==='slice'&&sliceLast?sliceLast.quantity:currentScale().quantity;
    return `${montageLabel(view,language||lang)} · ${quantityLabel(field,quantity,language||lang)} · ${unitOf(field)}`;
  }
  async function makeOnepageShots() {
    const note=text=>setOnepageStatus(text,false);
    if(!online){note('离线报告无法上传配图；请在工作台中打开本报告后再点此按钮。');return null;}
    const images=[],skipped=[],saved=cameraState();
    try {
      for(const name of ['front','left','top','current']) {
        if(name!=='current') {
          const cam=frame&&camera?fittedCamera(name):null;
          if(!cam){skipped.push(montageLabel(name,lang));continue;}
          setCamera(cam);
        }
        const r=renderExport({...exportOptions,scale:2,background:'white',colorbar:'overlay',lang});
        if(!r||r.error){skipped.push(montageLabel(name,lang));continue;}
        images.push({name,view:name,png_base64:String(r.dataUrl).replace(/^data:image\/png;base64,/,''),
          width:r.width,height:r.height,caption:snapshotCaption(name,lang)});
      }
    } finally {if(saved)setCamera(saved);}
    if(sliceLast&&($('volume-mode').value==='slice'||sliceZoomOpen)) {
      const url=sliceCanvasURL(1400,1200,true);
      if(url)images.push({name:'slice',view:'slice',png_base64:url.replace(/^data:image\/png;base64,/,''),
        width:1400,height:1200,caption:snapshotCaption('slice',lang)});
    }
    if(!images.length){note('没有可用配图'+(skipped.length?`（${skipped.join(' / ')} 不可用）`:'')+'。');return null;}
    note(`正在上传 ${images.length} 张配图…`);
    try {
      const out=await apiPost('snapshots',{images,replace:true});
      const kept=out&&out.snapshots&&Array.isArray(out.snapshots.items)?out.snapshots.items.length:images.length;
      setOnepageStatus(`已生成 ${kept} 张一页纸配图${skipped.length?`（${skipped.join(' / ')} 不可用）`:''}。`,true);
      return images.map(x=>x.name);
    } catch(err){note('上传失败：'+(err&&err.message||err));return null;}
  }
  // v0.14 (F1): tell a same-origin parent (the compare page) that the display changed, without waiting for a camera move.
  let announceOn=false,announceTimer=null;
  function announceChange() {
    if(!announceOn||announceTimer||!root.parent||root.parent===root)return;
    announceTimer=setTimeout(()=>{announceTimer=null;postParent({type:'wss-view:changed',family:'volume',run_identity:meta.run_identity||null});},120);
  }
  function refresh() {
    requestRender();announceChange();
    const isVelocity=$('volume-field').value==='velocity',values=fieldValues(),field=isVelocity?'velocity':'pressure',units=unitOf(field);
    const modeSelect=$('volume-mode');
    const wallOption=modeSelect.querySelector?modeSelect.querySelector('option[value="wall"]'):null,lineOption=modeSelect.querySelector?modeSelect.querySelector('option[value="streamlines"]'):null;
    // The hidden select mirrors data availability only; a field change away from wall / streamlines falls back below.
    if(wallOption)wallOption.disabled=!modeAvailable('wall');
    if(lineOption)lineOption.disabled=!modeAvailable('streamlines');
    let mode=modeSelect.value;
    if((mode==='wall'&&(isVelocity||!wallPressure))||(mode==='streamlines'&&(!isVelocity||!lines.length))) {modeSelect.value='cloud';mode='cloud';sliceSelected=false;}
    refreshModeTabs();
    $('volume-vectors').disabled=!isVelocity;
    setHidden('vectors-row',!isVelocity||mode!=='cloud');setHidden('streamline-settings',mode!=='streamlines');
    $('slice-branch-label').hidden=$('slice-basis').value!=='centerline';
    setHidden('slice-panel',!(mode==='slice'||cutActive));
    setText('volume-field-note',isVelocity?'速度展示排除壁面点。颜色表示速度大小，箭头表示预测向量方向。':'压力体内点和壁面使用同一色标；灰色壁面表示缺少有效插值支撑。');
    // Whole-field display scale (speed may be log); the 截面 page swaps in the slice's own scale below.
    const full=fieldStats[$('volume-field').value],range=globalScale(field,isVelocity?'speed':'pressure');
    const plane=currentPlane(),thickness=Number($('slice-thickness').value);
    const moduleValue=$('volume-module').value;
    const cutSide=cutSideFromValue(moduleValue);
    const selectedSegment=(moduleValue!==''&&cutSide===null)?Number(moduleValue):null;
    const branchFilter=selectedSegment!==null?selectedSegment:(cutSide===null&&raw.has_segments&&plane.segment!==null?plane.segment:null);
    let slicedAll=moduleIndices(slabIndices(pts,walls,segments,plane,thickness,branchFilter),segments,selectedSegment);
    slicedAll=sideIndices(slicedAll,pts,plane,cutSide);
    const sliced=visibleIndices(slicedAll);   // hidden branches change the drawing only; statistics stay on all points
    controls&&(controls.enabled=!gesture);
    setStats('volume-statistics',full,field);
    const isPick=$('slice-basis').value==='pick';
    setText('slice-details',`中心 [${plane.origin.map(v=>v.toFixed(2)).join(', ')}] mm；法向 [${plane.normal.map(v=>v.toFixed(3)).join(', ')}]；厚度 ${thickness.toFixed(1)} mm${branchFilter!==null?'；'+branchName(branchFilter):''}。`);
    setText('slice-cut-status',cutActive?'已按当前截面切割；可在“血管模块”中选择截面 A 侧、截面 B 侧或全部。':'尚未切割；“血管模块”可按中心线分支筛选。');
    setText('pick-status',pickInfo||(pickMode?'点击半透明壁面上的一点：截面垂直于该处中心线；再点第二点：截面通过两点。':'点「点选定位截面」后在血管壁上点 1 或 2 个点即可放置截面；放好后可直接拖动蓝色截面。'));
    setText('slice-position-label',isPick?'沿法向微调':'位置');
    setHidden('slice-tools',mode!=='slice');setHidden('pick-hint',!pickMode);
    setText('pick-hint',picks.length===1?'第 2 点（可选）：再点一处让截面通过两点；或按 Esc / 「结束点选」后直接拖动截面。':'第 1 点：在半透明血管壁上点击，截面将垂直于该处中心线（Esc 取消）。');
    setText('slice-position-value',isPick?((Number($('slice-position').value)-50)*.5).toFixed(1)+' mm':Number($('slice-position').value).toFixed(1)+'%');setText('slice-pitch-value',$('slice-pitch').value+'°');setText('slice-yaw-value',$('slice-yaw').value+'°');setText('slice-thickness-value',thickness.toFixed(1));setText('slice-offset-u-value',(Number($('slice-offset-u').value)||0).toFixed(1));setText('slice-offset-v-value',(Number($('slice-offset-v').value)||0).toFixed(1));setText('volume-opacity-value',Number($('volume-opacity').value).toFixed(2));
    setText('streamline-width-value',(Number($('streamline-width').value)||1).toFixed(1)+'×');
    setText('view-note',(pickMode?'点选模式：单击壁面放置截面点，空白处拖动旋转视图。 ':'')+{cloud:'体内预测点云 · 拖动旋转，滚轮缩放，悬停读值',slice:'有限厚度截面 · 右侧为平面投影'+(pickMode||compactOn?'':'；拖动蓝色截面调整（见顶部工具条）'),wall:'壁面压力 · 悬停读值；血管内部可通过体内点云与横截面查看',streamlines:'预测向量场的积分流线 · 颜色为局部速度大小'}[mode]
      +(probeEnabled?'':' · 探针已关闭（按 P 打开）')+(shortcuts&&!compactOn?' · 按 ? 查看快捷键':''));
    setText('view-direction-note',frame?(frame.direction_source&&frame.direction_source!=='unknown_stl'?`患者方向来源：${frame.direction_source}`:'标准视角按解剖坐标架推断，请核对左右。'):'本报告没有解剖坐标架，标准视角不可用。');
    drawSlice(sliced,plane,field);setStats('slice-statistics',statistics(values,slicedAll),field,sectionRows());   // statistics stay on the raw field
    const inSlice=mode==='slice'&&Boolean(sliceScaleNow),legendScale=inSlice?sliceScaleNow:range,legendQuantity=inSlice?quantityOf(field):(isVelocity?'speed':'pressure');
    legendScaleNow={...legendScale,quantity:legendQuantity};
    {const [lo,hi]=scaleEnds(legendScale);
     setText('legend-title',quantityLabel(field,legendQuantity,lang)+' · '+units);setText('legend-min',fmt(shown(lo,field)));setText('legend-max',fmt(shown(hi,field)));
     setText('legend-note',legendNote(field,legendScale,legendQuantity,inSlice));
     const bar=$('legend-bar');if(bar&&bar.style)bar.style.background=colormapCSS(legendScale.diverging?'bwr':undefined);
     // v0.15: text alternative for the colour bar (quantity, units and range)
     if(bar&&bar.setAttribute){bar.setAttribute('role','img');bar.setAttribute('aria-label',`色标 ${quantityLabel(field,legendQuantity,lang)} · ${units}：${fmt(shown(lo,field))} – ${fmt(shown(hi,field))}`);}}
    writeSliceControls(field);
    renderTrustLegend();renderNarrative();drawProfile();updateRegion();renderProbe();clearContent();
    if(!renderer||!content)return;
    contextMesh.visible=mode!=='wall';contextMesh.material.opacity=Number($('volume-opacity').value);applyCutPlanes(contextMesh.material,plane,cutSide);
    {const vf=visibleFaces();if(!contextMesh.geometry.index||contextMesh.geometry.index.array!==vf)contextMesh.geometry.setIndex(new THREE.BufferAttribute(vf,1));}
    {const side=gizmoSide(plane.origin);sliceGizmo.visible=mode==='slice';sliceGizmo.position.set(...plane.origin);sliceGizmo.quaternion.setFromUnitVectors(new THREE.Vector3(0,0,1),new THREE.Vector3(...plane.normal));
     planeMesh.scale.set(side,side,1);planeEdge.scale.set(side,side,1);planeArrow.setLength(side*.55,side*.14,side*.08);}
    if(mode==='wall') {
      const geometry=contextMesh.geometry.clone(),colors=new Float32Array(vertices.length);
      for(let i=0;i<wallPressure.length;i++){let c=scaleColor(wallPressure[i],range);if(trustOverlay&&wallTrust&&(wallTrust[i]&TRUST_WALL_SOFT))c=desaturate(c);colors.set(c,i*3);}
      geometry.setAttribute('color',new THREE.BufferAttribute(colors,3));const wallMaterial=new THREE.MeshPhongMaterial({vertexColors:true,side:THREE.DoubleSide,shininess:10});applyCutPlanes(wallMaterial,plane,cutSide);const wallMesh=new THREE.Mesh(geometry,wallMaterial);wallMesh.userData.kind='wall';content.add(wallMesh);
    } else if(mode==='streamlines') {
      addStreamlines(range,plane,cutSide);
    } else {
      const visible=mode==='slice'?sliced:visibleIndices(sideIndices(moduleIndices(interior,segments,selectedSegment),pts,plane,cutSide));
      // 截面 page: the slab points carry the slice quantity (speed or v·n) on the slice's own scale, like the map.
      const inSlab=mode==='slice'&&sliceLast,pointValues=inSlab?sliceLast.values:values,pointScale=inSlab?sliceLast.range:range;
      const cloud=new THREE.Points(geometryFor(visible,pointValues,pointScale),new THREE.PointsMaterial({vertexColors:true,size:diagonal/500,transparent:true,opacity:mode==='slice'?1:.8,sizeAttenuation:true}));cloud.userData.indices=visible;content.add(cloud);
      if(isVelocity&&$('volume-vectors').checked)addVectors(visible,globalScale('velocity','speed'));
    }
    drawHighlight();
  }
  for(const id of ['volume-field','volume-mode','volume-vectors','slice-basis','slice-branch','volume-module','streamline-density','streamline-thin','profile-branch','profile-quantity','region-branch']) $(id).addEventListener('change',()=>{
    if(id==='volume-mode') sliceSelected=$('volume-mode').value==='slice';
    if(id==='slice-basis'&&$('slice-basis').value==='pick'&&!pickPlane){$('slice-basis').value=groups.length?'centerline':'z';setText('pick-status','还没有选点；开启点选后在视图中点击。');}
    refresh();
  });
  $('colormap').addEventListener('change',()=>{setColormap($('colormap').value);try{root.localStorage&&root.localStorage.setItem('wss-volume-colormap',currentMap);}catch(_){}refresh();});
  $('color-bands').addEventListener('change',()=>{setBands($('color-bands').value);try{root.localStorage&&root.localStorage.setItem('wss-volume-bands',String(currentBands));}catch(_){}refresh();});
  $('pressure-unit').addEventListener('change',()=>{const v=$('pressure-unit').value;if(UNITS.pressure[v])pressureUnit=v;refresh();});
  $('velocity-unit').addEventListener('change',()=>{const v=$('velocity-unit').value;if(UNITS.velocity[v])velocityUnit=v;refresh();});
  $('trust-overlay').addEventListener('change',()=>{trustOverlay=Boolean($('trust-overlay').checked);refresh();});
  $('region-highlight').addEventListener('change',()=>{regionOn=Boolean($('region-highlight').checked);refresh();});
  // §17.3 automatic labels + the max-diameter ring / fly-to
  {const el=$('labels-findings');if(el)el.addEventListener('change',()=>{setLabels({findings:el.value});drawMarkers();});}
  {const el=$('labels-branches');if(el)el.addEventListener('change',()=>{setLabels({branches:el.checked});drawMarkers();});}
  {const el=$('labels-max-diameter');if(el)el.addEventListener('change',()=>{setLabels({max_diameter:el.checked});drawMarkers();});}
  {const b=$('max-diameter-fly');if(b)b.addEventListener('click',()=>{
    if(!morphMax||!morphMax.xyz_mm){setText('max-diameter-text',($('max-diameter-text')||{}).textContent||'');return;}
    flyTo(morphMax.xyz_mm,Number(morphMax.max_diameter_mm)/2||diagonal*0.03);});}
  let updatePending=false;
  for(const id of ['volume-opacity','slice-position','slice-pitch','slice-yaw','slice-thickness','slice-offset-u','slice-offset-v','streamline-width','region-smin','region-smax']) $(id).addEventListener('input',()=>{if(!updatePending){updatePending=true;root.requestAnimationFrame(()=>{updatePending=false;refresh();});}});
  for(const mode of MODES) {const button=modeButton(mode);if(button)button.addEventListener('click',()=>setMode(mode));}
  for(const id of ['pick-toggle','pick-toggle-2','pick-toggle-menu']){const b=$(id);if(b)b.addEventListener('click',()=>setPickMode(!pickMode));}
  for(const [id,kind] of [['drag-move','move'],['drag-rotate','rotate'],['drag-offset','offset']]){const b=$(id);if(b)b.addEventListener('click',()=>setDragMode(kind));}
  {const b=$('slice-reset-angle');if(b)b.addEventListener('click',()=>{for(const id of ['slice-pitch','slice-yaw','slice-offset-u','slice-offset-v'])$(id).value='0';refresh();});}
  $('pick-clear').addEventListener('click',clearPicks);
  $('probe-clear').addEventListener('click',()=>{probe=null;probePinned=false;renderProbe();});
  $('probe-record').addEventListener('click',()=>{const row=probeToRow();if(!row){setText('probe-log-note','没有探针读数可记录。');return;}probeLog=probeLog.concat([row]);renderProbeLog();setText('probe-note',`已记录为 ${row.id}。`);});
  $('probe-log-copy').addEventListener('click',()=>{const text=probeSerializer('tsv')(probeLog,lang);const done=()=>setText('probe-log-note','已复制 TSV 到剪贴板。');if(root.navigator&&root.navigator.clipboard&&root.navigator.clipboard.writeText)root.navigator.clipboard.writeText(text).then(done,()=>setText('probe-log-note','剪贴板不可用；请改用导出 CSV。'));else setText('probe-log-note','剪贴板不可用；请改用导出 CSV。');});
  $('probe-log-export').addEventListener('click',()=>{const csv=probeSerializer('csv')(probeLog,lang);download(`${(meta.case_id||'volume')}_probes.csv`,'data:text/csv;charset=utf-8,'+encodeURIComponent(csv));setText('probe-log-note',`已导出 ${probeLog.length} 条探针记录 CSV。`);});
  $('probe-log-clear').addEventListener('click',()=>{probeLog=[];renderProbeLog();});
  // C7 measurement tools
  for(const key of Object.keys(MEASURE_MODES)){const b=$('measure-'+key);if(b)b.addEventListener('click',()=>setMeasureMode(key));}
  {
    const clear=$('measure-clear');
    if(clear)clear.addEventListener('click',()=>{measurements=[];measurePicks=[];renderMeasurements();drawMarkers();setText('measure-note','已清空测量。');});
    const copy=$('measure-copy');
    if(copy)copy.addEventListener('click',()=>{
      if(!measurements.length){setText('measure-note','还没有测量可复制。');return;}
      const text=measureTSV(lang);
      const done=()=>setText('measure-note',`已复制 ${measurements.length} 条测量（TSV）。`);
      if(root.navigator&&root.navigator.clipboard&&root.navigator.clipboard.writeText)root.navigator.clipboard.writeText(text).then(done,()=>setText('measure-note','剪贴板不可用；测量随视图状态 JSON 导出。'));
      else setText('measure-note','剪贴板不可用；测量随视图状态 JSON 导出。');
    });
  }
  // C8 annotations
  {
    const add=$('annot-add');
    if(add)add.addEventListener('click',()=>{
      if(!renderer&&!annotMode){setText('annot-status','三维视图不可用，无法点选位置。');return;}
      setAnnotMode(!annotMode);
      if(!annotMode)setText('annot-status','已取消钉标注。');
    });
    const save=$('annot-save');
    if(save)save.addEventListener('click',()=>{if(annotSaveTimer){clearTimeout(annotSaveTimer);annotSaveTimer=null;}saveAnnotations();if(!online)setText('annot-status',`离线只读（内嵌副本）：${annotations.length} 条标注只随视图状态保存。`);});
    const clear=$('annot-clear');
    if(clear)clear.addEventListener('click',()=>{annotations=[];renderAnnotations();drawMarkers();scheduleAnnotSave();});
  }
  // C9 presets
  {
    const save=$('preset-save');
    if(save)save.addEventListener('click',()=>{
      const input=$('preset-name');
      let name=String((input&&input.value)||'').trim();
      if(!name&&typeof root.prompt==='function'){try{name=String(root.prompt('预设名称')||'').trim();}catch(_){name='';}}
      if(savePreset(name)&&input)input.value='';
    });
    const out=$('preset-export');
    if(out)out.addEventListener('click',()=>{
      const text=JSON.stringify({schema_version:'wss-deploy.presets/v1',family:'volume',items:presetList},null,1);
      download(`${(meta.case_id||'volume')}_presets.json`,'data:application/json;charset=utf-8,'+encodeURIComponent(text));
      setText('preset-note',`已导出 ${presetList.length} 个用户预设 JSON。`);
    });
  }
  $('findings-clear').addEventListener('click',()=>activateFinding(null));
  $('finding-add').addEventListener('click',()=>{
    const text=(($('finding-add-text')||{}).value||'').trim();
    if(!text){setText('findings-review-status','请先输入发现文字，再点选位置。');return;}
    if(!renderer){setText('findings-review-status','三维视图不可用，无法点选位置。');return;}
    addFindingMode=!addFindingMode;setText('findings-review-status',addFindingMode?'在三维视图中点击一处放置该发现（再点本按钮取消）。':'已取消新增。');
  });
  for(const id of ['export-scale','export-background','export-colorbar','export-hide-ui'])$(id).addEventListener('change',()=>readExportOptions());
  $('export-lang').addEventListener('change',()=>{readExportOptions();refresh();});
  $('export-png').addEventListener('click',()=>exportPNG('custom'));
  $('defaults-save').addEventListener('click',()=>saveDefaults());
  $('profile-canvas').addEventListener('click',event=>{
    if(!profileLayout||!profileLayout.branch)return;const canvas=$('profile-canvas');const rect=canvas.getBoundingClientRect?canvas.getBoundingClientRect():{left:0,width:canvas.width};
    const x=(event.clientX-rect.left)*(canvas.width/Math.max(rect.width,1));const t=clamp((x-profileLayout.left)/profileLayout.width,0,1);
    sliceAtArc(profileLayout.branch,profileLayout.sMin+t*(profileLayout.sMax-profileLayout.sMin));
  });
  $('slice-show').addEventListener('click',()=>{$('volume-mode').value='slice';sliceSelected=true;refresh();});
  $('slice-cut').addEventListener('click',()=>{
    cutActive=true;sliceSelected=true;$('volume-mode').value='cloud';
    const select=$('volume-module');
    if(!select.querySelector||!select.querySelector('option[value="cut-positive"]')) {option(select,'cut-positive','截面 A 侧');option(select,'cut-negative','截面 B 侧');}
    select.value='cut-positive';refresh();
  });
  $('slice-clear-cut').addEventListener('click',()=>{
    cutActive=false;const select=$('volume-module');select.value='';if(select.querySelectorAll)select.querySelectorAll('option[value="cut-positive"],option[value="cut-negative"]').forEach(el=>el.remove());refresh();
  });
  $('slice-apply-auto').addEventListener('click',()=>{const p=presets[Number($('auto-presets').value)];if(!p)return;$('slice-basis').value='centerline';$('slice-branch').value=String(p.segment);$('slice-position').value=String(p.fraction*100);$('slice-pitch').value='0';$('slice-yaw').value='0';$('slice-offset-u').value='0';$('slice-offset-v').value='0';$('volume-mode').value='slice';sliceSelected=true;refresh();});
  $('volume-fit').addEventListener('click',fit);
  for(const name of Object.keys(STANDARD_VIEWS)){const button=$('view-'+name);if(!button)continue;button.disabled=!frame;button.addEventListener('click',()=>showStandardView(name));}
  $('view-save').addEventListener('click',()=>{try{root.localStorage.setItem(viewKey(),JSON.stringify(captureView()));setText('view-status','视图状态已保存到本浏览器。');}catch(_){setText('view-status','浏览器不允许保存。');}});
  $('view-restore').addEventListener('click',()=>{try{const text=root.localStorage.getItem(viewKey());if(!text){setText('view-status','没有已保存的视图状态。');return;}applyView(JSON.parse(text));setText('view-status','已恢复保存的视图状态。');}catch(_){setText('view-status','读取失败。');}});
  $('view-export').addEventListener('click',()=>{const text=JSON.stringify(captureView(),null,1);download(`${(meta.case_id||'volume')}_view.json`,'data:application/json;charset=utf-8,'+encodeURIComponent(text));setText('view-status','已导出视图状态 JSON。');});
  $('view-import').addEventListener('change',event=>{const file=event&&event.target&&event.target.files&&event.target.files[0];if(!file||typeof FileReader==='undefined')return;const reader=new FileReader();reader.onload=()=>{try{applyView(JSON.parse(String(reader.result)));setText('view-status','已导入视图状态。');}catch(err){setText('view-status','导入失败：'+err.message);}};reader.readAsText(file);});
  $('view-link').addEventListener('click',()=>{
    const encoded=encodeView(captureView());let url='';
    try{if(root.location){root.location.hash='view='+encoded;url=root.location.href;}}catch(_){}
    const done=()=>setText('view-status','复现链接已写入地址栏'+(url?'并复制到剪贴板':'')+'；打开该链接即可复现此图。');
    if(url&&root.navigator&&root.navigator.clipboard&&root.navigator.clipboard.writeText)root.navigator.clipboard.writeText(url).then(done,()=>setText('view-status','复现链接已写入地址栏（剪贴板不可用）。'));else done();
  });
  $('view-replay').addEventListener('click',()=>{const state=viewFromHash();if(state){applyView(state);setText('view-status','已按链接复现视图。');}else setText('view-status','地址栏没有视图状态。');});
  $('view-six').addEventListener('click',exportSixViews);
  // Accordion: opening one top-level menu closes the others (browser only).
  const MENUS=['menu-display','menu-findings','menu-profiles','menu-measure','menu-annot','menu-presets','menu-slice','menu-view','menu-stats'];
  for(const id of MENUS) {const el=$(id);if(el&&el.addEventListener)el.addEventListener('toggle',()=>{if(el.open)for(const other of MENUS)if(other!==id&&$(other))$(other).open=false;});}
  {const f=$('slice-fill');if(f)f.addEventListener('change',refresh);}
  // Slice colour controls (panel and zoom view), velocity log scale.
  for(const id of ['slice-range-mode','slice-zoom-range']){const el=$(id);if(el)el.addEventListener('change',()=>setSliceDisplay({mode:el.value}));}
  for(const [a,b] of [['slice-min','slice-max'],['slice-zoom-min','slice-zoom-max']])for(const id of [a,b]){const el=$(id);if(el)el.addEventListener('change',()=>setSliceDisplay({min:($(a)||{}).value,max:($(b)||{}).value}));}
  for(const id of ['slice-quantity','slice-zoom-quantity']){const el=$(id);if(el)el.addEventListener('change',()=>setSliceDisplay({quantity:el.value}));}
  for(const id of ['slice-arrows','slice-zoom-arrows']){const el=$(id);if(el)el.addEventListener('change',()=>setSliceDisplay({arrows:el.checked}));}
  {const el=$('velocity-log');if(el)el.addEventListener('change',()=>setSliceDisplay({log:el.checked}));}
  {const b=$('slice-zoom-open');if(b)b.addEventListener('click',()=>openSliceZoom(true));}
  {const b=$('slice-zoom-close');if(b)b.addEventListener('click',()=>openSliceZoom(false));}
  {const z=$('slice-zoom');if(z)z.addEventListener('click',event=>{if(event.target===z)openSliceZoom(false);});}
  {const c=$('slice-zoom-points');if(c)c.addEventListener('change',renderSliceZoom);}
  {const c=$('slice-zoom-fill');if(c)c.addEventListener('change',()=>{if($('slice-fill'))$('slice-fill').checked=c.checked;refresh();});}
  {const b=$('slice-zoom-png');if(b)b.addEventListener('click',()=>{const canvas=$('slice-zoom-canvas');if(!canvas||!canvas.toDataURL||!sliceLast)return;const c=common();const name=c&&typeof c.exportFilename==='function'?c.exportFilename({case_id:meta.case_id,view:'slice',field:sliceLast.quantity==='normal'?'through_plane':sliceLast.field,scale:1,ext:'png'}):`${meta.case_id||'case'}_slice_${sliceLast.field}.png`;download(name,canvas.toDataURL('image/png'));});}
  {const b=$('slice-zoom-csv');if(b)b.addEventListener('click',()=>{exportSliceCSV();});}
  {const b=$('slice-series-export');if(b)b.addEventListener('click',()=>{exportSliceSeries();});}
  {const b=$('montage-export');if(b)b.addEventListener('click',()=>{exportMontage();});}
  {const b=$('profile-svg');if(b)b.addEventListener('click',()=>{exportProfileSVG();});}
  {const b=$('onepage-shots');if(b)b.addEventListener('click',()=>{makeOnepageShots();});}
  {const c=$('slice-zoom-canvas');if(c)c.addEventListener('mousemove',event=>sliceZoomReadout(event.clientX,event.clientY));}
  const sliceCollapse=$('slice-panel-collapse');if(sliceCollapse)sliceCollapse.addEventListener('click',()=>{const body=$('slice-panel-body');if(!body)return;body.hidden=!body.hidden;sliceBodyUser=body.hidden?'closed':'open';sliceCollapse.textContent=body.hidden?'展开':'收起';});
  // ---- compact layout: a narrow window or a compare-page iframe.  The side menu uses an adaptive column
  // when there is room and falls back to an overlay on very narrow screens; the right dock shrinks to a
  // small legend + a probe strip along the bottom, while header and footer lose their padding.
  const COMPACT_BREAK=1180,SHORT_LABELS={'pick-toggle':['点选定位截面','点选'],'volume-fit':['复位视角','复位'],'volume-save':['保存截图','截图'],'volume-print':['打印 / 保存 PDF','打印']};
  try{const v=root.localStorage&&root.localStorage.getItem('wss-report-compact');if(v==='on'||v==='off')compactOverride=v;}catch(_){}
  function menuEl(){return document.querySelector?document.querySelector('aside.menu'):null;}
  function setMenuOpen(open){const aside=menuEl();if(!aside||!aside.classList)return;aside.classList.toggle('open',Boolean(open));const main=aside.closest&&aside.closest('main');if(main&&main.classList)main.classList.toggle('menu-open',Boolean(open));const b=$('menu-toggle');if(b&&b.setAttribute)b.setAttribute('aria-expanded',String(Boolean(open)));if(typeof root.dispatchEvent==='function'&&typeof root.Event==='function')root.dispatchEvent(new root.Event('resize'));}
  function applyCompact(force){
    const want=compactOverride==='on'?true:compactOverride==='off'?false:(Number(root.innerWidth)||1e4)<COMPACT_BREAK;
    if(!force&&want===compactOn)return;compactOn=want;
    const aside=menuEl();if(aside&&aside.style){aside.style.transition='none';setTimeout(()=>{try{aside.style.removeProperty('transition');}catch(_){ }},0);}
    if(document.body&&document.body.classList)document.body.classList.toggle('compact',compactOn);
    for(const [id,[long,short]] of Object.entries(SHORT_LABELS)){const b=$(id);if(!b)continue;if(!(id==='pick-toggle'&&pickMode))b.textContent=compactOn?short:long;if(b.setAttribute)b.setAttribute('title',long);}
    const lt=$('layout-toggle');if(lt)lt.textContent=compactOn?'完整布局':'紧凑布局';
    setMenuOpen(false);
    const body=$('slice-panel-body');if(body&&sliceBodyUser===null){body.hidden=compactOn;if(sliceCollapse)sliceCollapse.textContent=body.hidden?'展开':'收起';}
    if(started)refresh();   // the view note is shorter in the compact layout
  }
  {const b=$('menu-toggle');if(b)b.addEventListener('click',()=>{const aside=menuEl();setMenuOpen(!(aside&&aside.classList&&aside.classList.contains('open')));});}
  {const b=$('menu-tab');if(b)b.addEventListener('click',()=>setMenuOpen(true));}
  {const b=$('menu-close');if(b)b.addEventListener('click',()=>setMenuOpen(false));}
  root.addEventListener('keydown',event=>{if(event.key==='Escape'&&compactOn&&!event.defaultPrevented)setMenuOpen(false);});
  {const b=$('layout-toggle');if(b)b.addEventListener('click',()=>{compactOverride=compactOn?'off':'on';try{root.localStorage&&root.localStorage.setItem('wss-report-compact',compactOverride);}catch(_){}applyCompact(true);});}
  root.addEventListener('resize',()=>applyCompact(false));
  applyCompact(true);
  function snapshot() {
    if(!renderer)return null;renderer.render(scene,camera);
    const canvas=document.createElement('canvas');canvas.width=renderer.domElement.width;canvas.height=renderer.domElement.height;
    const ctx=canvas.getContext('2d');ctx.drawImage(renderer.domElement,0,0);const scale=canvas.width/Math.max(view.clientWidth,1);ctx.save();ctx.scale(scale,scale);ctx.fillStyle='#ffffffee';ctx.fillRect(12,12,320,65);ctx.fillStyle='#20374d';ctx.font='14px sans-serif';ctx.fillText($('legend-title').textContent,22,34);ctx.fillText($('legend-min').textContent+' — '+$('legend-max').textContent,22,57);ctx.restore();return canvas.toDataURL('image/png');
  }
  function saveSnapshot() {
    const url=snapshot();if(!url){setText('export-status','三维视图不可用，无法保存截图。');return null;}
    const name=`${safeFile(caseName())}_view_${fileFieldKey()}.png`;download(name,url);setText('export-status',`已保存截图 ${name}。`);return name;
  }
  $('volume-save').addEventListener('click',saveSnapshot);
  // ---------------------------------------------------------------- v0.12 (§19.9, B4, A9)
  // B4: a thin closable yellow banner on top of the viewport when the geometry leaves the release's reference
  // range or the ensemble quality is not good; the full list sits in 「统计与口径 → 参考范围」.
  const warnings=warningItems(meta);
  function renderWarnings() {
    const text=warningText(warnings),banner=$('warn-banner');
    setText('warn-banner-text',text);
    {const el=$('warn-banner-text');if(el&&el.setAttribute)el.setAttribute('title',warnings.map(x=>x.text).join('\n'));}
    if(banner)banner.hidden=!text;
    if(view&&view.classList)view.classList.toggle('has-banner',Boolean(text));
    const ra=meta.reference_assessment&&typeof meta.reference_assessment==='object'?meta.reference_assessment:null;
    setHidden('reference-card',!ra&&!warnings.length);
    const status=ra?{pass:'输入几何在本发布包声明的参考范围内。',review:'部分几何测量超出本发布包声明的参考范围，请复核。',unknown:'部分几何测量缺失或本发布包未声明参考范围，无法完成范围检查。'}[ra.status]||'参考范围检查状态未知。':'本报告没有参考范围检查。';
    setText('reference-status',status+(ra&&ra.note?' '+ra.note:''));
    const list=$('reference-list');
    if(list&&list.replaceChildren){list.replaceChildren();for(const w of warnings){const li=document.createElement('li');li.textContent=w.text;list.appendChild(li);}}
  }
  function closeWarnings() {const banner=$('warn-banner');if(banner)banner.hidden=true;if(view&&view.classList)view.classList.remove('has-banner');}
  {const b=$('warn-banner-close');if(b)b.addEventListener('click',closeWarnings);}
  {const b=$('warn-banner-more');if(b)b.addEventListener('click',()=>{const card=$('menu-stats');if(card)card.open=true;if(compactOn)setMenuOpen(true);const ref=$('reference-card');if(ref&&ref.scrollIntoView){try{ref.scrollIntoView({block:'nearest'});}catch(_){}}});}
  // P: the hover / click probe on or off (a hovering read-out can hide the vessel when reading figures).
  function setProbeEnabled(on) {
    probeEnabled=Boolean(on);
    if(!probeEnabled){probe=null;probePinned=false;}
    renderProbe();refresh();
    return probeEnabled;
  }
  // L: automatic labels on / off; turning them back on restores the last choice (default: top 5 + branch names).
  let labelsRemembered={findings:5,branches:true};
  function toggleAutoLabels() {
    const on=labelState.findings>0||labelState.branches;
    if(on){labelsRemembered={findings:labelState.findings,branches:labelState.branches};setLabels({findings:0,branches:false});}
    else setLabels(labelsRemembered.findings>0||labelsRemembered.branches?labelsRemembered:{findings:5,branches:true});
    drawMarkers();
    return {...labelState};
  }
  // 技术信息 popover (footer button and the 统计与口径 menu, which stays reachable in the compact layout).
  for(const id of ['tech-info-toggle','tech-info-menu']){const b=$(id);if(b)b.addEventListener('click',ev=>{if(ev&&ev.stopPropagation)ev.stopPropagation();setTechInfo(!techOpen);});}
  {const b=$('tech-info-close');if(b)b.addEventListener('click',()=>setTechInfo(false));}
  {const b=$('tech-info-copy');if(b)b.addEventListener('click',()=>{
    const text=techInfoText(),done=()=>setText('tech-info-status','已复制到剪贴板。');
    if(root.navigator&&root.navigator.clipboard&&root.navigator.clipboard.writeText)root.navigator.clipboard.writeText(text).then(done,()=>setText('tech-info-status','剪贴板不可用。'));
    else setText('tech-info-status','剪贴板不可用。');
  });}
  if(typeof document.addEventListener==='function') {
    document.addEventListener('click',event=>{
      if(!techOpen)return;const t=event&&event.target;
      if(t&&t.closest&&(t.closest('#tech-info')||t.closest('.gloss-pop')||t.closest('#tech-info-toggle')||t.closest('#tech-info-menu')))return;
      setTechInfo(false);
    });
    document.addEventListener('keydown',event=>{if(event.key==='Escape'&&techOpen&&!event.defaultPrevented){setTechInfo(false);if(event.preventDefault)event.preventDefault();}});
  }
  // A9 keyboard shortcuts (§19.9).  Arrow keys / PgUp / PgDn / [ ] stay with the slice gizmo handler; they are
  // listed in the help overlay only.  Nothing fires while the slice zoom view is open, except ? / Esc.
  function shortcutBindings() {
    const free=fn=>ev=>!sliceZoomOpen&&(!fn||fn(ev));
    const views=['front','back','left','right','top','bottom'].map((name,i)=>({keys:[String(i+1)],label:`${STANDARD_VIEWS[name].label}视（撑满视口）`,group:'视角',
      run:()=>showStandardView(name),when:free(()=>Boolean(frame&&camera))}));
    const info=(keys,label)=>({keys,label,group:'截面（截面页签下）',run(){},when:()=>false});
    return views.concat([
      {keys:['0','r'],label:'复位视角（解剖前视）',group:'视角',run:fit,when:free(()=>Boolean(camera))},
      {keys:['v'],label:'物理量：速度',group:'显示',run:()=>setField('velocity'),when:free(()=>Boolean(speed))},
      {keys:['b'],label:'物理量：压力',group:'显示',run:()=>setField('pressure'),when:free(()=>Boolean(pressure))},
      {keys:['t'],label:'循环显示页签（点云 → 截面 → 壁面压力 → 流线）',group:'显示',run:cycleMode,when:free()},
      {keys:['l'],label:'自动标注开 / 关（发现与分支名）',group:'显示',run:toggleAutoLabels,when:free()},
      {keys:['p'],label:'探针开 / 关（悬停读数）',group:'显示',run:()=>setProbeEnabled(!probeEnabled),when:free()},
      {keys:['s'],label:'保存截图（PNG）',group:'导出',run:saveSnapshot,when:free(()=>Boolean(renderer))},
      {keys:['x'],label:'点选定位截面 开 / 关',group:'截面（截面页签下）',run:()=>setPickMode(!pickMode),when:free(()=>Boolean(renderer))},
      info(['ArrowUp','ArrowDown'],'截面沿法向移动（Shift 5 mm）'),info(['ArrowLeft','ArrowRight'],'截面绕纵轴旋转'),
      info(['PageUp','PageDown'],'截面绕横轴倾斜'),info(['[',']'],'截面变薄 / 变厚'),info(['Escape'],'结束点选 / 关闭放大图与弹层')]);
  }
  function installKeys() {
    const c=common();
    if(!c||typeof c.installShortcuts!=='function')return null;
    try{return c.installShortcuts(shortcutBindings(),{doc:document,title:'体场报告快捷键',helpGroup:'帮助'});}catch(_){return null;}
  }
  {const b=$('shortcuts-help');if(b)b.addEventListener('click',()=>{if(shortcuts&&shortcuts.showHelp)shortcuts.showHelp();});}
  $('volume-print').addEventListener('click',()=>{const url=snapshot();if(url)$('volume-snapshot').src=url;root.print();});
  root.addEventListener('beforeprint',()=>{const url=snapshot();if(url)$('volume-snapshot').src=url;});
  if(lines.length>320&&$('streamline-density'))$('streamline-density').value='2';
  // ---- remembered report defaults (C6): #view= link > server preferences > localStorage > built-in ----
  function currentDefaults() {return {colormap:currentMap,bands:currentBands,pressure_units:pressureUnit,speed_units:velocityUnit,opacity:num('volume-opacity',.1),lang};}
  function applyDefaults(d) {
    if(!d||typeof d!=='object')return;
    if(d.colormap&&COLORMAPS[d.colormap]){setColormap(d.colormap);setVal('colormap',currentMap);}
    if(d.bands!==undefined){setBands(d.bands);setVal('color-bands',currentBands);}
    if(UNITS.pressure[d.pressure_units]){pressureUnit=d.pressure_units;setVal('pressure-unit',pressureUnit);}
    if(UNITS.velocity[d.speed_units]){velocityUnit=d.speed_units;setVal('velocity-unit',velocityUnit);}
    if(Number.isFinite(Number(d.opacity))&&d.opacity!==null&&d.opacity!=='')setVal('volume-opacity',d.opacity);
    if(d.lang==='en'||d.lang==='zh'){lang=d.lang;setVal('export-lang',lang);}
  }
  function loadLocalDefaults() {try{const text=root.localStorage&&root.localStorage.getItem('wss-report-defaults');const doc=text?JSON.parse(text):null;return doc&&doc.volume||null;}catch(_){return null;}}
  async function loadServerDefaults() {
    if(!online)return null;
    try{const r=await fetch(online.root+'api/preferences',{credentials:'same-origin'});if(!r.ok)return null;const j=await r.json();const p=j&&j.preferences||{};return p.report_defaults&&p.report_defaults.volume||null;}catch(_){return null;}
  }
  async function saveDefaults() {
    const d=currentDefaults();
    try{const text=root.localStorage&&root.localStorage.getItem('wss-report-defaults');const doc=text?JSON.parse(text):{};doc.volume=d;if(root.localStorage)root.localStorage.setItem('wss-report-defaults',JSON.stringify(doc));}catch(_){}
    if(!online){setText('defaults-status','已保存为本浏览器的默认口径。');return;}
    try {
      const r=await fetch(online.root+'api/preferences',{credentials:'same-origin'});const prefs=r.ok?((await r.json()).preferences||{}):{};
      prefs.schema_version=prefs.schema_version||'wss-deploy.preferences/v1';prefs.report_defaults=Object.assign({},prefs.report_defaults||{},{volume:d});
      const token=await csrf();
      const put=await fetch(online.root+'api/preferences',{method:'PUT',credentials:'same-origin',headers:{'Content-Type':'application/json','X-CSRF-Token':token},body:JSON.stringify(prefs)});
      setText('defaults-status',put.ok?'已保存为账户默认口径（服务器）。':'服务器保存失败，已保存到本浏览器。');
    } catch(_){setText('defaults-status','服务器保存失败，已保存到本浏览器。');}
  }
  // §15.10 / §15.11 / §15.8 controls: the series branch list mirrors the slice branches; the montage
  // starts on 前 / 左 / 当前 / 截面 and drops the standard views when there is no anatomical frame.
  groups.forEach(g=>option($('slice-series-branch'),g.segment,branchName(g.segment)));
  if(!groups.length){for(const id of ['slice-series-branch','slice-series-count','slice-series-export']){const el=$(id);if(el)el.disabled=true;}
    setText('slice-series-note','本报告没有可用中心线，不能生成截面系列。');}
  for(const name of MONTAGE_VIEWS) {
    const box=$('mv-'+name);if(!box)continue;
    box.checked=MONTAGE_DEFAULT.includes(name);
    if(!frame&&name!=='current'&&name!=='slice'){box.checked=false;box.disabled=true;}
  }
  setVal('montage-columns','2');
  if(!online){const b=$('onepage-shots');if(b)b.disabled=true;setOnepageStatus('离线报告不上传配图；在工作台中打开本报告后可用。',false);}
  applyDefaults(loadLocalDefaults());writeExportOptions();
  presetList=loadLocalPresets();
  renderBranchVisibility();renderProbeLog();renderFindings();renderMeasurements();renderPresets();
  writeLabelControls();renderMorphologyRow();renderNarrative();renderWarnings();setTechInfo(false);
  shortcuts=installKeys();setHidden('shortcuts-row',!shortcuts);
  refresh();drawMarkers();started=true;
  const hashState=viewFromHash();
  if(hashState){try{applyView(hashState);setText('view-status','已按链接复现视图。');}catch(_){}}
  else loadServerDefaults().then(d=>{if(d&&!viewFromHash()){applyDefaults(d);refresh();}});
  loadFindingsReview();loadAnnotations();loadServerPresets();
  postParent({type:'wss-view:ready',family:'volume',run_identity:meta.run_identity||null,case_id:meta.case_id||null,webgl:Boolean(renderer)});
  setTimeout(()=>{announceOn=true;},0);
  // Node-only hook: picks and pinning need WebGL in a browser, so tests drive the handlers with world coordinates.
  if(typeof module!=='undefined'&&module.exports) module.exports.__test={
    setMeasureMode,measurePick:addMeasurePick,measure:(kind,points)=>{measureMode=null;setMeasureMode(kind);let out=null;for(const p of points)out=addMeasurePick(p);return out;},
    measurements:()=>measurements.map(m=>({...m})),measureLabel,measureTSV,
    setAnnotMode,addAnnotation,annotations:()=>annotations.map(a=>({...a})),
    presetNames:()=>builtinPresetList().map(p=>p.name),applyPreset:applyPresetByName,savePreset,userPresets:()=>presetList.map(p=>({...p})),
    colorbarSVG:language=>colorbarSvgCurrent(language||lang),fieldLabel,branchLabel,viewLabel,labelItems,drawOverlayLabels,
    probeTSV:(rows,language)=>probeSerializer('tsv')(rows,language||lang),
    centerlineGroups:()=>clGroups.length,capture:captureView,apply:applyView,
    sliceCSV,exportSliceCSV,sliceSection:()=>sliceSection&&{...sliceSection},sectionRows,
    profileSVGs:buildProfileSVGs,exportProfileSVG,exportSliceSeries,exportMontage,makeOnepageShots,
    montageSelection,seriesFractions,sliceSeries:()=>sliceSeries&&{...sliceSeries,fractions:sliceSeries.fractions.slice()},
    labels:()=>({...labelState}),setLabels,findingChipText,activateFinding,findings:()=>currentFindings().map(x=>({...x})),
    // v0.12 (§19.9 / A9 / B4)
    camera:()=>cameraState(),autoView:()=>autoView,fit,showStandardView,fittedCamera,setField,setMode,cycleMode,
    probeEnabled:()=>probeEnabled,setProbeEnabled,toggleAutoLabels,setTechInfo,techOpen:()=>techOpen,techInfoText,
    shortcuts:()=>shortcuts,warnings:()=>warnings.map(x=>({...x})),pickMode:()=>pickMode,
    sliceZoom:open=>{openSliceZoom(open);return sliceZoomOpen;},
    // §21.4 label layout
    layoutLabels:()=>layoutLabelsNow().map(x=>({kind:x.item.kind,severity:x.item.severity||null,text:x.item.text,x:x.x,y:x.y,w:x.w,h:x.h,anchor:x.anchor,moved:x.moved,hidden:x.hidden})),
    labelsDirty:()=>labelsDirty,cameraKey,
    // slice readability (用户试用反馈 7)
    setSliceDisplay,sliceScale:()=>sliceScaleNow&&{...sliceScaleNow},legendScale:()=>legendScaleNow&&{...legendScaleNow},
    sliceInfo:()=>sliceLast&&{quantity:sliceLast.quantity,arrows:sliceLast.arrows||0,plane:{origin:sliceLast.plane.origin,normal:sliceLast.plane.normal},
      samples:sliceLast.data.finite.map(x=>({i:x.i,v:x.v}))},zoomArrows:()=>sliceZoomLast?sliceZoomLast.arrows:0,
    morphMax:()=>morphMax&&{max_diameter_mm:Number(morphMax.max_diameter_mm),ring:Boolean(morphMax.polygon_world),xyz:morphMax.xyz_mm},
    morphSeries:branchIndex=>{const b=profileBranches[Number(branchIndex)||0]||null;const m=morphSeriesFor(b);return m&&{x:m.x,max:m.max,equiv:m.equiv,offset_mm:m.offset_mm};}};
})(typeof globalThis!=='undefined'?globalThis:this);
