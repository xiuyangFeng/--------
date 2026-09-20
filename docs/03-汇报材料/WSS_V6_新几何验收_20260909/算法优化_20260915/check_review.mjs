import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL,fileURLToPath} from 'node:url';
const base=path.resolve(process.argv[2]||path.dirname(fileURLToPath(import.meta.url)));
const port=process.argv[3]||9337;
const tabs=await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
const tab=tabs.find(t=>t.type==='page');
const ws=new WebSocket(tab.webSocketDebuggerUrl);
await new Promise((res,rej)=>{ws.onopen=res;ws.onerror=rej});
let id=0;const pending=new Map(),errors=[];
ws.onmessage=evt=>{const m=JSON.parse(evt.data);if(m.id){let p=pending.get(m.id);pending.delete(m.id);m.error?p.reject(m.error):p.resolve(m.result)}else if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails)};
function call(method,params={}){return new Promise((resolve,reject)=>{let n=++id;pending.set(n,{resolve,reject});ws.send(JSON.stringify({id:n,method,params}));setTimeout(()=>{if(pending.has(n)){pending.delete(n);reject(new Error('timeout '+method))}},30000).unref()})}
async function js(expression){let r=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw new Error(JSON.stringify(r.exceptionDetails));return r.result.value}
await call('Page.enable');await call('Runtime.enable');
await call('Emulation.setDeviceMetricsOverride',{width:1500,height:1050,deviceScaleFactor:1,mobile:false});
await call('Page.navigate',{url:pathToFileURL(path.join(base,'index.html')).href});
let loaded=false;
for(let i=0;i<60;i++){await new Promise(r=>setTimeout(r,250));try{loaded=await js(`typeof D!=='undefined' && D.length===170 && !!document.getElementById('curve').data`)}catch{}if(loaded)break}
if(!loaded)throw new Error('review page did not load');
const initial=await js(`({manual_options:document.getElementById('case').options.length,expected:S.manual_case_count,curve_traces:document.getElementById('curve').data.length,cases:D.length})`);
if(initial.manual_options!==initial.expected)throw new Error('manual list mismatch');
await new Promise(r=>setTimeout(r,400));
let shot=await call('Page.captureScreenshot',{format:'png'});fs.writeFileSync(path.join(base,'review_preview.png'),Buffer.from(shot.data,'base64'));
await js(`document.getElementById('all').click();document.getElementById('case').value='AG__slow__YIN_YU_RONG';show();document.getElementById('event').value=String(cur.events.findIndex(e=>e.station===74));event();`);
await new Promise(r=>setTimeout(r,400));
const recovered=await js(`({all_options:document.getElementById('case').options.length,station:cur.events[Number(document.getElementById('event').value)],traces:document.getElementById('contour').data.length})`);
if(recovered.all_options!==170||recovered.traces!==2||recovered.station.status!=='recovered_stable_plane')throw new Error('recovered-case interaction failed');
shot=await call('Page.captureScreenshot',{format:'png'});fs.writeFileSync(path.join(base,'recovered_YIN_YU_RONG_74.png'),Buffer.from(shot.data,'base64'));
await call('Page.navigate',{url:pathToFileURL(path.join(base,'remaining_bifurcation_review.html')).href});
for(let i=0;i<40;i++){await new Promise(r=>setTimeout(r,100));if(await js(`document.querySelectorAll('canvas').length===4`))break}
const bifurcation=await js(`({cards:document.querySelectorAll('section.card').length,canvases:[...document.querySelectorAll('canvas')].map(c=>({width:c.width,height:c.height})),links:[...document.querySelectorAll('a')].map(a=>a.href),before:document.getElementById('view0').toDataURL()})`);
await js(`document.getElementById('wall0').click()`);
const toggled=await js(`document.getElementById('view0').toDataURL()`);
bifurcation.toggle_changed_image=toggled!==bifurcation.before;delete bifurcation.before;
bifurcation.all_links_exist=bifurcation.links.every(url=>fs.existsSync(fileURLToPath(url)));
if(bifurcation.cards!==2||!bifurcation.toggle_changed_image||!bifurcation.all_links_exist)throw new Error('bifurcation review failed');
await js(`document.getElementById('wall0').click()`);
shot=await call('Page.captureScreenshot',{format:'png'});fs.writeFileSync(path.join(base,'bifurcation_review_preview.png'),Buffer.from(shot.data,'base64'));
const result={passed:errors.length===0,initial,recovered:{station:recovered.station.station,old_area:recovered.station.old_area,new_area:recovered.station.new_area,normal_change_deg:recovered.station.angle,plot_traces:recovered.traces},bifurcation,js_errors:errors};
fs.writeFileSync(path.join(base,'browser_validation.json'),JSON.stringify(result,null,2)+'\n');
ws.close();console.log(JSON.stringify(result));if(errors.length)process.exitCode=1;
