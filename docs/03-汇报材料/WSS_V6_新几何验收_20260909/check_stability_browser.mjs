// Read-only check of the saved historical-top-six recheck viewer.
import fs from 'node:fs';
const [url, screenshot, port = '9231', expectedVersion = 'v1.3'] = process.argv.slice(2);
const target = await (await fetch(`http://127.0.0.1:${port}/json/new?about:blank`, {method:'PUT'})).json();
const socket = new WebSocket(target.webSocketDebuggerUrl);
await new Promise(resolve => socket.addEventListener('open', resolve, {once:true}));
let token=0;
const pending=new Map(), errors=[];
socket.addEventListener('message', event=>{
  const message=JSON.parse(event.data);
  if(message.method==='Runtime.exceptionThrown') errors.push(message.params.exceptionDetails);
  if(message.id&&pending.has(message.id)) {
    const {resolve,reject}=pending.get(message.id);pending.delete(message.id);
    message.error?reject(message.error):resolve(message.result);
  }
});
function send(method,params={}) {return new Promise((resolve,reject)=>{const id=++token;pending.set(id,{resolve,reject});socket.send(JSON.stringify({id,method,params}));});}
async function evaluate(expression) {const r=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;}
const pause=ms=>new Promise(resolve=>setTimeout(resolve,ms));
await send('Runtime.enable');await send('Page.enable');
await send('Emulation.setDeviceMetricsOverride',{width:1440,height:1100,deviceScaleFactor:1,mobile:false});
await send('Page.navigate',{url});
for(let i=0;i<50;i++){await pause(300);if(await evaluate('document.readyState==="complete"&&!!document.querySelector("#space canvas")'))break;}
const metadata=await evaluate('({version:D.candidate_version,appliedGate:D.manifest.applied_gate_schema,diagnostics:D.diagnostics.length,notice:document.querySelector("header").textContent})');
if(metadata.version!==expectedVersion||!metadata.appliedGate.endsWith('_'+expectedVersion))errors.push({message:'Version mismatch',metadata});
const checks=[];
for(let i=0;i<metadata.diagnostics;i++){
  await evaluate(`render(${i});true`);await pause(300);
  const result=await evaluate(`(()=>{const r=D.diagnostics[${i}],p=r.perturbed;return {rank:r.rank,canonical_id:r.canonical_id,selectedIndex:+document.getElementById('select').value,baseline_valid:r.baseline.pipeline_valid,perturbed_valid:p.pipeline_valid,perturbed_reason:p.pipeline_reason,continuous_crossing_groups:p.same_segment_crossing_group_count,nonlocal_groups:p.same_segment_nonlocal_crossing_count,crossings:p.same_segment_plane_intersections,plotCrossingMarkers:document.getElementById('selected').data[3].x.length,perturbed_color:document.getElementById('selected').data[1].line.color,plots:document.querySelectorAll('.js-plotly-plot').length}})()`);
  if(result.selectedIndex!==i||result.plots!==3||(!result.perturbed_valid&&result.perturbed_color!=='#94a3b8'))errors.push({message:'Plot state/mask inconsistency',result});
  if(result.plotCrossingMarkers!==result.crossings.inside_crossing_groups.reduce((n,g)=>n+g.points_xyz_mm.length,0))errors.push({message:'Crossing coordinate marker mismatch',result});
  checks.push(result);
}
await evaluate('render(2);window.scrollTo(0,document.getElementById("metrics").getBoundingClientRect().top+window.scrollY-50);true');
await pause(600);
if(screenshot){const r=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});fs.writeFileSync(screenshot,Buffer.from(r.data,'base64'));}
const result={url,metadata,checks,runtimeErrors:errors,pass:errors.length===0};
console.log(JSON.stringify(result,null,2));
await send('Page.close');socket.close();
if(!result.pass)process.exitCode=1;
