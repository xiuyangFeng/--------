// Read-only navigation and image check for the final priority entry point.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const here=path.dirname(fileURLToPath(import.meta.url));
const expected=JSON.parse(fs.readFileSync(path.join(here,'priority_review_manifest.json'),'utf8'));
const [port='9231']=process.argv.slice(2);
const target=await (await fetch(`http://127.0.0.1:${port}/json/new?about:blank`,{method:'PUT'})).json();
const socket=new WebSocket(target.webSocketDebuggerUrl);
await new Promise(resolve=>socket.addEventListener('open',resolve,{once:true}));
let token=0;const pending=new Map(),errors=[];
socket.addEventListener('message',event=>{const m=JSON.parse(event.data);if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails);if(m.id&&pending.has(m.id)){const {resolve,reject}=pending.get(m.id);pending.delete(m.id);m.error?reject(m.error):resolve(m.result);}});
function send(method,params={}){return new Promise((resolve,reject)=>{const id=++token;pending.set(id,{resolve,reject});socket.send(JSON.stringify({id,method,params}));});}
async function evaluate(expression){const r=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;}
const pause=ms=>new Promise(resolve=>setTimeout(resolve,ms));
await send('Runtime.enable');await send('Page.enable');
await send('Emulation.setDeviceMetricsOverride',{width:1440,height:1100,deviceScaleFactor:1,mobile:false});
await send('Page.navigate',{url:new URL('./index.html',import.meta.url).href});
await pause(700);
const index=await evaluate(`({firstPanel:document.querySelector('main').firstElementChild.textContent,caseLinks:new Set([...document.querySelectorAll('a[href^="cases/"]')].map(a=>a.getAttribute('href'))).size,priorityLink:!!document.querySelector('a[href="priority_review.html"]')})`);
if(!index.priorityLink||!index.firstPanel.includes(`${expected.priority_union_cases} 例`)||index.caseLinks!==172)errors.push({message:'Index first panel or case count mismatch',index});
const shot=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
fs.writeFileSync(path.join(here,'priority_index_v1.3.png'),Buffer.from(shot.data,'base64'));
await evaluate(`document.querySelector('a[href="priority_review.html"]').click();true`);
await pause(800);
const priority=await evaluate('({url:location.href,rows:document.querySelectorAll("#section-table tbody tr").length,text:document.body.textContent,images:[...document.images].map(i=>({src:i.getAttribute("src"),width:i.naturalWidth})),links:[...document.links].map(a=>a.getAttribute("href"))})');
if(!priority.url.endsWith('/priority_review.html')||priority.rows!==expected.section_disagreements||!priority.text.includes(expected.source_candidate_schema)||priority.images.length!==4||priority.images.some(i=>i.width===0)||!priority.links.includes('priority_stability_v1.3.html'))errors.push({message:'Priority page content/images mismatch'});
const missingLinks=priority.links.filter(href=>!href.includes('://')&&!href.startsWith('#')&&!fs.existsSync(path.resolve(here,decodeURIComponent(href.split('#')[0]))));
if(missingLinks.length)errors.push({message:'Missing local links',missingLinks});
const search=await evaluate('document.getElementById("search").value="YIN_YU_RONG";document.getElementById("search").dispatchEvent(new Event("input"));[...document.querySelectorAll("#section-table tbody tr")].filter(r=>!r.hidden).map(r=>r.textContent)');
if(!search.length||search.some(t=>!t.includes('YIN_YU_RONG')))errors.push({message:'Search mismatch',search});
delete priority.text;
console.log(JSON.stringify({expected,index,priority,search,runtimeErrors:errors,pass:errors.length===0},null,2));
await send('Page.close');socket.close();if(errors.length)process.exitCode=1;
