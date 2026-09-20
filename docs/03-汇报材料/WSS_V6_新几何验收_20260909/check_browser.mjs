// Read-only smoke check against a locally launched Chromium CDP endpoint.
// Usage: node check_browser.mjs file:///absolute/cases/CASE.html screenshot.png [port]
import fs from 'node:fs';
const [url, screenshot, port = '9231', screenshotTab = 'overview', expectation = 'candidate'] = process.argv.slice(2);
const [screenshotPanel, sectionText] = screenshotTab.split(':');
const requestedSection = sectionText === undefined ? null : Number(sectionText);
if (!url) throw new Error('URL required');
const target = await (await fetch(`http://127.0.0.1:${port}/json/new?about:blank`, {method: 'PUT'})).json();
const socket = new WebSocket(target.webSocketDebuggerUrl);
await new Promise(resolve => socket.addEventListener('open', resolve, {once: true}));
let id = 0;
const pending = new Map();
const errors = [];
socket.addEventListener('message', event => {
  const message = JSON.parse(event.data);
  if (message.method === 'Runtime.exceptionThrown') errors.push(message.params.exceptionDetails);
  if (message.id && pending.has(message.id)) {
    const {resolve, reject} = pending.get(message.id);
    pending.delete(message.id);
    message.error ? reject(message.error) : resolve(message.result);
  }
});
function send(method, params = {}) {
  return new Promise((resolve, reject) => {
    const token = ++id;
    pending.set(token, {resolve, reject});
    socket.send(JSON.stringify({id: token, method, params}));
  });
}
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
async function evaluate(expression) {
  const result = await send('Runtime.evaluate', {expression, returnByValue: true, awaitPromise: true});
  if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
  return result.result.value;
}
await send('Runtime.enable');
await send('Page.enable');
await send('Emulation.setDeviceMetricsOverride', {width: 1440, height: 1100, deviceScaleFactor: 1, mobile: false});
await send('Page.navigate', {url});
for (let i = 0; i < 60; i++) {
  await pause(500);
  if (await evaluate('document.readyState === "complete" && !!document.querySelector("#overview-plot .gl-container canvas")')) break;
}
const checks = {};
for (const tab of ['overview', 'sections', 'profiles', 'mapping', 'stability', 'coverage']) {
  await evaluate(`showTab(${JSON.stringify(tab)})`);
  await pause(1200);
  checks[tab] = await evaluate(`({visible:!document.getElementById(${JSON.stringify(tab)}).hidden,plots:document.getElementById(${JSON.stringify(tab)}).querySelectorAll('.js-plotly-plot').length})`);
}
const sectionCount = await evaluate('D.sections.length');
const candidate = await evaluate('({canonical_id:D.canonical_id,schema:D.report.schema,config_sha256:D.report.config_sha256,preview_only:D.report.preview_only,sections:D.sections.length,validContourCount:D.sections.filter(s=>s.valid&&s.selected.length>=3).length,rawIntersectionCount:D.sections.filter(s=>s.raw.length>=2).length})');
if (expectation !== 'preview' && (candidate.preview_only || !candidate.validContourCount || !candidate.rawIntersectionCount)) {
  errors.push({message: 'Expected real candidate with valid contours and intersection coordinates', candidate});
}
if (expectation.startsWith('v') && !candidate.schema?.includes(expectation)) errors.push({message:'Candidate schema version mismatch', expected:expectation, actual:candidate.schema});
if (sectionCount) {
  await evaluate('showTab("sections"); showSection(D.sections.length-1)');
  await pause(600);
  checks.lastSection = await evaluate('document.querySelector("#section-meta").textContent');
}
const masks = await evaluate(`(()=>{let s=D.sections.find(s=>!s.valid&&s.selected.length>=3);let check={invalidReferenceWithContour:!!s};if(s){showSection(s.index);check.referenceLineColor=document.getElementById('section2').data[1].line.color;check.selectorMatches=document.getElementById('section-select').value===String(s.index)}check.mapsHaveOnlyFiniteColoredValues=Object.values(D.maps).every(p=>p.data.every(t=>!Array.isArray(t.marker?.color)||t.marker.color.every(x=>x!==null&&(typeof x!=='number'||Number.isFinite(x)))));return check})()`);
if ((masks.invalidReferenceWithContour && (masks.referenceLineColor !== '#94a3b8' || !masks.selectorMatches)) || !masks.mapsHaveOnlyFiniteColoredValues) errors.push({maskCheck: masks});
await evaluate(`showTab(${JSON.stringify(screenshotPanel)}); if(${JSON.stringify(screenshotPanel)}==='sections' && D.sections.length){let valid=D.sections.filter(s=>s.valid&&s.selected.length>=3);showSection(${requestedSection === null ? 'valid[Math.floor(valid.length/2)].index' : requestedSection})} window.scrollTo(0, document.getElementById(${JSON.stringify(screenshotPanel)}).getBoundingClientRect().top + window.scrollY - 95)`);
const screenshotSection = screenshotPanel === 'sections' ? await evaluate('(()=>{let s=D.sections[+document.getElementById("section-select").value];return {index:s.index,segment:s.segment,reference_valid:s.valid,pointcloud_valid:s.pc_valid,reference_area_mm2:s.area_mm2,pointcloud_area_mm2:s.pc_area_mm2}})()') : null;
await pause(700);
if (screenshot) {
  const result = await send('Page.captureScreenshot', {format: 'png', captureBeyondViewport: false});
  fs.writeFileSync(screenshot, Buffer.from(result.data, 'base64'));
}
const result = {url, sectionCount, candidate, screenshotSection, masks, checks, runtimeErrors: errors, pass: errors.length === 0 && Object.values(checks).filter(x => typeof x === 'object').every(x => x.visible)};
console.log(JSON.stringify(result, null, 2));
await send('Page.close');
socket.close();
if (!result.pass) process.exitCode = 1;
