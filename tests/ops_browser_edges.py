"""Firefox regressions for operations/support sessions, drafts, races and narrow layouts.

Runs only against a temporary synthetic service; no production job directory or worker.
Usage: python -m tests.ops_browser_edges --out <validation-directory>
Requires the Firefox binary used by wss_deploy.devshot.
"""
import argparse
import os,json,socket,tempfile,time
from pathlib import Path
from tests._c_helpers import Service,finished
from tests._ops_browser_viewport import set_css_viewport
from wss_deploy.users import UserStore
from wss_deploy.devshot import Browser
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out', type=Path, required=True, help='Directory for the browser log and JSON results')
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
(args.out / 'browser-edges.json').unlink(missing_ok=True)
os.environ['WSS_DEPLOY_ALLOW_LEGACY_TOKEN']='1'
def wait(b,e,timeout=12):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        if b.js('return '+e):return
        time.sleep(.1)
    raise AssertionError(e+'\n'+str(b.js('return document.body.innerText')))
def click_text(b,selector,text):b.js("Array.from(document.querySelectorAll(arguments[0])).find(x=>x.textContent===arguments[1]).click()",[selector,text])
def view(b,name):
    b.js("document.querySelector('#ops-nav [data-view=\"'+arguments[0]+'\"]').click()",[name])
    wait(b,f"!document.getElementById('view-{name}').hidden")
def open_asset_review(b):
    b.js("let d=document.querySelector('#assets-rows details.ops-asset-review');if(d&&!d.open)d.querySelector('summary').click()")
results={}
with tempfile.TemporaryDirectory(prefix='wss-frontend-regression-') as t:
    root=Path(t);users=UserStore(root/'users.json');users.add('admin','admin-test-secret',admin=True);users.add('alice','alice-test-secret');users.add('bob','bob-test-secret')
    svc=Service(root,shared=True,token='legacy-test-secret',users=users);store=svc.server.operations
    try:
        job=finished(svc.manager,owner='alice',case_id='TEST',content=b'solid x\nendsolid x');asset=store.archive_asset(job_id=job['id'],owner='alice')
        ticket=store.create_ticket(owner='alice',title='Alpha target',description='Alpha description');store.create_ticket(owner='alice',title='Beta target',description='Beta description')
        for i in range(30):store.create_ticket(owner='alice',title=f'Paging {i}',description='Synthetic feedback')
        with socket.socket() as sk:sk.bind(('127.0.0.1',0));port=sk.getsockname()[1]
        with Browser(marionette_port=port,log_path=args.out / 'browser-edges.log') as b:
            base=f'http://127.0.0.1:{svc.port}';b.go(base+'/ops',wait=.3);wait(b,'!document.getElementById("ops-login").hidden')
            b.js("document.getElementById('login-mode').value='token';document.getElementById('login-mode').dispatchEvent(new Event('change'));document.dispatchEvent(new Event('visibilitychange'))")
            time.sleep(.4);assert b.js("return document.getElementById('login-mode').value==='token'");results['ops_token_choice_persists']=True
            b.js("document.getElementById('login-mode').value='password';document.getElementById('login-user').value='admin';document.getElementById('login-password').value='admin-test-secret';document.getElementById('ops-login-form').requestSubmit()")
            wait(b,"!document.getElementById('ops-content').hidden");view(b,'assets')
            wait(b,'document.querySelectorAll("#assets-rows input").length>0')
            open_asset_review(b)
            b.js("let el=document.querySelector('#assets-rows input');el.value='asset draft';el.dispatchEvent(new Event('input'));document.getElementById('assets-status').value='approved';document.getElementById('assets-filter').requestSubmit()")
            wait(b,"document.getElementById('assets-rows').textContent.includes('没有匹配')")
            b.js("document.getElementById('assets-status').value='';document.getElementById('assets-filter').requestSubmit()")
            wait(b,"document.querySelector('#assets-rows input')?.value==='asset draft'");results['asset_filter_preserves_hidden_draft']=True
            open_asset_review(b)
            store.review_asset(asset['sha256'],review_status='excluded',note='Other operator',version=asset['version'])
            click_text(b,'#assets-rows button','保存');wait(b,"document.getElementById('ops-notice').textContent.includes('草稿已保留')")
            click_text(b,'#assets-rows button','刷新版本');wait(b,"document.getElementById('ops-notice').textContent.includes('服务端状态')")
            assert b.js("return document.querySelector('#assets-rows input').value==='asset draft'")
            click_text(b,'#assets-rows button','保存');wait(b,"document.getElementById('ops-notice').textContent.includes('素材审阅已保存')");assert store.asset(asset['sha256'])['note']=='asset draft';results['asset_conflict_reload_and_save']=True
            view(b,'tickets');b.js("document.getElementById('tickets-query').value='Alpha';document.getElementById('tickets-filter').requestSubmit()")
            wait(b,"document.getElementById('tickets-rows').textContent.includes('Alpha target')");click_text(b,'#tickets-rows button','Alpha target')
            b.js("let e=document.querySelector('#ticket-detail textarea');e.value='Ops draft';e.dispatchEvent(new Event('input'));document.dispatchEvent(new Event('visibilitychange'))")
            time.sleep(.4);assert b.js("return document.querySelector('#ticket-detail textarea').value==='Ops draft'")
            store.update_ticket(ticket['id'],version=ticket['version'],status='in_progress',message='Concurrent update',actor='second_admin')
            click_text(b,'#ticket-detail button','保存处理');wait(b,"document.getElementById('ops-notice').textContent.includes('草稿已保留')")
            click_text(b,'#ticket-detail button','读取最新进展（保留草稿）');wait(b,"document.getElementById('ticket-detail').textContent.includes('Concurrent update')")
            assert b.js("return document.querySelector('#ticket-detail textarea').value==='Ops draft'&&document.querySelector('#ticket-detail select').value==='in_progress'")
            click_text(b,'#ticket-detail button','保存处理');wait(b,"document.querySelector('#ticket-detail .ticket-messages').textContent.includes('Ops draft')");results['ops_ticket_conflict_preserves_reply_and_latest_status']=True
            # Delay a background list response; newly focused editing field must survive that response.
            view(b,'assets')
            wait(b,"document.querySelector('#assets-rows input')");open_asset_review(b)
            b.js("window.originalFetch=window.fetch;window.delayStarted=false;window.fetch=async(...args)=>{if(String(args[0]).startsWith('/api/ops/assets?')){window.delayStarted=true;await new Promise(r=>setTimeout(r,600));}return originalFetch(...args);};document.dispatchEvent(new Event('visibilitychange'))")
            wait(b,'window.delayStarted');b.js("window.editField=document.querySelector('#assets-rows input');editField.focus();editField.value='In-flight draft';editField.dispatchEvent(new Event('input'));editField.setSelectionRange(2,2)")
            time.sleep(.9);assert b.js("return document.querySelector('#assets-rows input')===editField&&document.activeElement===editField&&editField.selectionStart===2&&editField.value==='In-flight draft'");results['background_response_preserves_focus_and_caret']=True
            b.js("window.fetch=originalFetch;window.scrollTo(0,document.documentElement.scrollHeight)")
            results['ops_mobile_viewports'] = []
            for width in [390,320]:
                viewport = set_css_viewport(b,width,844)
                assert viewport['innerWidth'] == width and viewport['innerHeight'] == 844
                assert viewport['scrollWidth'] <= viewport['clientWidth'], viewport
                results['ops_mobile_viewports'].append(viewport)
                assert b.js("let r=document.getElementById('ops-notice').getBoundingClientRect();return r.bottom<=innerHeight&&r.top>=0")
            results['ops_mobile_feedback_visible_320_390']=True
            set_css_viewport(b,1400,1000);b.go(base+'/support',wait=.3);b.js("document.getElementById('support-logout').click()");wait(b,'!document.getElementById("support-login").hidden')
            b.js("document.getElementById('support-mode').value='token';document.getElementById('support-mode').dispatchEvent(new Event('change'));document.dispatchEvent(new Event('visibilitychange'))")
            time.sleep(.4);assert b.js("return document.getElementById('support-mode').value==='token'");results['support_token_choice_persists']=True
            b.js("document.getElementById('support-mode').value='password';document.getElementById('support-username').value='alice';document.getElementById('support-password').value='alice-test-secret';document.getElementById('support-login-form').requestSubmit()")
            wait(b,'!document.getElementById("support-next").disabled');b.js("document.getElementById('support-next').click()");wait(b,"document.getElementById('support-page').textContent.startsWith('2 /')");results['support_paging']=True
            b.js("window.originalFetch=window.fetch;window.delayStarted=false;window.fetch=async (...args)=>{if(String(args[0]).includes('/api/support/tickets?')&&String(args[0]).includes('Alpha')){window.delayStarted=true;await new Promise(r=>setTimeout(r,600));}return originalFetch(...args);};document.getElementById('support-query').value='Alpha';document.getElementById('support-filter').requestSubmit()")
            wait(b,'window.delayStarted');b.js("document.getElementById('support-query').value='Beta';document.getElementById('support-filter').requestSubmit()")
            wait(b,"document.getElementById('support-rows').textContent.includes('Beta target')&&!document.getElementById('support-rows').textContent.includes('Alpha target')");results['support_latest_filter_wins']=True
            b.js("window.fetch=originalFetch;document.getElementById('support-query').value='Alpha';document.getElementById('support-filter').requestSubmit()")
            wait(b,"document.getElementById('support-rows').textContent.includes('Alpha target')");click_text(b,'#support-rows button','Alpha target')
            b.js("document.querySelector('#support-detail textarea').value='User draft';document.querySelector('#support-detail textarea').dispatchEvent(new Event('input'));document.getElementById('support-title').value='New title draft';document.dispatchEvent(new Event('visibilitychange'))")
            time.sleep(.4);assert b.js("return document.querySelector('#support-detail textarea').value==='User draft'&&document.getElementById('support-title').value==='New title draft'")
            current=store.ticket(ticket['id']);store.update_ticket(ticket['id'],version=current['version'],message='Latest reply',actor='admin')
            click_text(b,'#support-detail button','发送回复');wait(b,"document.getElementById('support-notice').textContent.includes('草稿已保留')")
            click_text(b,'#support-detail button','读取最新回复（保留草稿）');wait(b,"document.getElementById('support-detail').textContent.includes('Latest reply')");assert b.js("return document.querySelector('#support-detail textarea').value==='User draft'")
            click_text(b,'#support-detail button','发送回复');wait(b,"document.querySelector('#support-detail .ticket-messages').textContent.includes('User draft')");results['support_conflict_reload_preserves_reply']=True
            # Force cross-tab session termination, then verify no old owner text or draft remains visible.
            b.js("window.logoutDone=false;fetch('/api/session').then(r=>r.json()).then(s=>fetch('/api/session/logout',{method:'POST',headers:{'Content-Type':'application/json','X-CSRF-Token':s.csrf_token},body:'{}'})).then(()=>{window.logoutDone=true;document.dispatchEvent(new Event('visibilitychange'));})")
            wait(b,"window.logoutDone&&!document.getElementById('support-login').hidden");assert b.js("return !document.getElementById('support-content').textContent.includes('Alpha target')&&document.getElementById('support-title').value===''&&document.getElementById('support-query').value===''")
            b.js("document.getElementById('support-username').value='bob';document.getElementById('support-password').value='bob-test-secret';document.getElementById('support-login-form').requestSubmit()")
            wait(b,"!document.getElementById('support-content').hidden");assert b.js("return !document.getElementById('support-content').textContent.includes('User draft')")
            results['expired_session_and_account_switch_clear_private_data']=True
            results['support_mobile_viewports'] = []
            for width in [390,320]:
                viewport = set_css_viewport(b,width,844)
                assert viewport['innerWidth'] == width and viewport['innerHeight'] == 844
                assert viewport['scrollWidth'] <= viewport['clientWidth'], viewport
                results['support_mobile_viewports'].append(viewport)
            results['support_mobile_320_390']=True
            results['javascript_errors']=b.errors();assert not results['javascript_errors']
        print(json.dumps(results,ensure_ascii=False,indent=2),flush=True)
        (args.out / 'browser-edges.json').write_text(json.dumps(results,ensure_ascii=False,indent=2), encoding='utf-8')
    finally:svc.close()
