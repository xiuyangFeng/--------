"""Repeatable Firefox acceptance of /ops and /support, using synthetic temporary data.

    PYTHONPATH=. python -m tests.ops_browser_check --out outputs/wss_ops_browser_check

Requires the system Firefox used by wss_deploy.devshot. No production jobs,
credentials, GPU, training, or network downloads are involved.
"""
import argparse
import json
import socket
import tempfile
import time
from pathlib import Path

from tests._c_helpers import Service, finished
from wss_deploy.users import UserStore
from wss_deploy.devshot import Browser
from tests._ops_browser_viewport import set_css_viewport

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out', type=Path, default=Path('outputs/wss_ops_browser_check'))
args = parser.parse_args()
out = args.out.resolve()
out.mkdir(parents=True, exist_ok=True)
(out / 'browser-check.json').unlink(missing_ok=True)

def wait(browser, expression, timeout=15):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        result = browser.js('return ' + expression)
        if result:
            return result
        time.sleep(.2)
    raise AssertionError(expression + '\n' + str(browser.js('return document.body.innerText')))

def view(browser, name):
    browser.js("document.querySelector('#ops-nav [data-view=\"'+arguments[0]+'\"]').click()", [name])
    wait(browser, f"!document.getElementById('view-{name}').hidden")

with tempfile.TemporaryDirectory(prefix='wss-ops-browser-') as temporary:
    root = Path(temporary)
    users = UserStore(root / 'users.json')
    users.add('ops_admin', 'browser-admin-secret', admin=True, display_name='运维管理员')
    users.add('researcher_a', 'browser-user-secret', display_name='研究用户 A')
    users.add('researcher_b', 'browser-user-b-secret', display_name='研究用户 B')
    service = Service(root, shared=True, users=users)
    try:
        a = finished(service.manager, owner='researcher_a', case_id='DEMO_A', content=b'solid synthetic A\nendsolid synthetic A')
        b = finished(service.manager, owner='researcher_b', case_id='DEMO_B', content=b'solid synthetic B\nendsolid synthetic B')
        c = finished(service.manager, owner='researcher_a', case_id='DEMO_TRASH', content=b'solid synthetic C\nendsolid synthetic C')
        service.manager.delete(c['id'], 'researcher_a', {'version': c['version']})
        store = service.server.operations
        store.create_ticket(owner='researcher_a', title='核查病例几何', description='请检查输入确认步骤并保留 STL 供后续复查。', job_id=a['id'])
        store.create_ticket(owner='researcher_b', title='结果导出咨询', description='需要确认结果导出的单位。', priority='high', job_id=b['id'])
        base = f'http://127.0.0.1:{service.port}'
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0)); marionette_port = sock.getsockname()[1]
        with Browser(width=1520, height=1100, marionette_port=marionette_port, log_path=out / 'browser.log') as browser:
            browser.go(base + '/ops', wait=.5)
            wait(browser, "!document.getElementById('ops-login').hidden")
            browser.js("document.getElementById('login-user').value='ops_admin'; document.getElementById('login-password').value='browser-admin-secret'; document.getElementById('ops-login-form').requestSubmit();")
            wait(browser, "!document.getElementById('ops-content').hidden")
            view(browser, 'jobs')
            wait(browser, "document.getElementById('jobs-rows').textContent.includes('DEMO_A')")
            assert 'DEMO_B' in browser.js("return document.getElementById('jobs-rows').textContent")
            browser.js("document.querySelector('#jobs-rows button').click()")
            view(browser, 'assets')
            wait(browser, "document.querySelectorAll('#assets-rows select').length > 0")
            view(browser, 'events')
            wait(browser, "document.querySelectorAll('#events-rows details.event-entry').length > 0")
            browser.shot(out / 'ops-desktop.png', full=True)
            assert browser.js('return document.documentElement.scrollWidth <= innerWidth'), 'desktop overflow'
            ops_mobile_viewport = set_css_viewport(browser, 390, 844)
            time.sleep(.3)
            browser.shot(out / 'ops-mobile.png', full=True)
            assert browser.js('return document.documentElement.scrollWidth <= innerWidth'), 'mobile overflow'
            set_css_viewport(browser, 1520, 1100)
            browser.go(base + '/support', wait=.5)
            wait(browser, "!document.getElementById('support-content').hidden")
            browser.js("document.getElementById('support-logout').click()")
            wait(browser, "!document.getElementById('support-login').hidden")
            browser.js("document.getElementById('support-username').value='researcher_a'; document.getElementById('support-password').value='browser-user-secret'; document.getElementById('support-login-form').requestSubmit();")
            wait(browser, "document.getElementById('support-rows').textContent.includes('核查病例几何')")
            assert '结果导出咨询' not in browser.js("return document.getElementById('support-rows').textContent")
            browser.js("document.getElementById('support-title').value='浏览器端到端验证'; document.getElementById('support-description').value='测试工单提交与管理员处理'; document.getElementById('support-create').requestSubmit();")
            wait(browser, "document.getElementById('support-detail').textContent.includes('浏览器端到端验证')")
            browser.js("const el=document.querySelector('#support-detail textarea'); el.value='用户追加说明'; el.dispatchEvent(new Event('input',{bubbles:true})); document.querySelector('#support-detail button.primary').click();")
            wait(browser, "document.querySelector('#support-detail .ticket-messages').textContent.includes('用户追加说明')")
            browser.shot(out / 'support-desktop.png', full=True)
            browser.go(base + '/ops', wait=.5)
            wait(browser, "!document.getElementById('ops-login').hidden")
            assert browser.js("return document.getElementById('ops-content').hidden")
            browser.js("document.getElementById('login-user').value='ops_admin'; document.getElementById('login-password').value='browser-admin-secret'; document.getElementById('ops-login-form').requestSubmit();")
            wait(browser, "!document.getElementById('ops-content').hidden")
            view(browser, 'tickets')
            wait(browser, "document.getElementById('tickets-rows').textContent.includes('浏览器端到端验证')")
            browser.js("Array.from(document.querySelectorAll('#tickets-rows button')).find(e=>e.textContent==='浏览器端到端验证').click(); document.querySelector('#ticket-detail select[aria-label=处理状态]').value='in_progress'; const el=document.querySelector('#ticket-detail textarea'); el.value='运维人员已受理'; el.dispatchEvent(new Event('input',{bubbles:true})); document.querySelector('#ticket-detail button.primary').click();")
            wait(browser, "document.querySelector('#ticket-detail .ticket-messages').textContent.includes('运维人员已受理')")
            # State changes appear without reloading or pressing refresh.
            view(browser, 'jobs')
            finished(service.manager, owner='researcher_b', case_id='LIVE_POLL_DEMO', content=b'solid live\nendsolid live')
            wait(browser, "document.getElementById('jobs-rows').textContent.includes('LIVE_POLL_DEMO')", timeout=15)
            view(browser, 'events')
            browser.shot(out / 'ops-desktop.png', full=True)
            browser.go(base + '/support', wait=.5)
            browser.js("window.switchDone=false; fetch('/api/session',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({username:'researcher_a',password:'browser-user-secret'})}).then(r=>r.json()).then(()=>window.switchDone=true)")
            wait(browser, 'window.switchDone')
            browser.js("document.getElementById('support-refresh').click()")
            wait(browser, "document.getElementById('support-rows').textContent.includes('浏览器端到端验证')")
            browser.js("Array.from(document.querySelectorAll('#support-rows button')).find(e=>e.textContent==='浏览器端到端验证').click()")
            wait(browser, "document.getElementById('support-detail').textContent.includes('运维人员已受理')")
            support_mobile_viewport = set_css_viewport(browser, 390, 844); time.sleep(.3)
            browser.shot(out / 'support-mobile.png', full=True)
            assert browser.js('return document.documentElement.scrollWidth <= innerWidth'), 'support mobile overflow'
            browser.js("window.switchDone=false; fetch('/api/session',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({username:'researcher_b',password:'browser-user-b-secret'})}).then(r=>r.json()).then(()=>window.switchDone=true)")
            wait(browser, 'window.switchDone')
            wait(browser, "document.getElementById('support-actor').textContent.includes('研究用户 B')", timeout=15)
            assert '浏览器端到端验证' not in browser.js("return document.getElementById('support-content').textContent")
            result = {'ops_login': True, 'cross_owner_jobs': True, 'archive_from_job_button': True,
                      'desktop_mobile_no_overflow': True, 'support_owner_isolation': True,
                      'ticket_create_reply_admin_process': True, 'automatic_refresh': True,
                      'cross_tab_account_switch': True, 'javascript_errors': browser.errors(),
                      'viewports': {'ops_mobile': ops_mobile_viewport, 'support_mobile': support_mobile_viewport}}
            print(json.dumps(result, ensure_ascii=False), flush=True)
            assert not result['javascript_errors'], result
            (out / 'browser-check.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        service.close()
