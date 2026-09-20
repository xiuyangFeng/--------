import contextlib, fcntl, json, multiprocessing, os, signal, subprocess, tempfile, time
from pathlib import Path
from unittest.mock import patch
from training_wss_min.tools import borrow_m2_diagnostic_slot as b

root=Path(tempfile.mkdtemp(prefix='m2-borrow-safety-'))
plan={'job_id':'99999','token':'t'*32,'step_name':'m2b_0123456789abcdef','directory':str(root)}
with patch.object(b.subprocess,'check_output',return_value='99999.3|m2b_0123456789abcdef    \n99999.batch|batch\n99999.4|other\n') as check:
    assert b.unique_steps(plan)=={'99999.3'}
    assert any('%100j' in arg for arg in check.call_args.args[0])
for bad in ['99999.batch','99999.extern','88888.3','99999']:
    with patch.object(b.subprocess,'check_output',return_value=f'{bad}|{plan["step_name"]}\n'):
        try: b.unique_steps(plan)
        except RuntimeError: pass
        else: raise AssertionError(bad)
b.save_json(root/'step_started.json',dict(token=plan['token'],job_id='99999',step_id='5'))
with patch.object(b.subprocess,'check_output',return_value='99999.5|truncated\n'):
    assert b.unique_steps(plan)=={'99999.5'}
(root/'step_started.json').unlink()
assert len(b.active_arms({'arms':{'x':{'id':'x','worker':'99999','status':'claimed'},'y':{'id':'y','worker':'99999','status':'eval_last'},'z':{'id':'z','worker':'99999','status':'complete'}}},'99999'))==2
with (root/'lease').open('a') as one, (root/'lease').open('a') as two:
    fcntl.flock(one,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try: fcntl.flock(two,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: pass
    else: raise AssertionError('duplicate lease accepted')

real_popen=subprocess.Popen
coordinator=real_popen(['/bin/sleep','60'],start_new_session=True)
plan.update(coordinator=b.identity(coordinator.pid),owner=b.identity(os.getpid()),cancel_pending='')
planpath=root/'plan.json'; b.save_json(planpath,plan)

def child():
    b.EXP=root
    state={'workers':{'99999':{'status':'running','pid':coordinator.pid}},'arms':{'x':{'id':'x','worker':'99999','status':'train'}}}
    @contextlib.contextmanager
    def locked(): yield state
    b.locked_state=locked
    b.verified_worker=lambda job, worker:b.identity(worker['pid'])
    def fake_popen(command,**kw):
        assert command[0].endswith('/srun')
        local=real_popen(['/bin/sleep','60'],start_new_session=True)
        remote=real_popen(['/bin/sleep','60'],start_new_session=True)
        b.save_json(root/'test_children.json',{'local':b.identity(local.pid),'remote':b.identity(remote.pid)})
        return local
    def fake_check(command,**kw):
        assert command[0].endswith('/squeue') and '--steps' in command
        info=json.loads((root/'test_children.json').read_text())
        return f'99999.7|{plan["step_name"]}\n' if b.same_process(info['remote']) else ''
    def fake_run(command,**kw):
        assert command==[b.SLURM+'scancel','99999.7'], command
        info=json.loads((root/'test_children.json').read_text())
        (root/'exact_cancel.txt').write_text('99999.7')
        os.killpg(info['remote']['pid'],signal.SIGTERM)
        return subprocess.CompletedProcess(command,0)
    b.subprocess.Popen=fake_popen
    b.subprocess.check_output=fake_check
    b.subprocess.run=fake_run
    try:
        b.controller(planpath)
    except RuntimeError as e:
        (root/'expected_abort.txt').write_text(str(e))
    else: raise AssertionError('controller ignored abort')

proc=multiprocessing.get_context('fork').Process(target=child)
try:
    proc.start()
    deadline=time.monotonic()+10
    while not (root/'test_children.json').exists():
        assert time.monotonic()<deadline, 'launch timeout'
        time.sleep(.02)
    assert b.identity(coordinator.pid)['state']=='T'
    os.kill(proc.pid,signal.SIGTERM)
    proc.join(20)
    assert not proc.is_alive(), 'controller cleanup timeout'
    assert proc.exitcode==0, proc.exitcode
    assert b.identity(coordinator.pid)['state'] not in {'T','t'}
    ev=json.loads((root/'controller.json').read_text())
    assert ev['diagnostic_step_absent_confirmed_at']<=ev['coordinator_resumed_at']
    assert (root/'exact_cancel.txt').read_text()=='99999.7'
    info=json.loads((root/'test_children.json').read_text())
    assert not b.same_process(info['local']) and not b.same_process(info['remote'])
    print(json.dumps({'passed':True,'checks':['exact numeric step only','foreign/non-numeric refused','padded step name','marker fallback','claimed arms counted','exclusive lease','SIGTERM cancels diagnostic before SIGCONT'],'evidence':str(root)}))
finally:
    if proc.is_alive(): proc.kill(); proc.join()
    os.kill(coordinator.pid,signal.SIGCONT)
    coordinator.terminate(); coordinator.wait()
    if (root/'test_children.json').exists():
        for item in json.loads((root/'test_children.json').read_text()).values():
            if b.same_process(item): os.killpg(item['pid'],signal.SIGKILL)
