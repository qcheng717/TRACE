"""MOTIF20 single-arm evaluation using RoboTwin's official task protocol.

Expert-selected arm is an explicit oracle protocol, not a learned arm selector.
Bimanual dump-bin scenes are excluded and recorded, matching training scope.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, nullcontext
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import queue
import signal
import subprocess
import sys
import traceback
import time
import threading

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from trace_robotwin.data.tasks import ROBOTWIN_TASKS


def atomic_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)


@contextmanager
def evaluation_lock(out):
    """Only one coordinator may own an output; its scene children share it."""
    with (out/'.eval.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Another evaluation owns {out}; do not run two coordinators here') from exc
        lock.seek(0); lock.truncate(); lock.write(str(os.getpid())+'\n'); lock.flush()
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def special_scene_arm(task, env):
    """Return explicit scene arm; 'both' scenes are outside this policy's scope."""
    if task == 'place_empty_cup':
        return 'right' if env.cup.get_pose().p[0] > 0 else 'left'
    if task == 'dump_bin_bigbin':
        return 'both' if env.deskbin.get_pose().p[0] >= 0 else 'left'
    return None


def expert_grasp_failure(stage, exc):
    if stage != 'expert':
        return False
    if isinstance(exc, AssertionError):
        return str(exc) == 'target_pose cannot be None for move action.'
    if not isinstance(exc, TypeError) or str(exc) != "'NoneType' object is not subscriptable":
        return False
    # Match only the official alarm-clock grasp call at the failing frame.
    import linecache
    tb = exc.__traceback__
    if tb is None:
        return False
    while tb.tb_next is not None:
        tb = tb.tb_next
    filename = Path(tb.tb_frame.f_code.co_filename).resolve()
    expected = ROOT/'third_party/RoboTwin/envs/click_alarmclock.py'
    return (filename == expected.resolve() and tb.tb_frame.f_code.co_name == 'play_once'
            and 'self.get_grasp_pose(self.alarm,' in linecache.getline(str(filename),tb.tb_lineno)
            and '[:3]' in linecache.getline(str(filename),tb.tb_lineno))


def compatible_expert_fix(old, new):
    return False and (old.get('driver_sha256') in ('e88223bc498d4da3b64999d168b0c500bb79bee1701598f2a25ccd9ab87f5470','0b061ad0b209852a157e0d9a3b6c252930a1741496e7535cea1e6e7684789aed')
            and old.get('robotwin_commit') == new.get('robotwin_commit')
            and old.get('worker_sha256') == new.get('worker_sha256'))


def compatible_renderer_fix(old, new):
    """Only migrate the known V4 driver; policy, sim and data remain strict."""
    return (old.get('driver_sha256') in (
                '6b08f8047b012725f2a9900426d628a679c7f36ef19ff90f5b8bf6e22494c1a4',
                'e767054d59b58bd4f81b1298c492a47ca6c9ab87c74bac8f64ca8ad8d3b668c4',
                '701893cb235f38a97173578f9801e03f5e30e8de9feef287ba6deb61190dab21')
            and 'postprocess_sha256' in old
            and {k:v for k,v in old.items() if k != 'driver_sha256'}
                == {k:v for k,v in new.items() if k != 'driver_sha256'})


def resume_config(previous, requested, extend=False):
    """Grow only the scene budget, preserving the original language protocol."""
    requested = dict(requested)
    if previous is None:
        if extend: raise ValueError('--extend-episodes requires an existing evaluation')
        return requested
    language_count = previous.get('instruction_episodes', previous['episodes'])
    if 'instruction_episodes' in previous:
        requested['instruction_episodes'] = language_count
    if extend and requested['episodes'] > previous['episodes']:
        requested['instruction_episodes'] = language_count
        same_protocol = ({k:v for k,v in previous.items() if k not in ('episodes','instruction_episodes')}
            == {k:v for k,v in requested.items() if k not in ('episodes','instruction_episodes')})
        if same_protocol: return requested
    if requested != previous:
        raise ValueError('Evaluation configuration differs; to increase only episodes use '
            '--resume --extend-episodes, otherwise use a new output directory')
    return requested


def write_summary(out, config):
    tasks = {}
    for task in config['tasks']:
        rows = [json.loads(f.read_text()) for f in (out/task).glob('episode_*.json')]
        evaluated = [r for r in rows if r['status'] in ('completed','truncated')]
        successes = sum(r['success'] for r in evaluated)
        tasks[task] = dict(successes=successes, evaluated=len(evaluated), requested=config['episodes'],
            success_rate=successes/len(evaluated) if evaluated else None,
            truncated=sum(r['status']=='truncated' for r in evaluated),
            errors=sum(r['status']=='error' for r in rows),
            skipped=sum(r['status']=='skipped' for r in rows),
            bimanual_excluded=sum(r.get('reason')=='bimanual handover scene' for r in rows))
    total = sum(t['evaluated'] for t in tasks.values())
    success = sum(t['successes'] for t in tasks.values())
    atomic_json(out/'summary.json', dict(protocol=config['protocol'], debug=config['max_steps'] is not None,
        complete=all(t['evaluated']==config['episodes'] for t in tasks.values()),
        successes=success,evaluated=total,success_rate=success/total if total else None,tasks=tasks))


def scene_command(a, out, task, seed, accepted):
    # Keep the full task list and requested scene budget. Children read the
    # frozen language generation count from the coordinator's configuration.
    command = [sys.executable, str(Path(__file__).resolve()), '--protocol', 'v4',
        '--checkpoint', str(Path(a.checkpoint).resolve()), '--output', str(out),
        '--tasks', *a.tasks, '--episodes', str(a.episodes), '--seed', str(a.seed),
        '--execution-horizon', str(a.execution_horizon), '--instruction-type', a.instruction_type,
        '--policy-device', a.policy_device, '--policy-python', a.policy_python,
        '--resume', '--_scene', task, str(seed), str(accepted)]
    if a.video: command.append('--video')
    if a.max_steps is not None: command.extend(['--max-steps', str(a.max_steps)])
    return command


def isolated_scenes(a, out, config):
    """Release all native renderer resources by exiting after each scene."""
    print('V4 renderer isolation: a fresh process for every candidate scene.', flush=True)
    gpus = getattr(a, 'gpus', None)
    stop = threading.Event()
    summary_lock = threading.Lock()
    process_lock = threading.Lock()
    live = {}

    def summary():
        # Scene children write only their own episodes; the coordinator is
        # the sole summary/config/provenance writer, including in parallel.
        with summary_lock:
            write_summary(out, config)

    def cancel_children():
        stop.set()
        with process_lock:
            processes = list(live.values())
        for process in processes:
            try: os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError: pass
        deadline = time.monotonic()+10
        for process in processes:
            try: process.wait(timeout=max(.01, deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                try: os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError: pass
                process.wait()

    def run_scene(command, gpu, log):
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), TAO_EVAL_GPU=str(gpu))
        with log.open('w') as stream:
            with process_lock:
                if stop.is_set(): return None
                process = subprocess.Popen(command, env=env, stdout=stream,
                    stderr=subprocess.STDOUT, start_new_session=True)
                live[process.pid] = process
            try:
                return subprocess.CompletedProcess(command, process.wait())
            finally:
                with process_lock:
                    live.pop(process.pid, None)

    write_summary(out, config)

    def run_task(task, gpu=None):
        dest = out/task; dest.mkdir(exist_ok=True)
        accepted = 0
        for seed in range(a.seed, a.seed+max(100,a.episodes*50)):
            if stop.is_set(): return
            if accepted >= a.episodes: break
            resultfile = dest/f'episode_{seed:06d}.json'
            if resultfile.exists():
                row = json.loads(resultfile.read_text())
                if row['status'] in ('completed','truncated'):
                    accepted += 1
                    continue
                if row['status'] == 'skipped': continue
                if row['status'] != 'error': raise ValueError(f'Unknown result status in {resultfile}')
                # Preserve failed infrastructure attempts before retrying this
                # SAME seed. They never count as successes or policy failures.
                history = out/'error_history'/task
                history.mkdir(parents=True, exist_ok=True)
                atomic_json(history/f'episode_{seed:06d}_{time.time_ns()}.json', row)
            command = scene_command(a, out, task, seed, accepted)
            if gpu is None:
                result = subprocess.run(command)
            else:
                logs = out/'worker_logs'/task
                logs.mkdir(parents=True, exist_ok=True)
                log = logs/f'episode_{seed:06d}_gpu{gpu}_{time.time_ns()}.log'
                print(f'[GPU{gpu}] {task} seed={seed}: start; log={log}', flush=True)
                result = run_scene(command, gpu, log)
                if result is None: return
            summary()
            if result.returncode:
                stop.set()
                if gpu is not None:
                    print(f'[GPU{gpu}] {task} seed={seed}: ERROR; see {log}', flush=True)
                raise subprocess.CalledProcessError(result.returncode, result.args)
            if not resultfile.exists(): raise RuntimeError(f'Scene child wrote no result: {task} seed={seed}')
            row = json.loads(resultfile.read_text())
            if row['status'] in ('completed','truncated'): accepted += 1
            elif row['status'] != 'skipped': raise RuntimeError(f'Scene child failed: {row}')
            if gpu is not None:
                print(f'[GPU{gpu}] {task} seed={seed}: {row["status"]}; '
                    f'{accepted}/{a.episodes} success={row.get("success")} steps={row.get("steps")}', flush=True)
        if accepted < a.episodes: raise RuntimeError(f'{task}: insufficient feasible single-arm seeds')
        if gpu is not None:
            print(f'[GPU{gpu}] {task}: complete ({accepted}/{a.episodes})', flush=True)

    if not gpus:
        for task in a.tasks: run_task(task)
    else:
        tasks = queue.Queue()
        for task in a.tasks: tasks.put(task)
        print(f'Parallel task dispatch: GPUs={gpus}; one scene process per GPU.', flush=True)
        records = out/'dispatch_runs'; records.mkdir(exist_ok=True)
        atomic_json(records/f'run_{time.time_ns()}.json', dict(gpus=gpus, tasks=a.tasks,
            coordinator_pid=os.getpid(), checkpoint=config.get('checkpoint'),
            driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))

        def gpu_worker(gpu):
            try:
                while not stop.is_set():
                    try: task = tasks.get_nowait()
                    except queue.Empty: return
                    run_task(task, gpu)
            except BaseException:
                stop.set()
                raise

        pool = ThreadPoolExecutor(max_workers=len(gpus))
        try:
            futures = [pool.submit(gpu_worker, gpu) for gpu in gpus]
            for future in as_completed(futures): future.result()
        except BaseException:
            cancel_children()
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
            summary()
    summary()


def main(default_protocol='v4'):
    p = argparse.ArgumentParser()
    p.add_argument('--protocol',choices=['v4'],default=default_protocol)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--tasks', nargs='+', choices=[*ROBOTWIN_TASKS, 'open_laptop'], default=ROBOTWIN_TASKS)
    p.add_argument('--episodes', type=int, default=100, help='Accepted scenes per task')
    p.add_argument('--seed', type=int, default=100000)
    p.add_argument('--execution-horizon', type=int, default=8)
    p.add_argument('--instruction-type', choices=['seen','unseen'], default='seen')
    p.add_argument('--max-steps', type=int, help='Debug truncation only; omitted uses official task limit')
    p.add_argument('--video', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--extend-episodes', action='store_true',
        help='With --resume, increase accepted scenes per task while retaining existing results and language generation')
    p.add_argument('--policy-device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--policy-python', default=os.environ.get('TRACE_POLICY_PYTHON',sys.executable))
    p.add_argument('--gpus', nargs='+', type=int,
        help='V4 parallel evaluation on physical GPU indices, e.g. --gpus 0 1 2 3 5 6 7')
    p.add_argument('--_scene', nargs=3, metavar=('TASK','SEED','ACCEPTED'), help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.episodes < 1 or a.execution_horizon < 1 or (a.max_steps is not None and a.max_steps < 1):
        p.error('Counts must be positive')
    if len(set(a.tasks)) != len(a.tasks): p.error('Tasks must be unique')
    if a.extend_episodes and (not a.resume or a.protocol != 'v4' or a._scene is not None):
        p.error('--extend-episodes requires --resume in V4 coordinator mode')
    if a.gpus is not None and (a.protocol != 'v4' or a._scene is not None
            or any(gpu < 0 for gpu in a.gpus) or len(set(a.gpus)) != len(a.gpus)):
        p.error('--gpus requires V4 coordinator mode and unique nonnegative GPU indices')
    if a._scene is not None:
        task, scene_seed, accepted = a._scene
        a._scene = (task, int(scene_seed), int(accepted))
        if (a.protocol != 'v4' or task not in a.tasks or not 0 <= int(accepted) < a.episodes
                or not a.seed <= int(scene_seed) < a.seed+max(100,a.episodes*50)):
            p.error('Invalid internal V4 scene request')
    out = Path(a.output).resolve(); out.mkdir(parents=True, exist_ok=True)
    with (evaluation_lock(out) if a.protocol == 'v4' and a._scene is None else nullcontext()):
        evaluate(a, out)


def evaluate(a, out):
    version='robotwin_'+a.protocol
    worker_path=ROOT/'tools/robotwin/policy_worker.py'
    action_dim,action_type=(14,'qpos') if a.protocol=='v4' else (16,'ee')
    ckpt = Path(a.checkpoint).resolve(strict=True)
    digest = hashlib.sha256()
    with ckpt.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024), b''): digest.update(block)
    config = dict(checkpoint=str(ckpt), checkpoint_sha256=digest.hexdigest(), tasks=a.tasks,
        episodes=a.episodes, seed=a.seed, execution_horizon=a.execution_horizon,
        instruction_type=a.instruction_type, max_steps=a.max_steps, ema=True, video=a.video,policy_device=a.policy_device,model_version=version,
        protocol='demo_clean single_arm oracle_expert_arm; excludes bimanual dump_bin_bigbin')
    cfgpath = out/'evaluation_config.json'
    previous_config = None
    if cfgpath.exists():
        if not a.resume: raise ValueError('Output already exists; use --resume or a new output directory')
        previous_config = json.loads(cfgpath.read_text())
    elif a._scene is not None:
        raise ValueError('Scene child requires an existing coordinator configuration')
    config = resume_config(previous_config, config, getattr(a, 'extend_episodes', False))
    root = Path(os.environ['ROBOTWIN_ROOT']).resolve()
    provenance = dict(robotwin_commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        worker_sha256=hashlib.sha256(worker_path.read_bytes()).hexdigest())
    if a.protocol=='v4':
        for name,path in [('postprocess',ROOT/'src/trace_robotwin/eval/actions.py'),('normalization',ROOT/'src/trace_robotwin/data/dataset.py')]:
            provenance[name+'_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    provenance_path = out/'provenance.json'
    if provenance_path.exists() and json.loads(provenance_path.read_text()) != provenance:
        previous = json.loads(provenance_path.read_text())
        render_fix = a.protocol == 'v4' and compatible_renderer_fix(previous, provenance)
        if not (a.resume and a._scene is None and (compatible_expert_fix(previous,provenance) or render_fix)):
            raise ValueError('Evaluation code changed since this output was created; use a new output directory')
        name = 'provenance_before_scene_isolation.json' if render_fix else 'provenance_before_expert_grasp_fix.json'
        if previous.get('driver_sha256') == 'e767054d59b58bd4f81b1298c492a47ca6c9ab87c74bac8f64ca8ad8d3b668c4':
            name = 'provenance_before_parallel_dispatch.json'
        if previous.get('driver_sha256') == '701893cb235f38a97173578f9801e03f5e30e8de9feef287ba6deb61190dab21':
            name = 'provenance_before_episode_extension.json'
        atomic_json(out/name,previous)
        print('Resuming compatible evaluation driver update; original provenance preserved.',flush=True)
    if a._scene is None:
        if previous_config is not None and config != previous_config:
            history = out/'evaluation_extensions'; history.mkdir(exist_ok=True)
            snapshot = dict(previous_config=previous_config, new_config=config,
                previous_summary=json.loads((out/'summary.json').read_text()) if (out/'summary.json').exists() else None,
                previous_provenance=json.loads(provenance_path.read_text()) if provenance_path.exists() else None)
            atomic_json(history/f'episodes_{previous_config["episodes"]}_to_{config["episodes"]}_{time.time_ns()}.json',snapshot)
            print(f'Extending accepted scenes per task: {previous_config["episodes"]} -> {config["episodes"]}; '
                f'language generation count stays {config["instruction_episodes"]}. Existing episodes are retained.',flush=True)
        atomic_json(cfgpath, config)
    if a._scene is None: atomic_json(provenance_path,provenance)
    elif not provenance_path.exists(): raise ValueError('Scene child requires coordinator provenance')
    if a.protocol == 'v4' and a._scene is None:
        isolated_scenes(a, out, config)
        return
    os.chdir(root); sys.path.insert(0,str(root)); sys.path.insert(0,str(root/'scripts'))
    import imageio.v2 as imageio
    import numpy as np
    from eval_policy_xpolicylab import load_task_args, class_decorator, build_instruction, safe_close_env, is_episode_end
    from envs.utils import UnStableError
    worker = subprocess.Popen([a.policy_python,str(worker_path),str(ckpt),a.policy_device],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    def call(req):
        pickle.dump(req, worker.stdin, protocol=4); worker.stdin.flush()
        return pickle.load(worker.stdout)
    def observation(obs):
        # Transfer only the fields consumed by the policy, excluding simulator objects/depth.
        result=dict(endpose=obs['endpose'], observation={k:dict(rgb=v['rgb']) for k,v in obs['observation'].items() if 'rgb' in v})
        if a.protocol=='v4':result['joint_action']=obs['joint_action']
        return result
    def summary():
        if a._scene is None: write_summary(out, config)
    try:
        ready = pickle.load(worker.stdout)
        if ready['model_version']!=version:raise ValueError('Policy worker version differs')
        if a.execution_horizon > ready['horizon']: raise ValueError('Execution horizon exceeds trained horizon')
        print('Policy ready:', ready, flush=True)
        for task in ([a._scene[0]] if a._scene else a.tasks):
            dest = out/task; dest.mkdir(exist_ok=True)
            args,_ = load_task_args(dict(task_name=task,task_config='demo_clean',policy_name='motif_v3'))
            args.update(eval_mode=True,collect_data=False,render_freq=0,eval_video_log=False)
            accepted = a._scene[2] if a._scene else 0
            for seed in ([a._scene[1]] if a._scene else range(a.seed, a.seed+max(100,a.episodes*50))):
                if accepted >= a.episodes: break
                resultfile = dest/f'episode_{seed:06d}.json'
                if resultfile.exists():
                    row = json.loads(resultfile.read_text())
                    if row['status'] in ('completed','truncated'): accepted += 1; continue
                    if row['status'] == 'skipped': continue
                    # Previously failed infrastructure/rollout errors are retried, not silently counted as successes.
                env = class_decorator(task)
                writer = None
                stage = 'expert'
                started = time.monotonic()
                print(f'{task} seed={seed}: official expert feasibility check',flush=True)
                try:
                    env.setup_demo(now_ep_num=accepted,seed=seed,is_test=True,**args)
                    special_arm = special_scene_arm(task,env)
                    if special_arm == 'both':
                        atomic_json(resultfile,dict(task=task,seed=seed,status='skipped',reason='bimanual handover scene'))
                        continue
                    info = env.play_once()
                    feasible = bool(env.plan_success and env.check_success())
                    if not feasible:
                        atomic_json(resultfile,dict(task=task,seed=seed,status='skipped',reason='official expert feasibility failed'))
                        continue
                    arm = special_arm or info.get('info',{}).get('{a}')
                    if arm not in ('left','right'): raise ValueError(f'Cannot determine explicit arm: {info}')
                    instruction = build_instruction(args,info,a.instruction_type,
                        config.get('instruction_episodes', a.episodes),seed)
                    if instruction == task: raise ValueError('Official language generation failed; refusing task-name fallback')
                    safe_close_env(env)
                    stage = 'policy'
                    env.setup_demo(now_ep_num=accepted,seed=seed,is_test=True,**args)
                    env.set_instruction(instruction=instruction)
                    call(dict(op='reset',seed=seed,instruction=instruction,arm=arm))
                    obs = env.get_obs()
                    call(dict(op='obs',obs=observation(obs)))
                    if a.video:
                        writer = imageio.get_writer(str(dest/f'episode_{seed:06d}.mp4'),fps=10)
                        writer.append_data(obs['observation']['head_camera']['rgb'])
                    trace = []
                    limit = min(env.step_lim,a.max_steps) if a.max_steps else env.step_lim
                    while not is_episode_end(env) and env.take_action_cnt < limit:
                        chunk = call(dict(op='action',execution_horizon=a.execution_horizon))
                        if chunk.ndim != 2 or chunk.shape[1] != action_dim or not np.isfinite(chunk).all():
                            raise ValueError('Invalid action chunk')
                        for action in chunk[:a.execution_horizon]:
                            env.take_action(action,action_type=action_type)
                            obs = env.get_obs()
                            call(dict(op='obs',obs=observation(obs)))
                            trace.append(dict(step=int(env.take_action_cnt),action=action.tolist(),
                                active_pose=np.asarray(obs['endpose'][arm+'_endpose']).tolist()))
                            if writer: writer.append_data(obs['observation']['head_camera']['rgb'])
                            if is_episode_end(env) or env.take_action_cnt >= limit: break
                    success = bool(env.eval_success)
                    truncated = bool(not success and limit < env.step_lim)
                    row = dict(task=task,seed=seed,status='truncated' if truncated else 'completed',
                        success=success,active_arm=arm,instruction=instruction,steps=int(env.take_action_cnt),
                        official_step_limit=int(env.step_lim),elapsed_seconds=time.monotonic()-started,actions=trace)
                    atomic_json(resultfile,row); accepted += 1
                    print(f'{task} seed={seed} {accepted}/{a.episodes} success={success} steps={env.take_action_cnt}',flush=True)
                    summary()
                except Exception as exc:
                    if expert_grasp_failure(stage,exc):
                        atomic_json(resultfile,dict(task=task,seed=seed,status='skipped',stage=stage,
                            reason='official expert found no grasp target',error=repr(exc),traceback=traceback.format_exc()))
                        print(f'{task} seed={seed}: skip expert no-grasp-target scene',flush=True)
                        summary()
                        continue
                    if stage == 'expert' and isinstance(exc,UnStableError):
                        atomic_json(resultfile,dict(task=task,seed=seed,status='skipped',reason='official unstable scene',error=str(exc)))
                        continue
                    atomic_json(resultfile,dict(task=task,seed=seed,status='error',stage=stage,error=repr(exc),traceback=traceback.format_exc()))
                    summary()
                    raise  # Fail visibly; infrastructure errors must not inflate success rates.
                finally:
                    if writer: writer.close()
                    safe_close_env(env)
            if a._scene is None and accepted < a.episodes: raise RuntimeError(f'{task}: insufficient feasible single-arm seeds')
        summary()
    finally:
        worker.stdin.close()
        try: worker.wait(timeout=15)
        except subprocess.TimeoutExpired:
            worker.terminate(); worker.wait(timeout=15)


if __name__ == '__main__': main()
