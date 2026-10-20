"""真实PPO训练观察：human教学窗口或选定真实回合RGB记录；不修改既有实验。"""
import argparse,time
from pathlib import Path
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from anchor_common import *
from anchor_ppo import build
from training_observation import ActualTrainingEnv,SamplerObserver,ObservedAnchorPPO

def run(args):
    c=config(args.config);dest=Path(args.output or HERE/'online_trials'/f'{args.group}_seed{args.training_seed}_{time.time_ns()}').resolve()
    assert args.steps>0 and args.steps%1024==0,'steps必须是1024的整数倍'
    assert args.record_every_n_episodes>=1 and args.max_recorded_episodes>=1
    if args.render_training and args.record_training_rollouts:raise ValueError('human教学与RGB录制请分开运行，避免多上下文影响观察；两种模式都记录真实动作/状态CSV')
    if (dest/'complete.json').exists():
        info=read(dest/'complete.json');assert sha(info['model_path'])==info['model_sha256'];print('ALREADY COMPLETE',dest,flush=True);return info
    if (dest/'training_complete.json').exists():
        info=read(dest/'training_complete.json');assert sha(info['model_path'])==info['model_sha256']
        print('RECOVER POSTPROCESSING ONLY; NO ENVIRONMENT STEPS',dest,flush=True)
        from training_rollout_analysis import evaluation_replays,analyze
        original=read(dest/'protocol.json')['arguments']
        if not (args.skip_evaluation_replays or original['skip_evaluation_replays']):evaluation_replays(c,dest,original['diagnostic_seed'])
        analyze(dest);atomic_json(dest/'complete.json',info);return info
    assert not dest.exists() or not any(dest.iterdir()),'输出目录已有未完成内容；请使用新目录，不覆盖历史模型'
    dest.mkdir(parents=True,exist_ok=True)
    settings=phase_settings(c,args.group,'short');settings['total_steps']=args.steps
    observer=SamplerObserver(dest,args.record_training_rollouts,args.record_every_n_episodes,args.max_recorded_episodes,
        args.render_training,args.render_fps,args.disable_render_after,args.inject_rgb_failure)
    render_mode='human' if args.render_training else 'rgb_array' if args.record_training_rollouts else None
    e=ActualTrainingEnv(env(render_mode,width=640,height=480,default_camera_config=dict(distance=.65,elevation=-90,azimuth=90,lookat=np.zeros(3),trackbodyid=-1)),observer)
    monitored=Monitor(e,filename=str(dest/'training.monitor.csv'))
    model=build(c,args.group,args.training_seed,monitored,settings=settings)
    # A pure instrumentation subclass: parameters, buffers, optimizer, env and RNG are untouched.
    model.__class__=ObservedAnchorPPO;model.training_observer=observer;observer.active_wrapper=e
    model.on_update=lambda m,row,minibatches:setattr(observer,'last_report',dict(row))
    model.set_logger(configure(str(dest/'tensorboard'),['tensorboard']))
    observer.checkpoint(model)
    initial=mean_actor_hash(model);protocol=dict(arguments=vars(args),source=source(c,args.training_seed),settings=settings,ppo=c['ppo'],
        initial_actor_sha256=initial,environment=dict(name='Reacher-v5',max_episode_steps=250,dt=.02,action_bounds=[-1,1],reward_weights=[1,1]),
        camera=dict(distance=.65,elevation=-90,azimuth=90,lookat=[0,0,0]),
        recording_selection='Episode1,1+N,1+2N,... capped before training; includes failures without outcome filtering',
        data_source='Actual Training Rollout: inherited SB3 collect_rollouts samples stochastic actions; wrapper calls real env.step exactly once',
        implementation_hashes={n:sha(HERE/n) for n in ['training_observation.py','record_training_rollouts.py','anchor_ppo.py']})
    atomic_json(dest/'protocol.json',protocol);started=time.perf_counter()
    try:model.learn(args.steps,callback=observer)
    finally:
        observer.safe(lambda:observer.finish(model),'finish_observation');monitored.close()
    training_seconds=time.perf_counter()-started
    assert model.num_timesteps==args.steps and e.global_step==args.steps
    final=dest/'checkpoints'/f'step{args.steps}.zip';assert final.exists()
    info=dict(group=args.group,seed=args.training_seed,steps=model.num_timesteps,rollout_count=observer.rollout_id,policy_update_count=observer.version,
        sb3_n_updates=model._n_updates,model_path=str(final),model_sha256=sha(final),initial_actor_sha256=initial,final_actor_sha256=mean_actor_hash(model),
        training_seconds=training_seconds,render_seconds=observer.render_seconds,output=str(dest),observer_errors=observer.errors,
        human_visible=observer.human_visible,rendering_disabled_at=observer.closed_render_at,finished_unix=time.time())
    atomic_json(dest/'training_complete.json',info)
    if not args.skip_evaluation_replays:
        # Per-update deterministic diagnostics happen only AFTER training, hence cannot touch its RNG stream.
        from training_rollout_analysis import evaluation_replays
        evaluation_replays(c,dest,args.diagnostic_seed)
    from training_rollout_analysis import analyze
    analyze(dest)
    atomic_json(dest/'complete.json',info);print('ACTUAL TRAINING COMPLETE',json.dumps(info,ensure_ascii=False),flush=True)
    return info

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config');p.add_argument('--output',help='独立且未使用的目录；已有有效完成任务只核对后跳过')
    p.add_argument('--group',choices=['b0','weak','medium','strong'],default='b0');p.add_argument('--training-seed',type=int,choices=[42,123,2026],default=42)
    p.add_argument('--steps',type=int,default=4096);p.add_argument('--render-training',action='store_true',help='直接显示PPO所用训练环境；R/Esc/关闭窗口后继续无渲染训练')
    p.add_argument('--record-training-rollouts',action='store_true',help='只为预选回合捕获真实训练环境RGB，所有步均记动作/状态')
    p.add_argument('--record-every-n-episodes',type=int,default=4);p.add_argument('--max-recorded-episodes',type=int,default=4)
    p.add_argument('--render-fps',type=float,default=50);p.add_argument('--disable-render-after',type=int,help='自动关闭渲染的调试测试步数；不终止训练')
    p.add_argument('--diagnostic-seed',type=int,default=17059,help='仅用于训练完成后的Evaluation Replay，不用于训练reset')
    p.add_argument('--skip-evaluation-replays',action='store_true');p.add_argument('--inject-rgb-failure',action='store_true',help=argparse.SUPPRESS)
    return p

if __name__=='__main__':
    a=parser().parse_args()
    try:run(a)
    except Exception as exc:
        import traceback
        if a.output:
            Path(a.output).mkdir(parents=True,exist_ok=True);atomic_json(Path(a.output)/'error.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
