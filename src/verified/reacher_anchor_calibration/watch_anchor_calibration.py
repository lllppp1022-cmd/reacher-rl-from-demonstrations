"""Independent native MuJoCo checkpoint replay. Never renders a training environment."""
import argparse
import glfw, mujoco
from PIL import Image
from anchor_common import *
from evaluate_anchor import load_policy

def run(c,args):
    setup(c);seed_everything(42)
    if args.policy in ('weak','medium','strong'):args.group=args.policy;args.policy='anchor'
    elif args.policy in ('fixed','decay'):args.phase='long_'+args.policy;args.policy='anchor'
    e=env('human',width=1080,height=800,default_camera_config=dict(distance=.65,elevation=-90,azimuth=90,lookat=np.zeros(3),trackbodyid=-1))
    closed=[False];paused=[False];speed=[args.speed];frames=loads=0;last=None;current=None;model=None;bc=None
    status=out(c)/f'live/viewer_{os.getpid()}.json';expert=Expert()
    try:
        obs,_=e.reset(seed=args.seed);e.render();viewer=e.unwrapped.mujoco_renderer.viewer
        assert viewer.window and glfw.get_window_attrib(viewer.window,glfw.VISIBLE)
        viewer._create_overlay=lambda:None
        glfw.set_window_close_callback(viewer.window,lambda w:closed.__setitem__(0,True))
        original=viewer._key_callback
        def key(window,k,scan,action,mods):
            if action==glfw.RELEASE:
                if k==glfw.KEY_ESCAPE:closed[0]=True
                elif k==glfw.KEY_SPACE:paused[0]=not paused[0]
                elif k==glfw.KEY_UP:speed[0]=min(8.,speed[0]*2)
                elif k==glfw.KEY_DOWN:speed[0]=max(.125,speed[0]/2)
                else:original(window,k,scan,action,mods)
        glfw.set_key_callback(viewer.window,key)
        modes=['ikpd','bc','control','anchor'] if args.policy=='compare' else [args.policy]
        while not closed[0]:
            for mode in modes:
                if closed[0]:break
                if not args.watch:
                    source_bc=source(c,args.training_seed)['path'];bc=load_policy(source_bc)
                    path=args.model
                    if mode=='bc':path=source_bc
                    elif mode=='control':path=str(folder(c,'b0',args.training_seed,'long_control' if args.phase.startswith('long') else 'short')/f'step{args.step}.zip')
                    elif mode=='anchor' and not path:path=str(folder(c,args.group,args.training_seed,args.phase)/f'step{args.step}.zip')
                    if mode!='ikpd' and not Path(path).exists():raise FileNotFoundError(f'Checkpoint has not been executed/saved: {path}; no replacement model is used')
                    model=load_policy(path) if mode!='ikpd' else None
                    current=dict(group=args.group if mode=='anchor' else mode,seed=args.training_seed,steps=args.step if mode in ('anchor','control') else 0,phase=args.phase)
                    if mode in ('anchor','control'):
                        actual=Path(path)
                        current.update(group=actual.parent.parent.name,seed=model.seed,steps=model.num_timesteps,phase=actual.parent.parent.parent.name)
                        current.update(beta_initial=model.group_settings['anchor_beta'],beta_effective=model.beta_value(),beta_schedule=model.group_settings['beta_schedule'])
                        current['evaluation']=str(out(c)/f'results/validation/{current["phase"]}_{current["group"]}_seed{model.seed}_step{model.num_timesteps}/summary.json')
                obs,_=e.reset(seed=args.seed);total=0.;history=[];episode_step=0
                if mode=='ikpd' and not args.watch:expert.reset(e,obs)
                for step in range(args.max_steps):
                    if closed[0] or (out(c)/'live/stop_viewer.json').exists():closed[0]=True;break
                    while paused[0] and not closed[0]:
                        if (out(c)/'live/stop_viewer.json').exists():closed[0]=True;break
                        e.render();glfw.poll_events();time.sleep(.02)
                    if closed[0]:break
                    start=time.perf_counter()
                    if args.watch:
                        try:
                            snapshot=read(out(c)/'live/current.json')
                            if snapshot['sha256']!=last:
                                assert sha(snapshot['path'])==snapshot['sha256']
                                candidate=load_policy(snapshot['path'])
                                model=candidate;bc=load_policy(snapshot['bc_path']);current=snapshot;last=snapshot['sha256'];loads+=1
                                obs,_=e.reset(seed=args.seed);total=0.;history=[];episode_step=0
                        except (OSError,ValueError,AssertionError):pass
                        if model is None:time.sleep(.05);continue
                    a=expert.action(obs) if mode=='ikpd' and not args.watch else model.predict(obs,deterministic=True)[0]
                    ba=bc.predict(obs,deterministic=True)[0] if bc else a
                    distance=float(np.linalg.norm(obs[8:10])*100)
                    evaluation=current.get('evaluation');stable='pending independent evaluation'
                    if evaluation and Path(evaluation).exists():stable=f"{read(evaluation)['stable_success_pct']:.1f}% (100 validation episodes)"
                    fields=[('BC Anchoring independent replay',f"{current['group']} train_seed={current['seed']} {current['phase']} steps={current['steps']}"),
                        ('Beta initial / current / schedule',f"{current.get('beta_initial',0)} / {current.get('beta_effective',0):.6f} / {current.get('beta_schedule','none')}"),
                        ('Demo seed / step / sim seconds',f'{args.seed} / {episode_step+1} / {(episode_step+1)*.02:.2f}'),
                        ('XY cm / within 1 cm',f'{distance:.3f} / {distance<1}'),('Actions',f'{a[0]:+.5f}, {a[1]:+.5f}'),
                        ('Difference from BC (same state)',f'{np.linalg.norm(a-ba):.6f}'),('Return',f'{total:.3f}'),
                        ('Stable success',stable),('Distance history cm',','.join(f'{x:.1f}' for x in history[-8:])),
                        ('Space pause; Up/Down speed; Esc close',f'{speed[0]:.2f}x')]
                    for label,value in fields:viewer.add_overlay(mujoco.mjtGridPos.mjGRID_TOPRIGHT,label,value)
                    obs,r,term,trunc,_=e.step(a);total+=r;history.append(float(np.linalg.norm(obs[8:10])*100));frames+=1;episode_step+=1
                    if frames==10:
                        pixels=np.zeros((viewer.viewport.height,viewer.viewport.width,3),np.uint8)
                        mujoco.mjr_readPixels(pixels,None,viewer.viewport,viewer.con)
                        Image.fromarray(np.flipud(pixels)).save(out(c)/f'live/viewer_{os.getpid()}.png')
                    if frames%25==0:
                        print('REPLAY',current,'demo',args.seed,'step',episode_step,'distance_cm',history[-1],'action',a.tolist(),'reward',total,flush=True)
                        atomic_json(status,dict(status='running',visible=True,pid=os.getpid(),frames=frames,loads=loads,current=current))
                    if term or trunc:break
                    time.sleep(max(0,1/(args.fps*speed[0])-(time.perf_counter()-start)))
            if not args.repeat and not args.watch:break
    finally:
        e.close();atomic_json(status,dict(status='closed',pid=os.getpid(),frames=frames,loads=loads,current=current))

def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config');p.add_argument('--policy',choices=['ikpd','bc','control','anchor','weak','medium','strong','fixed','decay','compare'],default='anchor')
    p.add_argument('--group',default='medium');p.add_argument('--training-seed',type=int,default=42);p.add_argument('--seed',type=int,default=7059)
    p.add_argument('--phase',choices=['short','long_fixed','long_decay','long_control','smoke'],default='short');p.add_argument('--step',type=int,default=20480)
    p.add_argument('--model');p.add_argument('--fps',type=float,default=25);p.add_argument('--speed',type=float,default=1.)
    p.add_argument('--repeat',action='store_true');p.add_argument('--watch',action='store_true');p.add_argument('--max-steps',type=int,default=250)
    return p

if __name__=='__main__':
    a=parser().parse_args();c=config(a.config);(out(c)/'live/stop_viewer.json').unlink(missing_ok=True);run(c,a)
