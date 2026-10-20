"""Capture genuinely new frames and validate encoded animation, not just its filename."""
import argparse
import imageio.v2 as imageio
from PIL import Image
from bc_common import *

def record(c,method,model_path=None,seed=7100,fps=50,fmt='gif',suffix=''):
    setup(c);seed_everything(42);expert=Expert()
    model_path=Path(model_path or policy_path(c,method)) if method!='ikpd' else None
    model=load_bc(model_path) if model_path else None
    if model_path:suffix+='_'+sha(model_path)[:8]
    e=env('rgb_array',width=640,height=480,default_camera_config=dict(distance=.65,elevation=-90,azimuth=90,lookat=np.zeros(3),trackbodyid=-1))
    frames=[];states=[];trace=[]
    try:
        obs,_=e.reset(seed=seed)
        if method=='ikpd':expert.reset(e,obs)
        for step in range(250):
            a=expert.action(obs) if method=='ikpd' else model.predict(obs,deterministic=True)[0]
            obs,r,term,trunc,_=e.step(a);frames.append(e.render().copy());states.append(e.unwrapped.data.qpos[:2].copy())
            trace.append(dict(step=step+1,time_s=(step+1)*.02,action1=float(a[0]),action2=float(a[1]),distance_cm=float(np.linalg.norm(obs[8:10]))*100))
            assert not term and trunc==(step==249)
    finally:e.close()
    destination=out(c)/'videos'/f'{method}_seed{seed}{suffix}.{fmt}';fallback=None
    if fmt=='mp4':
        try:
            with imageio.get_writer(destination,fps=fps,codec='libx264') as writer:
                for frame in frames:writer.append_data(frame)
        except (ImportError,ValueError) as exc:fallback=repr(exc);destination=destination.with_suffix('.gif');fmt='gif'
    if fmt=='gif':imageio.mimsave(destination,frames,format='GIF',duration=1000/fps,loop=0)
    unique=len({hashlib.sha256(frame.tobytes()).hexdigest() for frame in frames})
    diffs=np.array([np.abs(b.astype(np.float32)-a.astype(np.float32)).mean() for a,b in zip(frames,frames[1:])])
    if fmt=='gif':
        with Image.open(destination) as im:
            count=im.n_frames;duration=0
            for i in range(count):im.seek(i);im.load();duration+=im.info.get('duration',0)
    else:
        with imageio.get_reader(destination) as reader:count=sum(1 for _ in reader)
        duration=1000*count/fps
    movement=float(np.linalg.norm(np.array(states)-states[0],axis=1).max())
    info=dict(policy=method,seed=seed,path=str(destination),raw_frames=len(frames),unique_raw_frames=unique,
        encoded_frames=count,duration_ms=duration,mean_adjacent_pixel_difference=float(diffs.mean()),
        changed_frame_pairs=int(np.sum(diffs>0)),joint_displacement_rad=movement,file_bytes=destination.stat().st_size,
        fps=fps,mp4_fallback_reason=fallback,sha256=sha(destination),model_path=str(model_path) if model_path else None,
        model_sha256=sha(model_path) if model_path else None,training_seed=model.seed if model else None)
    assert len(frames)==250 and abs(duration-250*1000/fps)<30
    if method=='ikpd':assert unique>30 and count>20 and movement>.1 and info['changed_frame_pairs']>30
    atomic_json(destination.with_suffix('.json'),info);write_csv(destination.with_suffix('.csv'),trace)
    print('ANIMATION VERIFIED',json.dumps(info,ensure_ascii=False),flush=True);return info

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config');p.add_argument('--policy',choices=['ikpd','bc','bc_ppo','ppo'],default='ikpd')
    p.add_argument('--model');p.add_argument('--seed',type=int,default=7100);p.add_argument('--fps',type=int,default=50);p.add_argument('--format',choices=['gif','mp4'],default='gif')
    a=p.parse_args();record(config(a.config),a.policy,a.model,a.seed,a.fps,a.format)

if __name__=='__main__':main()
