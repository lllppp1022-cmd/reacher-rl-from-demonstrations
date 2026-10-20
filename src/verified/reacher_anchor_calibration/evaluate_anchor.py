"""Batched policy inference, independent identical seeded MuJoCo episodes and cached evaluations."""
import argparse
from anchor_common import *

def load_policy(path):
    if not path:return None
    path=Path(path)
    if path.stem.startswith('step'):
        from anchor_ppo import AnchorPPO
        return AnchorPPO.load(path,device='cpu')
    return load_bc(path)

def summarize(rows,c):
    mean=lambda key:float(np.mean([r[key] for r in rows]))
    near_count=sum(r['near_1cm_steps'] for r in rows)
    return dict(n_episodes=len(rows),mean_final_cm=100*mean('final_distance_m'),
        mean_trajectory_cm=100*mean('trajectory_mean_m'),mean_action_sq_sum=mean('action_sq_sum'),
        stable_success_pct=100*mean('stable_success_1cm'),final_success_pct=100*mean('final_success_1cm'),
        ever_success_pct=100*mean('ever_success_1cm'),mean_final_50_cm=mean('final_50_mean_cm'),
        mean_final_50_std_cm=mean('final_50_std_cm'),mean_final_50_speed_m_s=mean('final_50_speed_m_s'),mean_tip_speed_m_s=mean('tip_speed_m_s'),
        near_1cm_speed_m_s=sum(r['near_1cm_speed_sum'] for r in rows)/near_count if near_count else None,
        early_stop_outside_pct=100*mean('early_stop_outside'),mean_reward_sum=mean('reward_sum'),
        action_near_limit_pct=mean('action_near_limit_pct'),mean_action_delta=mean('action_delta_mean'),
        first_hit_conditional_s=float(np.mean([r['first_hit_1cm_s'] for r in rows if r['first_hit_1cm_s'] is not None]))
            if any(r['first_hit_1cm_s'] is not None for r in rows) else None)

def evaluate(c,job,path,seeds,split='diagnostic',expert=False,extra=None):
    dest=out(c)/'results'/split/job;dest.mkdir(parents=True,exist_ok=True)
    complete=dest/'complete.json';model_sha=sha(path) if path else None;seeds=list(seeds)
    if complete.exists():
        record=read(complete);assert record['model_sha256']==model_sha and record['seeds']==seeds
        return read(dest/'summary.json')
    model=load_policy(path);environments=[env() for _ in seeds];experts=[Expert() for _ in seeds] if expert else None
    observations=[];initial_states=[];initial_errors=[];targets=[]
    for i,(e,seed) in enumerate(zip(environments,seeds)):
        o,_=e.reset(seed=seed);observations.append(o);initial_errors.append(o[8:10].copy())
        initial_states.append(np.r_[o,e.unwrapped.data.qpos,e.unwrapped.data.qvel])
        targets.append(e.unwrapped.get_body_com('target').copy())
        if expert:experts[i].reset(e,o)
    obs=np.array(observations);xy=[];acts=[];mus=[];rewards=[];states=[];steps=250
    start=time.perf_counter()
    try:
        for step in range(steps):
            states.append(obs.copy())
            if expert:mu=np.array([x.action(o) for x,o in zip(experts,obs)])
            else:
                with torch.no_grad():mu=actor(model,obs).cpu().numpy()
            actions=np.clip(mu,-1,1).astype(np.float32);new=[];r=[]
            for i,e in enumerate(environments):
                o,reward,term,trunc,_=e.step(actions[i]);assert not term and trunc==(step==249)
                assert np.isfinite(o).all() and np.isfinite(reward)
                new.append(o);r.append(reward)
            obs=np.array(new);xy.append(obs[:,8:10].copy());acts.append(actions.copy());mus.append(mu.copy());rewards.append(r)
    finally:
        for e in environments:e.close()
    xy=np.array(xy);actions=np.array(acts);mu=np.array(mus);rewards=np.array(rewards)
    distances=np.linalg.norm(xy,axis=-1)
    speed=np.linalg.norm(np.diff(np.concatenate([np.array(initial_errors)[None],xy]),axis=0),axis=-1)/.02
    rows=[]
    for i,seed in enumerate(seeds):
        ds=distances[:,i];hit=np.flatnonzero(ds<.01);v=speed[:,i];a=actions[:,i]
        near=ds<.01;criteria=c['early_stop_outside'];n=criteria['last_steps']
        rows.append(dict(seed=seed,trajectory_mean_m=float(ds.mean()),final_distance_m=float(ds[-1]),
            closest_distance_m=float(ds.min()),action_sq_sum=float(np.square(a).sum(axis=1,dtype=np.float32).astype(np.float64).sum()),
            stable_success_1cm=int(np.all(ds[-5:]<.01)),final_success_1cm=int(ds[-1]<.01),ever_success_1cm=int(bool(len(hit))),
            first_hit_1cm_s=float((hit[0]+1)*.02) if len(hit) else None,
            final_50_mean_cm=float(ds[-50:].mean()*100),final_50_std_cm=float(ds[-50:].std()*100),
            tip_speed_m_s=float(v.mean()),final_50_speed_m_s=float(v[-50:].mean()),near_1cm_steps=int(near.sum()),near_1cm_speed_sum=float(v[near].sum()),
            early_stop_outside=int(np.all(ds[-n:]>criteria['distance_m']) and v[-n:].mean()<criteria['speed_m_s']),
            action_near_limit_pct=float(np.mean(np.any(np.abs(a)>=.95,axis=-1))*100),
            action_delta_mean=float(np.linalg.norm(np.diff(a,axis=0),axis=-1).mean()),
            reward_sum=float(rewards[:,i].sum()),executed_steps=250,terminated=0,truncated=1))
    summary=dict(job=job,path=str(path) if path else None,model_sha256=model_sha,split=split,**(extra or {}),**summarize(rows,c))
    write_csv(dest/'episodes.csv',rows)
    np.savez_compressed(dest/'trajectories.npz',seeds=seeds,xy_error_m=xy,actions=actions,raw_mu=mu,
        observations=np.array(states),distance_m=distances,speed_m_s=speed,rewards=rewards,
        initial_states=initial_states,target_positions=targets)
    atomic_json(dest/'summary.json',summary)
    atomic_json(complete,dict(model_sha256=model_sha,seeds=seeds,completed_unix=time.time(),elapsed_seconds=time.perf_counter()-start))
    print('EVALUATED',split,job,'stable',summary['stable_success_pct'],'final_cm',round(summary['mean_final_cm'],4),flush=True)
    return summary

def checkpoint_jobs(c,phase='short',groups=None):
    groups=groups or list(c['groups']);jobs=[]
    for g in groups:
        for seed in c['training_seeds']:
            info=read(folder(c,g,seed,phase)/'complete.json')
            for snap in info['snapshots']:
                jobs.append(dict(job=f'{phase}_{g}_seed{seed}_step{snap["step"]}',path=snap['path'],
                    group=g,training_seed=seed,phase=phase,step=snap['step']))
    return jobs

def early(c,groups=None,phase='short'):
    jobs=checkpoint_jobs(c,phase,groups)
    diagnostic=range(c['diagnostic_seeds'][0],c['diagnostic_seeds'][1]+1)
    validation=range(c['validation_seeds'][0],c['validation_seeds'][1]+1)
    for j in jobs:
        evaluate(c,j['job'],j['path'],diagnostic,'diagnostic',extra={k:v for k,v in j.items() if k not in ('job','path')})
        # Validation only zero/final for short screening; all saved nodes for chosen long runs.
        budget=c['long_steps' if phase.startswith('long') else 'short_steps']
        if phase.startswith('long') or j['step'] in (0,budget):
            evaluate(c,j['job'],j['path'],validation,'validation',extra={k:v for k,v in j.items() if k not in ('job','path')})
    if phase=='short' and (groups is None or set(groups)>=set(c['groups'])):finish(c,'evaluate_short',checkpoint_jobs=len(jobs))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config');p.add_argument('--groups',nargs='+');a=p.parse_args();early(config(a.config),a.groups)
