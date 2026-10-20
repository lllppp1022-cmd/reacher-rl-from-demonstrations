"""Unified original-reward evaluation of two fixed 100k checkpoints."""
from pathlib import Path
import csv
import json
import numpy as np
import gymnasium as gym
import torch
from stable_baselines3 import PPO
from evaluate_reacher_200k import solve_ik,pd_action,write_csv,THRESHOLD,N_STEPS
from validate_custom_100k import MODELS,CASES,OUT,ROOT

ORIGINAL='PPO-Original-100k'
CUSTOM='PPO-Distance3-100k'

def evaluate(seeds,policies,models,label):
    env=gym.make('Reacher-v5',max_episode_steps=250,reward_dist_weight=1.,reward_control_weight=1.)
    assert env.spec.max_episode_steps==N_STEPS==250
    assert env.observation_space.shape==(10,) and env.action_space.shape==(2,)
    assert np.all(env.action_space.low==-1) and np.all(env.action_space.high==1)
    assert np.isclose(env.unwrapped.dt,.02)
    assert env.unwrapped._reward_dist_weight==env.unwrapped._reward_control_weight==1.
    gear=env.unwrapped.model.actuator_gear[:2,0].copy()
    assert np.all(gear!=0)
    rows,traces=[],[]
    try:
        for seed in seeds:
            ref=None
            for name in policies:
                obs,_=env.reset(seed=seed); env.action_space.seed(seed)
                state=np.r_[obs,env.unwrapped.data.qpos.copy(),env.unwrapped.data.qvel.copy(),
                            env.unwrapped.get_body_com('target').copy()]
                if ref is None: ref=state
                else: np.testing.assert_allclose(state,ref,atol=1e-12,rtol=0)
                initial=float(np.linalg.norm(obs[8:10])); previous=obs[8:10].copy()
                projected=False
                if name=='IK+PD':
                    target_q,projected=solve_ik(env.unwrapped.get_body_com('target')[:2].copy(),
                                              np.arctan2(obs[2:4],obs[:2]))
                distances=[];actions=[];speeds=[];rewards=[]
                for step in range(N_STEPS):
                    if name=='IK+PD': action=pd_action(obs,target_q,gear)
                    elif name in models: action,_=models[name].predict(obs,deterministic=True)
                    elif name=='Random': action=env.action_space.sample()
                    else: action=np.zeros(2)
                    action=np.clip(np.asarray(action,dtype=np.float32),env.action_space.low,env.action_space.high)
                    obs,reward,terminated,truncated,_=env.step(action)
                    assert obs.shape==(10,) and np.isfinite(obs).all() and np.isfinite(reward)
                    assert np.all(np.abs(action)<=1)
                    relative=obs[8:10].copy()
                    speed=float(np.linalg.norm(relative-previous)/.02)
                    previous=relative
                    d=float(np.linalg.norm(relative))
                    distances.append(d); actions.append(action.copy());speeds.append(speed);rewards.append(float(reward))
                    if label=='standard' and seed in CASES:
                        traces.append(dict(policy=name,seed=seed,step=step+1,time_s=(step+1)*.02,
                            distance_cm=d*100,error_x_m=float(relative[0]),error_y_m=float(relative[1]),
                            tip_speed_m_s=speed,action_1=float(action[0]),action_2=float(action[1]),
                            action_norm=float(np.linalg.norm(action)),reward=float(reward)))
                    if terminated or truncated: break
                if len(distances)!=250 or terminated or not truncated:
                    raise RuntimeError(f'Unexpected end {name} seed={seed}: {len(distances)} {terminated} {truncated}')
                ds=np.array(distances); act=np.array(actions); speed=np.array(speeds)
                hit=np.flatnonzero(ds<THRESHOLD); hit5=np.flatnonzero(ds<.05)
                near4=ds<.04; near1=ds<.01
                rows.append(dict(policy=name,seed=seed,initial_distance_m=initial,
                    trajectory_mean_m=float(ds.mean()),final_distance_m=float(ds[-1]),closest_distance_m=float(ds.min()),
                    action_sq_sum=float(np.sum(np.sum(act**2,axis=1).astype(np.float64))),
                    action_norm_mean=float(np.linalg.norm(act,axis=1).astype(np.float64).mean()),
                    reward_sum=float(np.sum(rewards)),final_success_1cm=int(ds[-1]<THRESHOLD),
                    stable_success_1cm=int(np.all(ds[-5:]<THRESHOLD)),ever_success_1cm=int(bool(len(hit))),
                    ik_goal_projected=int(projected),executed_steps=len(ds),terminated=int(terminated),truncated=int(truncated),
                    first_hit_1cm_s=float((hit[0]+1)*.02) if len(hit) else '',
                    first_hit_5cm_s=float((hit5[0]+1)*.02) if len(hit5) else '',
                    final_50_mean_cm=float(ds[-50:].mean()*100),final_50_std_cm=float(ds[-50:].std()*100),
                    final_50_speed_m_s=float(speed[-50:].mean()),rebound_after_min_cm=float((ds[ds.argmin():].max()-ds.min())*100),
                    action_1_near_limit_pct=float(np.mean(np.abs(act[:,0])>=.95)*100),
                    action_2_near_limit_pct=float(np.mean(np.abs(act[:,1])>=.95)*100),
                    either_action_near_limit_pct=float(np.mean(np.any(np.abs(act)>=.95,axis=1))*100),
                    max_abs_action_1=float(np.abs(act[:,0]).max()),max_abs_action_2=float(np.abs(act[:,1]).max()),
                    max_action_norm=float(np.linalg.norm(act,axis=1).max()),
                    near_4cm_steps=int(near4.sum()),near_4cm_speed_sum=float(speed[near4].sum()),
                    near_4cm_speed_m_s=float(speed[near4].mean()) if near4.any() else '',
                    near_1cm_steps=int(near1.sum()),near_1cm_speed_sum=float(speed[near1].sum()),
                    near_1cm_speed_m_s=float(speed[near1].mean()) if near1.any() else '',
                    ever_but_final_fail=int(len(hit)>0 and ds[-1]>=THRESHOLD)))
            if (seed+1)%20==0: print(f'{label}: completed seed {seed}',flush=True)
    finally: env.close()
    summary=[]
    for name in policies:
        group=[r for r in rows if r['policy']==name]
        mean=lambda k:float(np.mean([r[k] for r in group]))
        s=dict(policy=name,n_episodes=len(group),mean_trajectory_cm=100*mean('trajectory_mean_m'),
            mean_final_cm=100*mean('final_distance_m'),mean_closest_cm=100*mean('closest_distance_m'),
            mean_action_sq_sum=mean('action_sq_sum'),mean_action_norm=mean('action_norm_mean'),
            mean_reward_sum=mean('reward_sum'),final_success_pct=100*mean('final_success_1cm'),
            stable_success_pct=100*mean('stable_success_1cm'),ever_success_pct=100*mean('ever_success_1cm'),
            ik_projected_count=sum(r['ik_goal_projected'] for r in group),ever_but_final_fail_pct=100*mean('ever_but_final_fail'),
            mean_final_50_cm=mean('final_50_mean_cm'),mean_final_50_std_cm=mean('final_50_std_cm'),
            mean_final_50_speed_m_s=mean('final_50_speed_m_s'),mean_rebound_after_min_cm=mean('rebound_after_min_cm'),
            mean_either_action_near_limit_pct=mean('either_action_near_limit_pct'),
            near_4cm_steps=sum(r['near_4cm_steps'] for r in group),near_1cm_steps=sum(r['near_1cm_steps'] for r in group),
            min_executed_steps=min(r['executed_steps'] for r in group),max_executed_steps=max(r['executed_steps'] for r in group))
        for threshold in ('4cm','1cm'):
            count=s[f'near_{threshold}_steps']
            s[f'near_{threshold}_speed_m_s']=sum(r[f'near_{threshold}_speed_sum'] for r in group)/count if count else ''
        summary.append(s)
    prefix='' if label=='standard' else 'holdout_'
    write_csv(OUT/f'{prefix}episodes.csv',rows); write_csv(OUT/f'{prefix}summary.csv',summary)
    return rows,summary,traces

def main():
    assert (OUT/'validation_metadata.json').exists()
    if (OUT/'episodes.csv').exists(): raise FileExistsError('Refusing to overwrite evaluation')
    torch.set_num_threads(1)
    models={name:PPO.load(ROOT/file,device='cpu') for name,file in MODELS.items()}
    rows,summary,traces=evaluate(range(100,200),('IK+PD',ORIGINAL,CUSTOM,'Random','Zero'),models,'standard')
    with (ROOT/'reacher_eval_results_250_new/episodes.csv').open(encoding='utf-8-sig') as f:
        history=list(csv.DictReader(f))
    deltas={}
    for name in ('IK+PD',ORIGINAL,'Random','Zero'):
        a=[r for r in rows if r['policy']==name]
        b=[r for r in history if r['policy']==('PPO' if name==ORIGINAL else name)]
        assert [r['seed'] for r in a]==[int(r['seed']) for r in b]
        delta=max(abs(x[k]-float(y[k])) for x,y in zip(a,b)
            for k in ('trajectory_mean_m','final_distance_m','action_sq_sum','reward_sum'))
        assert delta<1e-10,(name,delta)
        deltas[name]=delta
    (OUT/'historical_validation.json').write_text(json.dumps(deltas,indent=2),encoding='utf-8')
    write_csv(OUT/'case_timeseries.csv',traces)
    hrows,holdout,_=evaluate(range(2000,2100),('IK+PD',ORIGINAL,CUSTOM),models,'holdout')
    keys=['policy','seed','action_sq_sum','action_norm_mean','max_abs_action_1','max_abs_action_2',
          'max_action_norm','action_1_near_limit_pct','action_2_near_limit_pct','either_action_near_limit_pct',
          'near_4cm_steps','near_4cm_speed_m_s','near_1cm_steps','near_1cm_speed_m_s',
          'final_50_mean_cm','final_50_std_cm','final_50_speed_m_s','rebound_after_min_cm','ever_but_final_fail']
    write_csv(OUT/'action_diagnostics.csv',[dict(test_set=label,**{k:r[k] for k in keys})
        for label,group in [('standard',rows),('holdout',hrows)] for r in group])
    print('STANDARD',json.dumps(summary,ensure_ascii=False),flush=True)
    print('HOLDOUT',json.dumps(holdout,ensure_ascii=False),flush=True)

if __name__=='__main__': main()
