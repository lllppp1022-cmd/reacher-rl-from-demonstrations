"""Independent BC closed loop, optional one-round training-only DAgger and matched tests."""
import argparse
from bc_common import *
import evaluate_custom_100k as verified
from train_bc import train_one

def evaluate(c,split,seeds,policies):
    summaries=[];episodes=[];traces=[]
    verified.CASES=(7000,7010,7050,7099)
    for name,path in policies.items():
        folder=out(c)/'results'/split/name;folder.mkdir(parents=True,exist_ok=True)
        checksum=sha(path) if path else None
        if (folder/'complete.json').exists():
            assert read(folder/'complete.json')['model_sha256']==checksum
            er=csvread(folder/'episodes.csv');sr=csvread(folder/'summary.csv')
        else:
            verified.OUT=folder
            model=load_bc(path) if path else None
            er,sr,ts=verified.evaluate(seeds,(name,),{name:model} if model else {},'standard')
            hit=[float(r['first_hit_1cm_s']) for r in er if r['first_hit_1cm_s']!='']
            sr[0]['conditional_first_hit_1cm_mean_s']=float(np.mean(hit)) if hit else ''
            sr[0]['n_first_hits']=len(hit)
            write_csv(folder/'summary.csv',sr)
            if ts:write_csv(folder/'timeseries.csv',ts)
            atomic_json(folder/'complete.json',dict(model_sha256=checksum,seeds=[seeds.start,seeds.stop-1],completed_unix=time.time()))
        summaries += [dict(test_set=split,**r) for r in sr];episodes += [dict(test_set=split,**r) for r in er]
        if (folder/'timeseries.csv').exists():traces+=csvread(folder/'timeseries.csv')
        print('EVALUATED',split,name,'stable',sr[0]['stable_success_pct'],'final_cm',sr[0]['mean_final_cm'],flush=True)
    write_csv(out(c)/f'results/{split}_summary.csv',summaries);write_csv(out(c)/f'results/{split}_episodes.csv',episodes)
    if traces:write_csv(out(c)/f'results/{split}_timeseries.csv',traces)
    return summaries

def dagger(c,bc_models):
    destination=out(c)/'datasets/dagger.npz'
    if destination.exists():return
    models=[load_bc(m['path']) for m in bc_models];e=env();expert=Expert();rows={}
    try:
        for i in range(c['dagger_episodes']):
            seed=c['expert_train_seeds'][0]+i%(c['expert_train_seeds'][1]-c['expert_train_seeds'][0]+1)
            obs,_=e.reset(seed=seed);expert.reset(e,obs);model=models[i%len(models)]
            for step in range(250):
                before=obs.copy();qpos=e.unwrapped.data.qpos.copy();qvel=e.unwrapped.data.qvel.copy()
                label=expert.action(obs);action=model.predict(obs,deterministic=True)[0]
                obs,r,term,trunc,_=e.step(action)
                row=dict(observation=before.astype(np.float32),expert_action=label,next_observation=obs.astype(np.float32),
                    reward=r,terminated=term,truncated=trunc,episode_id=i,step=step,seed=seed,distance=float(np.linalg.norm(obs[8:10])),
                    qpos=qpos,qvel=qvel,expert_target_q=expert.target_q.copy(),ik_projected=expert.projected,executed_student_action=action)
                for k,v in row.items():rows.setdefault(k,[]).append(v)
        np.savez_compressed(destination,**{k:np.array(v) for k,v in rows.items()})
    finally:e.close()
    original=dict(np.load(out(c)/'datasets/expert_train.npz'));extra=dict(np.load(destination))
    combined={k:np.concatenate([original[k],extra[k]],axis=0) for k in original}
    # Episodes remain uniquely identifiable; states and labels remain training-only.
    combined['episode_id'][len(original['episode_id']):]+=int(original['episode_id'].max())+1
    np.savez_compressed(out(c)/'datasets/expert_train_dagger.npz',**combined)

def main(c):
    setup(c);require(c,'bc');seed_everything(42)
    if done(c,'evaluate_bc').exists():return
    models=read(out(c)/'models/bc/manifest.json')
    validation=evaluate(c,'bc_validation',range(c['expert_validation_seeds'][0],c['expert_validation_seeds'][1]+1),
        {f"BC-seed{m['seed']}":m['path'] for m in models})
    trigger=max(float(r['stable_success_pct']) for r in validation)<c['dagger_if_best_stable_pct_below']
    corrected=[]
    if trigger and c['dagger_episodes']:
        print('DAGGER TRIGGERED: fixed one-round training-only correction',flush=True);dagger(c,models)
        for seed in c['training_seeds']:
            corrected.append(train_one(c,seed,out(c)/'datasets/expert_train_dagger.npz',out(c)/'datasets/expert_validation.npz',
                out(c)/f'models/bc_dagger/seed{seed}'))
        val_corrected=evaluate(c,'dagger_validation',range(c['expert_validation_seeds'][0],c['expert_validation_seeds'][1]+1),
            {f"BC-DAgger-seed{m['seed']}":m['path'] for m in corrected})
    else:val_corrected=[]
    if corrected:atomic_json(out(c)/'models/bc_dagger/manifest.json',corrected)
    active=[]
    for m in models:
        score=lambda r:(-float(r['stable_success_pct']),float(r['mean_final_cm']))
        initial=next(r for r in validation if r['policy']==f"BC-seed{m['seed']}")
        candidate=next((r for r in val_corrected if r['policy']==f"BC-DAgger-seed{m['seed']}"),None)
        if candidate and score(candidate)<score(initial):
            chosen=next(x for x in corrected if x['seed']==m['seed']);active.append(dict(**chosen,variant='BC-DAgger'))
        else:active.append(dict(**m,variant='BC'))
    atomic_json(out(c)/'models/bc/active_manifest.json',active)
    policies={'IK+PD':None,'Zero':None,'PPO-200k':c['baseline_ppo_200k'],'PPO-300k':c['baseline_ppo_300k']}
    policies.update({f'PPO-100k-seed{s}':p for s,p in c['pure_ppo_100k_paths'].items()})
    policies.update({f"BC-seed{m['seed']}":m['path'] for m in models})
    policies.update({f"BC-DAgger-seed{m['seed']}":m['path'] for m in corrected})
    summaries=evaluate(c,'bc_selection',range(c['selection_seeds'][0],c['selection_seeds'][1]+1),policies)
    write_csv(out(c)/'results/bc_closed_loop_results.csv',summaries)
    finish(c,'evaluate_bc',dict(dagger_triggered=trigger,dagger_episodes=c['dagger_episodes'] if trigger else 0,
        active_bc=active,selection_models=len(policies)))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config');a=p.parse_args();main(config(a.config))
