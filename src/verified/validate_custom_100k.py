"""Audit saved experiments and test custom reward equivalence over full episodes."""
from pathlib import Path
import hashlib
import inspect
import json
import sys
import gymnasium as gym
import mujoco
import numpy as np
import torch
import stable_baselines3
import matplotlib
from stable_baselines3 import PPO
from custom_reacher_env import CustomReacherEnv

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'reacher_eval_custom_100k'
MODELS={'PPO-Original-100k':'ppo_reacher_250_A.zip',
        'PPO-Distance3-100k':'ppo_custom_distance3_100k.zip'}
CASES=(123,145,178,199)  # Selected before evaluation; never selected using results.

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def protected_files():
    return sorted(set(list(ROOT.glob('*.zip'))+
        [p for folder in ROOT.glob('reacher_eval*') if folder.is_dir() and folder!=OUT
         for p in folder.rglob('*') if p.is_file()]))

def main():
    OUT.mkdir(exist_ok=False)
    needed=['custom_reacher_env.py','test_custom_reacher.py','train_custom_reacher.py',
            *MODELS.values(),'evaluate_reacher_200k.py','evaluate_reacher_250_new.py',
            'reacher_eval_results_250_new/summary.csv']
    assert all((ROOT/p).is_file() for p in needed)
    data=dict(versions=dict(python=sys.version,mujoco=mujoco.__version__,gymnasium=gym.__version__,
        numpy=np.__version__,torch=torch.__version__,stable_baselines3=stable_baselines3.__version__,
        matplotlib=matplotlib.__version__),models={},fixed_case_seeds=CASES,
        protected_sha256={str(p.relative_to(ROOT)):digest(p) for p in protected_files()},
        source_sha256={p:digest(ROOT/p) for p in ['custom_reacher_env.py','train_custom_reacher.py','train_ppo_250.py']})
    keys=['learning_rate','n_steps','batch_size','gamma','gae_lambda','n_epochs','ent_coef',
          'vf_coef','max_grad_norm','seed','policy_kwargs','normalize_advantage','target_kl',
          'use_sde','sde_sample_freq','clip_range_vf']
    for name,file in MODELS.items():
        model=PPO.load(ROOT/file,device='cpu')
        hp={k:getattr(model,k) for k in keys}
        hp['clip_range']=model.clip_range(1.)
        hp['optimizer_class']=model.policy.optimizer_class.__name__
        hp['optimizer_kwargs']=model.policy.optimizer_kwargs
        assert model.num_timesteps==100352
        assert model.observation_space.shape==(10,) and model.action_space.shape==(2,)
        assert np.all(model.action_space.low==-1) and np.all(model.action_space.high==1)
        assert all(torch.isfinite(v).all() for v in model.policy.state_dict().values())
        data['models'][name]=dict(path=str(ROOT/file),sha256=digest(ROOT/file),
            num_timesteps=model.num_timesteps,observation_space=str(model.observation_space),
            action_space=str(model.action_space),hyperparameters=hp,network=str(model.policy))
        print(name,json.dumps(data['models'][name],ensure_ascii=False,indent=2),flush=True)
    a,b=data['models'].values()
    data['hyperparameter_differences']={k:[a['hyperparameters'][k],b['hyperparameters'][k]]
        for k in a['hyperparameters'] if a['hyperparameters'][k]!=b['hyperparameters'][k]}
    assert not data['hyperparameter_differences']
    assert a['network']==b['network']
    checks=[]
    for seed in (11,42,123,999):
        envs=[gym.make('Reacher-v5',max_episode_steps=250,reward_dist_weight=1.,reward_control_weight=1.),
              CustomReacherEnv(1.,0.),CustomReacherEnv(3.,0.)]
        max_obs=max_reward=max_dyn=0.
        try:
            obs=[e.reset(seed=seed)[0] for e in envs]
            for o in obs[1:]: np.testing.assert_array_equal(o,obs[0])
            rng=np.random.default_rng(seed+10000)
            for step,action in enumerate(rng.uniform(-1,1,(250,2)).astype(np.float32)):
                results=[e.step(action.copy()) for e in envs]
                o0,r0,t0,tr0,i0=results[0]
                for idx,(o,r,t,tr,info) in enumerate(results):
                    assert o.shape==(10,) and envs[idx].observation_space.contains(o)
                    np.testing.assert_allclose(o,o0,atol=1e-12,rtol=0)
                    assert t==t0==False and tr==tr0==(step==249)
                    expected=r0+(2*i0['reward_dist'] if idx==2 else 0.)
                    np.testing.assert_allclose(r,expected,atol=1e-12,rtol=0)
                    max_obs=max(max_obs,float(np.max(np.abs(o-o0))))
                    max_reward=max(max_reward,abs(r-expected))
                    if idx:
                        assert info['reward_velocity_extra']==0.
                        base=envs[idx].base_env.unwrapped
                        for attr in ('qpos','qvel'):
                            dv=np.max(np.abs(getattr(base.data,attr)-getattr(envs[0].unwrapped.data,attr)))
                            assert dv==0
                            max_dyn=max(max_dyn,float(dv))
            assert envs[0].spec.max_episode_steps==250
            for e in envs[1:]:
                assert e.base_env.spec.max_episode_steps==250 and e.dt==.02
            if seed==11:
                data['base_step_source']=inspect.getsource(type(envs[0].unwrapped).step)
                data['base_observation_source']=inspect.getsource(type(envs[0].unwrapped)._get_obs)
                data['base_class']=str(type(envs[0].unwrapped))
            checks.append(dict(seed=seed,steps=250,max_observation_delta=max_obs,
                max_reward_formula_delta=max_reward,max_physics_delta=max_dyn,passed=True))
        finally:
            for e in envs: e.close()
    data['environment_checks']=checks
    data['training_provenance_note']='Checkpoint confirms steps, seed and PPO config; reward weights and episode length are supported by current training source and consistency tests, not embedded immutable training provenance.'
    (OUT/'validation_metadata.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    print('完整回合一致性验证通过:',json.dumps(checks,ensure_ascii=False),flush=True)

if __name__=='__main__':
    main()
