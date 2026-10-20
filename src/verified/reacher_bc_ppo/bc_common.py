"""Shared experiment utilities and an SB3-compatible, privileged-input-free actor."""
from pathlib import Path
import sys
import os
import json
import hashlib
import csv
import time
import numpy as np
import gymnasium as gym
import torch
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from evaluate_reacher_200k import solve_ik,pd_action,wrap_angle

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def atomic_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    try:
        # Windows readers can briefly prevent replacement of a live status file.
        # Keep the old complete JSON visible until the atomic replacement succeeds.
        for attempt in range(8):
            try:
                os.replace(temp,path)
                break
            except PermissionError:
                if attempt==7:raise
                time.sleep(.05*(attempt+1))
    finally:
        temp.unlink(missing_ok=True)
def save_model(model,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.stem+f'.{os.getpid()}.tmp.zip');model.save(temp)
    # Windows virus scanners can briefly hold a just-created ZIP open.
    for attempt in range(8):
        try:os.replace(temp,path);break
        except PermissionError:
            if attempt==7:raise
            time.sleep(.05*(attempt+1))
def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def csvread(path):
    with Path(path).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def config(path=None):
    path=Path(path or HERE/'configs/experiment.json').resolve();c=read(path)
    c['_config_path']=str(path);c['output_root']=str(Path(c['output_root']).resolve());return c
def out(c):return Path(c['output_root'])
def env(render_mode=None,**kwargs):
    return gym.make('Reacher-v5',max_episode_steps=250,reward_dist_weight=1.,reward_control_weight=1.,render_mode=render_mode,**kwargs)
def setup(c):
    base=out(c)
    intervals=[set(range(c[k][0],c[k][1]+1)) for k in ['expert_train_seeds','expert_validation_seeds','selection_seeds','final_seeds']]
    assert all(not (a&b) for i,a in enumerate(intervals) for b in intervals[i+1:]),'Data seed ranges overlap'
    for name in ['audit','datasets','models/bc','models/bc_ppo','results','plots','videos','logs','docs','configs','tests','live']:(base/name).mkdir(parents=True,exist_ok=True)
    frozen=base/'configs/experiment_frozen.json'
    if frozen.exists():
        assert read(frozen)==c,'Configuration changed: choose a fresh output_root for a new experiment'
    else:atomic_json(frozen,c)
def done(c,stage):return out(c)/'logs'/f'{stage}_complete.json'
def finish(c,stage,info):atomic_json(done(c,stage),dict(completed_unix=time.time(),**info))
def require(c,stage):assert done(c,stage).exists(),f'Required stage is incomplete: {stage}'
def seed_everything(seed):
    np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(1)

class InputNormalizer(BaseFeaturesExtractor):
    """Fixed statistics from expert TRAIN observations only, serialized with the policy."""
    def __init__(self,observation_space,mean,std):
        super().__init__(observation_space,features_dim=10)
        self.register_buffer('obs_mean',torch.tensor(mean,dtype=torch.float32))
        self.register_buffer('obs_std',torch.tensor(std,dtype=torch.float32))
    def forward(self,observations):return (observations-self.obs_mean)/self.obs_std

def make_ppo(c,seed,mean=None,std=None,env_instance=None):
    kwargs={}
    if mean is not None:
        kwargs=dict(features_extractor_class=InputNormalizer,
                    features_extractor_kwargs=dict(mean=list(mean),std=list(std)))
    return PPO('MlpPolicy',env_instance or env(),learning_rate=3e-4,n_steps=1024,batch_size=64,
        gamma=.99,gae_lambda=.95,clip_range=.2,n_epochs=10,seed=seed,device='cpu',
        policy_kwargs=kwargs,verbose=0,tensorboard_log=str(out(c)/'logs/tensorboard'))

class Expert:
    def reset(self,e,obs):
        self.target_q,self.projected=solve_ik(e.unwrapped.get_body_com('target')[:2].copy(),np.arctan2(obs[2:4],obs[:2]))
        self.gear=e.unwrapped.model.actuator_gear[:2,0].copy()
    def action(self,obs):return np.clip(pd_action(obs,self.target_q,self.gear),-1,1).astype(np.float32)

def actor(model,obs):
    tensor=torch.as_tensor(obs,dtype=torch.float32)
    features=model.policy.extract_features(tensor)
    if isinstance(features,tuple):features=features[0]
    latent=model.policy.mlp_extractor.forward_actor(features)
    return model.policy.action_net(latent)
def actor_parameters(model):return list(model.policy.mlp_extractor.policy_net.parameters())+list(model.policy.action_net.parameters())
def load_bc(path):
    path=Path(path)
    if path.suffix=='.pt':
        payload=torch.load(path,map_location='cpu',weights_only=False)
        # Standalone Actor reconstruction: no sibling ZIP, privileged inputs or critic weights needed.
        e=env()
        model=PPO('MlpPolicy',e,seed=payload['seed'],device='cpu',policy_kwargs=dict(
            features_extractor_class=InputNormalizer,features_extractor_kwargs=dict(
                mean=payload['observation_mean'],std=payload['observation_std'])))
        incompatible=model.policy.load_state_dict(payload['actor_state'],strict=False)
        assert not incompatible.unexpected_keys
        assert not any('policy_net' in key or 'action_net' in key for key in incompatible.missing_keys)
        e.close();model.env=None
        return model
    return PPO.load(path,device='cpu')
def policy_path(c,mode,seed=None):
    defaults=out(c)/'results/representative_models.json'
    if seed is None and defaults.exists() and mode in read(defaults):return Path(read(defaults)[mode])
    seed=seed or c['training_seeds'][0]
    if mode=='bc':
        active=out(c)/'models/bc/active_manifest.json'
        if active.exists():return Path(next(m['path'] for m in read(active) if m['seed']==seed))
        return out(c)/f'models/bc/seed{seed}/bc_best.zip'
    if mode=='bc_ppo':return out(c)/f'models/bc_ppo/seed{seed}/final.zip'
    return Path(c['baseline_ppo_200k'])
