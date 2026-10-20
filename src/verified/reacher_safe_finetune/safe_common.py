"""Reuse the verified physics and actors, with a separate immutable experiment root."""
from pathlib import Path
import sys, os, json, time, csv, hashlib
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OLD=ROOT/'reacher_bc_ppo'
if str(OLD) not in sys.path:sys.path.insert(0,str(OLD))
import bc_common as base
from bc_common import atomic_json, read, sha, write_csv, csvread, env, Expert, load_bc, actor, seed_everything, save_model

def config(path=None):
    path=Path(path or HERE/'configs/experiment.json').resolve()
    c=read(path);c['_config_path']=str(path)
    c['output_root']=str(Path(c['output_root']).resolve())
    c['source_root']=str(Path(c['source_root']).resolve())
    return c

def out(c):return Path(c['output_root'])

def setup(c):
    for name in ['configs','models','checkpoints','results','plots','videos','logs','docs','tests','reports','live']:
        (out(c)/name).mkdir(parents=True,exist_ok=True)
    frozen=out(c)/'configs/protocol_frozen.json'
    if frozen.exists():assert read(frozen)==c,'Choose a new output_root when changing the protocol'
    else:atomic_json(frozen,c)

def source(c,seed,kind='bc_dagger'):
    root=Path(c['source_root'])
    manifest=root/('models/bc/manifest.json' if kind=='bc' else f'models/{kind}/manifest.json')
    m=next(x for x in read(manifest) if x['seed']==seed)
    assert sha(m['path'])==m['sha256'],f'Historical model hash mismatch: {m["path"]}'
    return m

def group_config(c,group):
    if group in c['groups']:return c['groups'][group]
    choice=read(out(c)/'configs/anchor_plan.json')
    return choice['groups'][group]

def folder(c,group,seed,phase='short'):
    return out(c)/f'checkpoints/{phase}/{group}/seed{seed}'

def reference(c):
    return np.load(out(c)/'configs/diagnostic_observations.npz')['observation']

def actor_vector(model):
    return torch.cat([p.detach().flatten().cpu() for n,p in model.policy.named_parameters()
                      if n=='log_std' or 'policy_net' in n or 'action_net' in n])

def mean_actor_vector(model):
    return torch.cat([p.detach().flatten().cpu() for n,p in model.policy.named_parameters()
                      if 'policy_net' in n or 'action_net' in n])

def publish(c,path,group,seed,step,phase):
    score=out(c)/f'results/validation/{phase}_{group}_seed{seed}_step{step}/summary.json'
    atomic_json(out(c)/'live/current.json',dict(path=str(path),sha256=sha(path),group=group,seed=seed,steps=step,
        phase=phase,bc_path=source(c,seed)['path'],evaluation=str(score),published_unix=time.time()))

def marker(c,stage):return out(c)/f'logs/{stage}_complete.json'
def finish(c,stage,**info):atomic_json(marker(c,stage),dict(completed_unix=time.time(),**info))
def require(c,stage):assert marker(c,stage).exists(),f'Run stage {stage} first'

def append_jsonl(path,row):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')

def jsonl(path):
    path=Path(path)
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line] if path.exists() else []

def finite_number(x):
    x=float(x)
    return x if np.isfinite(x) else None
