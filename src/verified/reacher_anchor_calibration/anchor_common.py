"""Isolated anchoring IO and protocol; import old physics and utilities read-only."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
REFERENCE=ROOT/'reacher_safe_finetune'
if str(REFERENCE) not in sys.path:sys.path.insert(0,str(REFERENCE))
from safe_common import *
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent

def config(path=None):
    path=Path(path or HERE/'configs/experiment.json').resolve()
    c=read(path);c['_config_path']=str(path)
    for key in ['output_root','source_root','implementation_reference']:c[key]=str(Path(c[key]).resolve())
    return c

def effective_beta(settings,step,n_steps=1024):
    beta=settings['anchor_beta']
    if settings.get('beta_schedule','fixed')=='linear':
        beta*=1.-.9*min(1.,max(0.,step-20480)/max(1,settings['total_steps']-20480))
    return float(beta)

def phase_settings(c,group,phase):
    s=dict(c['groups'][group]);
    if s['anchor_beta'] is None:s['anchor_beta']=read(out(c)/'configs/beta_calibration.json')['selected_betas'][group]
    s['total_steps']=c['long_steps'] if phase.startswith('long') else c['short_steps']
    s['beta_schedule']='linear' if phase=='long_decay' else 'fixed'
    return s

def publish(c,path,group,seed,step,phase):
    settings=phase_settings(c,group,phase)
    evaluation=out(c)/f'results/validation/{phase}_{group}_seed{seed}_step{step}/summary.json'
    atomic_json(out(c)/'live/current.json',dict(path=str(path),sha256=sha(path),group=group,seed=seed,steps=step,phase=phase,
        bc_path=source(c,seed)['path'],evaluation=str(evaluation),beta_initial=settings['anchor_beta'],
        beta_effective=effective_beta(settings,step),beta_schedule=settings['beta_schedule'],published_unix=time.time()))


def mean_actor_hash(model):
    h=hashlib.sha256()
    for n,p in model.policy.named_parameters():
        if 'policy_net' in n or 'action_net' in n:
            h.update(n.encode());h.update(str(tuple(p.shape)).encode());h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()
