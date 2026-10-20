"""Portable I/O adapter around the preserved, audited implementations."""
from pathlib import Path
import sys,json,shutil
ROOT=Path(__file__).resolve().parents[1]
VERIFIED=ROOT/'src/verified'
for folder in [VERIFIED,VERIFIED/'reacher_bc_ppo',VERIFIED/'reacher_safe_finetune',VERIFIED/'reacher_anchor_calibration']:
    sys.path.insert(0,str(folder))
from anchor_common import read,atomic_json,sha
from bc_common import load_bc
def registry():return read(ROOT/'configs/models.json')
def model(method='long_decay_strong',seed=42):
    entry=next(r for r in registry() if r['method']==method and r['training_seed']==seed)
    p=ROOT/entry['path']
    if not p.exists():raise FileNotFoundError(f'Provision authorized weights first: {entry["path"]}; expected SHA256={entry["sha256"]}')
    assert sha(p)==entry['sha256'];return p
def initialize():
    c=read(ROOT/'configs/anchor_template.json');dest=ROOT/'runtime'
    # Missing assets should identify the authorized model path and expected hash first.
    for entry in registry():model(entry['method'],entry['training_seed'])
    provenance=ROOT/'reports/asset_provenance.csv'
    if provenance.exists():
        import csv
        with provenance.open(encoding='utf-8-sig') as f:
            for asset in csv.DictReader(f):
                if asset['path'].endswith('diagnostic_observations.npz'):
                    if not (ROOT/asset['path']).exists():
                        raise FileNotFoundError(f"Provision authorized training-only reference data: {asset['path']}; expected SHA256={asset['sha256']}. From-scratch pipeline and results do not require it.")
                    assert sha(ROOT/asset['path'])==asset['sha256'],'Training-only reference data hash mismatch'
    c.update(output_root=str(dest),source_root=str(dest/'source'),implementation_reference=str(VERIFIED/'reacher_safe_finetune'))
    manifest=[]
    for seed in c['training_seeds']:
        p=model('BC-DAgger',seed);m=load_bc(p);ex=m.policy.features_extractor
        manifest.append(dict(seed=seed,path=str(p),sha256=sha(p),observation_mean=ex.obs_mean.cpu().numpy().tolist(),observation_std=ex.obs_std.cpu().numpy().tolist()))
    atomic_json(dest/'source/models/bc_dagger/manifest.json',manifest)
    for n in ['diagnostic_observations.npz','beta_calibration.json']:
        (dest/'configs').mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/'configs'/n,dest/'configs'/n)
    for r in registry():
        if r['steps']==0:continue
        group='b0' if r['method']=='long_control_b0' else 'medium' if r['method'].endswith('medium') else 'strong'
        phase=r['method'].rsplit('_',1)[0];target=dest/f'checkpoints/{phase}/{group}/seed{r["training_seed"]}/step{r["steps"]}.zip'
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():shutil.copy2(model(r['method'],r['training_seed']),target)
        assert sha(target)==r['sha256']
    atomic_json(dest/'experiment.json',c)
    return c
def configuration():
    p=ROOT/'runtime/experiment.json'
    return read(p) if p.exists() and read(p)['output_root']==str(ROOT/'runtime') else initialize()
