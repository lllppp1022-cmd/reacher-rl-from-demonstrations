"""Audit source, frozen jobs, runtime dependencies and publication provenance."""
from pathlib import Path
import sys,json,csv,importlib,importlib.metadata as md,platform
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.runtime import read,sha,atomic_json,registry,model
from prepare_portfolio import csvout
def main():
    H=ROOT.parent/'reacher_anchor_calibration'
    frozen=read(H/'configs/final_freeze.json')['jobs'];verified=[]
    for j in frozen:
        if j['path']:assert sha(j['path'])==j['sha256']
        verified.append(dict(job=j['job'],method=j['method'],training_seed=j['training_seed'],steps=j['step'],checkpoint_role=j['checkpoint_role'],sha256=j['sha256'],verified=True))
    csvout(ROOT/'reports/all_frozen_models.csv',verified)
    def clean(x):
        if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
        if isinstance(x,list):return [clean(v) for v in x]
        if isinstance(x,str):return x.replace(str(ROOT.parent)+'\\','').replace(str(ROOT.parent).replace('\\','/')+'/','')
        return x
    for pr in ['reacher_bc_ppo','reacher_safe_finetune','reacher_bc_anchor','reacher_anchor_calibration']:
        paths=[ROOT.parent/pr/'configs/experiment.json',ROOT.parent/pr/'tests/acceptance.json']
        paths+=list((ROOT.parent/pr/'reports').glob('*.md'))
        info=[]
        for p in paths:
            if not p.exists():continue
            info.append(dict(file=p.relative_to(ROOT.parent).as_posix(),sha256=sha(p),bytes=p.stat().st_size))
            if p.suffix=='.json':atomic_json(ROOT/'reports/historical_audit'/pr/p.name,clean(read(p)))
        atomic_json(ROOT/'reports/historical_audit'/pr/'inventory.json',info)
    for n in ['experiment_manifest.csv','experiment_status.csv']:
        with (H/'results'/n).open(encoding='utf-8-sig') as f:data=list(csv.DictReader(f))
        for r in data:
            if r.get('path'):r['path']=Path(r['path']).relative_to(ROOT.parent).as_posix()
        csvout(ROOT/'reports'/n,data)
    imports=['anchor_ppo','evaluate_anchor','training_observation','training_rollout_analysis','record_training_rollouts','watch_training_rollout','watch_anchor_calibration','collect_expert_data','train_bc','evaluate_bc','finetune_bc_ppo','record_reacher_video']
    loaded=[]
    for name in imports:importlib.import_module(name);loaded.append(name)
    import numpy as np
    from evaluate_anchor import load_policy
    from bc_common import env
    e=env();obs,_=e.reset(seed=7059);models=[]
    for r in registry():
        p=model(r['method'],r['training_seed']);m=load_policy(p);a=m.predict(obs,deterministic=True)[0]
        assert np.isfinite(a).all() and a.shape==(2,)
        assert int(m.num_timesteps)==r['steps'] if r['steps'] else True
        models.append(dict(method=r['method'],seed=r['training_seed'],sha256=sha(p),loaded=True))
    e.close();atomic_json(ROOT/'tests/import_and_model_validation.json',dict(imports=loaded,models=models))
    direct=['numpy','gymnasium','mujoco','stable-baselines3','torch','matplotlib','Pillow','imageio','glfw','tensorboard','packaging']
    pending=direct[:];seen=set();licenses=[]
    while pending:
        name=pending.pop(0);name=canonicalize_name(name)
        if name in seen:continue
        seen.add(name);dist=md.distribution(name);meta=dist.metadata
        license_text=meta.get('License-Expression') or meta.get('License') or 'not declared in installed metadata'
        license_text=license_text[:500]
        licensefiles=[str(x) for x in (dist.files or []) if any(t in Path(str(x)).name.lower() for t in ['license','copying','notice']) and '.dist-info' in str(x)]
        licenses.append(dict(package=meta['Name'],version=dist.version,license_metadata=license_text,license_files=';'.join(licensefiles),source='installed distribution metadata; not a legal authorization for project assets'))
        for namefile in licensefiles:
            source=dist.locate_file(namefile)
            if source.is_file():
                # Keep internal paths: many packages ship several distinct LICENSE files.
                dest=ROOT/'third_party'/name/Path(*Path(namefile).parts[1:]);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(source.read_bytes())
        for raw in dist.requires or []:
            req=Requirement(raw)
            if req.marker is None or req.marker.evaluate({'extra':''}):pending.append(req.name)
    csvout(ROOT/'reports/dependency_licenses.csv',licenses)
    # Reproducible installed-platform lock, with the direct dependency list distinguished.
    (ROOT/'requirements.txt').write_text('\n'.join(f'{name}=={md.version(name)}' for name in direct)+'\n',encoding='utf-8')
    (ROOT/'requirements-lock.txt').write_text('\n'.join(f'{r["package"]}=={r["version"]}' for r in sorted(licenses,key=lambda x:x['package'].lower()))+'\n',encoding='utf-8')
    atomic_json(ROOT/'reports/runtime_environment.json',dict(python=platform.python_version(),system=platform.system(),machine=platform.machine(),direct_packages=direct,dependency_closure_packages=len(licenses),new_clean_environment_install_test='not yet performed; installed-runtime tests only'))
    assets=[dict(path=r['path'],sha256=r['sha256'],kind='model',distribution='pending_owner_authorization') for r in registry()]
    diagnostic=ROOT/'configs/diagnostic_observations.npz'
    assets.append(dict(path=diagnostic.relative_to(ROOT).as_posix(),sha256=sha(diagnostic),kind='training-only reference observations',distribution='pending_owner_authorization'))
    csvout(ROOT/'reports/asset_provenance.csv',assets)
    facades={'envs':'from bc_common import env\n','expert':'from bc_common import Expert\n','imitation':'from train_bc import train_one\nfrom evaluate_bc import dagger\nfrom bc_common import load_bc\n','rl':'from anchor_ppo import AnchorPPO, anchoring_loss, build\nfrom anchor_common import effective_beta\n','evaluation':'from evaluate_anchor import evaluate, summarize\n','visualization':'from training_observation import ActualTrainingEnv, SamplerObserver, ObservedAnchorPPO\n'}
    for folder,body in facades.items():
        (ROOT/'src'/folder/'__init__.py').write_text('"""Public facade for the preserved verified implementation."""\nfrom src import runtime\n'+body,encoding='utf-8')
    print('AUDIT',len(verified),'frozen declarations;',len(licenses),'dependency distributions')
if __name__=='__main__':main()
