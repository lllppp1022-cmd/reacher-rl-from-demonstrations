"""Read-only historical audit and portable source/asset preparation."""
from pathlib import Path
import ast,csv,hashlib,json,shutil,os,sys
import numpy as np
P=Path(__file__).resolve().parents[1]; W=P.parent
H=W/'reacher_anchor_calibration'
PROJECTS=['reacher_bc_ppo','reacher_safe_finetune','reacher_bc_anchor','reacher_anchor_calibration']
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def js(p,o):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,ensure_ascii=False,indent=2),encoding='utf-8')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def rows(p):
    with Path(p).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def csvout(p,data):
    keys=list(dict.fromkeys(k for r in data for k in r));p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='',encoding='utf-8-sig') as f:
        x=csv.DictWriter(f,fieldnames=keys);x.writeheader();x.writerows(data)
def snapshot():
    result={};errors=[]
    for project in PROJECTS:
        def onerror(e):errors.append(str(e))
        for root,dirs,files in os.walk(W/project,onerror=onerror):
            dirs[:]=[d for d in dirs if d not in ['__pycache__','.git']]
            for name in files:
                p=Path(root)/name
                try:result[p.relative_to(W).as_posix()]=dict(bytes=p.stat().st_size,sha256=sha(p))
                except OSError as e:errors.append(str(e))
    return dict(files=result,unreadable=errors,excludes=['__pycache__','.git'])
def main():
    for d in ['docs','reports','tests','configs','models','results','assets/figures','assets/videos','src/verified','src/envs','src/expert','src/imitation','src/rl','src/evaluation','src/visualization']:(P/d).mkdir(parents=True,exist_ok=True)
    before=P/'tests/local_history_before.json'
    if not before.exists():js(before,snapshot())
    allrows=rows(H/'results/final_comparison.csv')
    wanted=['BC-DAgger','long_control_b0','long_fixed_medium','long_fixed_strong','long_decay_strong']
    per=[];models=[];checks=[]
    for r in allrows:
        if r['method'] not in wanted or r['checkpoint_role'] not in ['final','reference']:continue
        src=Path(r['path']);assert sha(src)==r['model_sha256']
        job=r['evaluation_reused_from'] or r['job'];folder=H/'results/final'/job
        ep=rows(folder/'episodes.csv');a=dict(np.load(folder/'trajectories.npz'))
        assert list(map(int,a['seeds']))==list(range(16000,16100))
        stable=np.all(a['distance_m'][-5:]<.01,axis=0)
        assert stable.tolist()==[bool(int(x['stable_success_1cm'])) for x in ep]
        assert np.max(np.abs(a['distance_m'][-1]-[float(x['final_distance_m']) for x in ep]))<1e-12
        cost=np.square(a['actions']).sum(axis=-1,dtype=np.float32).astype(np.float64).sum(axis=0)
        assert np.max(np.abs(cost-[float(x['action_sq_sum']) for x in ep]))<1e-9
        calc=dict(method=r['method'],training_seed=int(r['training_seed']),steps=int(r['step']),n_episodes=len(ep),
            stable_success_pct=100*float(np.mean([int(x['stable_success_1cm']) for x in ep])),
            mean_final_cm=100*float(np.mean([float(x['final_distance_m']) for x in ep])),
            mean_action_sq_sum=float(np.mean([float(x['action_sq_sum']) for x in ep])))
        for k in ['stable_success_pct','mean_final_cm','mean_action_sq_sum']:assert abs(calc[k]-float(r[k]))<1e-10
        per.append(calc);target=P/'models'/r['method']/f"seed{r['training_seed']}"/src.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target)
        dest=P/'results/archived_final'/r['job'];dest.mkdir(parents=True,exist_ok=True)
        shutil.copy2(folder/'episodes.csv',dest/'episodes.csv')
        # State arrays and distributions are local assets pending publication authorization.
        shutil.copy2(folder/'trajectories.npz',dest/'trajectories.npz')
        models.append(dict(method=r['method'],training_seed=int(r['training_seed']),steps=int(r['step']),path=target.relative_to(P).as_posix(),sha256=sha(target),bytes=target.stat().st_size,source_project=src.relative_to(W).parts[0],source_relative=src.relative_to(W).as_posix(),evaluation_seeds='16000-16099',distribution='pending_owner_authorization'))
        checks.append(dict(job=r['job'],csv_npz_match=True,archive_summary_match=True,model_hash_match=True))
    aggregate=[]
    for method in wanted:
        group=[r for r in per if r['method']==method];assert len(group)==3
        x=dict(method=method,training_seeds='42;123;2026',evaluation_seeds='16000-16099',episodes_per_seed=100,steps_per_seed=group[0]['steps'])
        for k in ['stable_success_pct','mean_final_cm','mean_action_sq_sum']:
            v=[r[k] for r in group];x[k]=float(np.mean(v));x[k+'_seed_sd']=float(np.std(v,ddof=1))
        aggregate.append(x)
    csvout(P/'reports/final_results.csv',aggregate);csvout(P/'reports/per_seed_results.csv',per);csvout(P/'reports/model_manifest.csv',models);js(P/'reports/archive_recalculation.json',checks)
    js(P/'configs/models.json',models)
    # Preserve the original sibling module relationships; AST-resolve every local import.
    seeds=[H/n for n in ['anchor_common.py','anchor_ppo.py','evaluate_anchor.py','record_training_rollouts.py','training_observation.py','training_rollout_analysis.py','watch_training_rollout.py','watch_anchor_calibration.py']]
    seeds += [W/'reacher_bc_ppo'/n for n in ['bc_common.py','train_bc.py','collect_expert_data.py','evaluate_bc.py','finetune_bc_ppo.py','record_reacher_video.py']]
    queue=seeds[:];seen=set();provenance=[]
    candidates=[H,W/'reacher_safe_finetune',W/'reacher_bc_ppo',W]
    while queue:
        src=queue.pop(0)
        if src in seen:continue
        seen.add(src);code=src.read_text(encoding='utf-8-sig');tree=ast.parse(code)
        for node in ast.walk(tree):
            names=[x.name for x in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) and node.module else []
            for name in names:
                for folder in [src.parent]+candidates:
                    local=folder/(name.split('.')[0]+'.py')
                    if local.exists():queue.append(local);break
        rel=src.relative_to(W);dst=P/'src/verified'/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        edits=[]
        if src.name=='evaluate_reacher_200k.py':
            code=code.replace('OUT.mkdir(exist_ok=True)','# Portfolio: no filesystem side effects during import.');edits.append('Remove import-time output-directory creation; physics unchanged')
        # Harmless OS font preference gets replaced with a portable font lookup.
        if 'C:/Windows/Fonts/arial.ttf' in code:code=code.replace('C:/Windows/Fonts/arial.ttf','DejaVuSans.ttf');edits.append('Portable optional font lookup')
        dst.write_text(code,encoding='utf-8')
        provenance.append(dict(source=rel.as_posix(),source_sha256=sha(src),destination=dst.relative_to(P).as_posix(),destination_sha256=sha(dst),edits='; '.join(edits) or 'none (newline normalization only)'))
    csvout(P/'reports/source_provenance.csv',provenance)
    c=read(H/'configs/experiment.json')
    for k in ['output_root','source_root','implementation_reference']:c[k]={'output_root':'runtime','source_root':'runtime/source','implementation_reference':'src/verified/reacher_safe_finetune'}[k]
    js(P/'configs/anchor_template.json',c)
    for name in ['diagnostic_observations.npz','beta_calibration.json']:
        value=H/'configs'/name;shutil.copy2(value,P/'configs'/name)
    # Keep only portable scalar calibration fields in public configuration.
    b=read(P/'configs/beta_calibration.json');js(P/'configs/beta_calibration.json',{'selected_betas':b['selected_betas'],'source':'historical calibration, training-only diagnostic observations'})
    js(P/'configs/protocol.json',dict(environment='Reacher-v5',episode_steps=250,dt=.02,reward_distance_weight=1,reward_control_weight=1,observation_dim=10,action_dim=2,action_limits=[-1,1],stable_success='all last 5 XY distances strictly < 0.01 m',train_seeds=[42,123,2026],validation_seeds=[15000,15099],final_seeds=[16000,16099],short_steps=20480,long_steps=100352,bc_actor_training_steps='supervised; not an online PPO budget',final_statistics='historical frozen evaluation recomputed; no new independent final set'))
    js(P/'tests/preparation.json',dict(models=len(models),source_modules=len(seen),historical_projects=PROJECTS))
    print('PREPARED',len(models),'models',len(seen),'source modules',aggregate)
if __name__=='__main__':main()
