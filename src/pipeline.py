"""Run the original expert/BC/DAgger implementations and original PPO engines."""
from src.runtime import ROOT,read,atomic_json,sha,configuration
from pathlib import Path
import numpy as np,time
def run(args):
    from collect_expert_data import collect
    from train_bc import train_one
    from evaluate_bc import dagger
    from bc_common import load_bc,make_ppo,save_model,seed_everything,env
    from anchor_ppo import build
    from anchor_common import phase_settings
    from stable_baselines3.common.logger import configure
    dest=args.output.resolve();assert not dest.exists() or not any(dest.iterdir()),'Use a new output directory'
    dest.mkdir(parents=True,exist_ok=True);small=args.profile=='smoke';n=2 if small else 1000;nv=1 if small else 100
    c=dict(output_root=str(dest),training_seeds=[42],bc_epochs=2 if small else 60,bc_patience=12,bc_batch_size=1024,bc_learning_rate=.001,expert_train_seeds=[5000,5000+n-1],dagger_episodes=1 if small else 100)
    (dest/'datasets').mkdir();started=time.perf_counter()
    tr,_=collect(c,range(5000,5000+n),dest/'datasets/expert_train.npz');collect(c,range(6000,6000+nv),dest/'datasets/expert_validation.npz')
    bc=train_one(c,42,dest/'datasets/expert_train.npz',dest/'datasets/expert_validation.npz',dest/'models/bc/seed42')
    dagger(c,[bc]);bc2=train_one(c,42,dest/'datasets/expert_train_dagger.npz',dest/'datasets/expert_validation.npz',dest/'models/bc_dagger/seed42')
    atomic_json(dest/'models/bc_dagger/manifest.json',[bc2]);steps=1024 if small else 100352
    if args.ppo_method=='anchor':
        # A from-scratch reproduction uses its own freshly trained teacher and observations.
        # It must not initialize the historical, intentionally undistributed model assets.
        c2=read(ROOT/'configs/anchor_template.json');c2.update(output_root=str(dest),source_root=str(dest));(dest/'configs').mkdir(exist_ok=True)
        import shutil
        shutil.copy2(ROOT/'configs/beta_calibration.json',dest/'configs/beta_calibration.json')
        np.savez_compressed(dest/'configs/diagnostic_observations.npz',observation=tr['observation'][::max(1,len(tr['observation'])//1000)])
        settings=phase_settings(c2,args.group,'long_decay' if args.schedule=='linear' else 'long_fixed');settings['total_steps']=steps
        m=build(c2,args.group,42,settings=settings)
    elif args.ppo_method=='init':
        from finetune_bc_ppo import one
        from stable_baselines3 import PPO
        c.update(finetune_steps=steps,initial_exploration_std=.005)
        info=one(c,dict(bc2,variant='BC-DAgger'))
        m=PPO.load(info['path'],device='cpu')
    else:
        seed_everything(42);m=make_ppo(c,42,bc2['observation_mean'],bc2['observation_std'])
    if args.ppo_method!='init':
        from finetune_bc_ppo import Snapshot
        m.set_logger(configure(str(dest/'ppo_logs'),['csv','tensorboard']))
        checkpoints=dest/'checkpoints';checkpoints.mkdir();save_model(m,checkpoints/'initial_0.zip')
        m.learn(steps,callback=Snapshot(c,checkpoints,42));m.get_env().close()
    save_model(m,dest/'ppo_final.zip')
    atomic_json(dest/'complete.json',dict(purpose='plumbing smoke, not a research result' if small else 'user-requested reproduction',profile=args.profile,ppo_method=args.ppo_method,ppo_environment_steps=m.num_timesteps,expert_train_samples=n*250,expert_validation_samples=nv*250,dagger_expert_labels=c['dagger_episodes']*250,dagger_online_steps=c['dagger_episodes']*250,supervised_epochs_per_stage=c['bc_epochs'],ppo_updates=m._n_updates,model_sha256=sha(dest/'ppo_final.zip'),elapsed_seconds=time.perf_counter()-started))
    print('PIPELINE COMPLETE',dest)
