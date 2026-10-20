"""Explicit actor transfer, exact initial-action verification, then 100k online PPO steps."""
import argparse
import subprocess
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from bc_common import *

def publish(c,path,seed,steps,phase):
    atomic_json(out(c)/'live/current.json',dict(path=str(path),sha256=sha(path),seed=seed,steps=steps,phase=phase))

class Snapshot(BaseCallback):
    def __init__(self,c,folder,seed):super().__init__();self.c=c;self.folder=folder;self.seed=seed;self.last=0
    def _on_step(self):return True
    def save(self,force=False):
        n=self.model.num_timesteps
        if n and (force or n//20000>self.last//20000):
            path=self.folder/f'checkpoint_{n}.zip'
            if not path.exists():save_model(self.model,path)
            publish(self.c,path,self.seed,n,'BC+PPO snapshot');self.last=n
            print('PPO SNAPSHOT',self.seed,n,flush=True)
    def _on_rollout_start(self):self.save()
    def _on_training_end(self):self.save(True)

def one(c,source):
    seed=source['seed'];folder=out(c)/f'models/bc_ppo/seed{seed}';folder.mkdir(exist_ok=True,parents=True)
    if (folder/'complete.json').exists():
        result=read(folder/'complete.json');assert sha(result['path'])==result['sha256'];return result
    e=Monitor(env(),filename=str(folder/f'monitor_{time.time_ns()}.csv'))
    seed_everything(seed);bc=load_bc(source['path'])
    model=make_ppo(c,seed,source['observation_mean'],source['observation_std'],e)
    model.policy.mlp_extractor.policy_net.load_state_dict(bc.policy.mlp_extractor.policy_net.state_dict(),strict=True)
    model.policy.action_net.load_state_dict(bc.policy.action_net.state_dict(),strict=True)
    with torch.no_grad():model.policy.log_std.fill_(np.log(c['initial_exploration_std']))
    samples=np.load(out(c)/'datasets/expert_validation.npz')['observation'][::25]
    expected=bc.predict(samples,deterministic=True)[0];actual=model.predict(samples,deterministic=True)[0]
    diff=float(np.abs(expected-actual).max());assert diff<1e-7
    if not (folder/'initial_0.zip').exists():save_model(model,folder/'initial_0.zip')
    restored=PPO.load(folder/'initial_0.zip',device='cpu')
    reload_diff=float(np.abs(restored.predict(samples,deterministic=True)[0]-expected).max());assert reload_diff<1e-7
    atomic_json(folder/'transfer_verification.json',dict(source=source['path'],source_sha256=source['sha256'],
        variant=source['variant'],actor_layers_transferred=['mlp_extractor.policy_net','action_net'],
        fixed_observation_normalizer_transferred=True,critic='fresh random initialization; no BC supervision',
        initial_exploration_std=c['initial_exploration_std'],max_bc_to_ppo_action_difference=diff,
        max_saved_reloaded_action_difference=reload_diff,checked_observations=len(samples)))
    snapshots=sorted(folder.glob('checkpoint_*.zip'),key=lambda p:int(p.stem.split('_')[-1]))
    resumed=bool(snapshots)
    if resumed:model=PPO.load(snapshots[-1],env=e,device='cpu',tensorboard_log=str(out(c)/'logs/tensorboard'))
    else:publish(c,folder/'initial_0.zip',seed,0,'BC initialized, zero PPO steps')
    start=time.perf_counter()
    try:
        model.learn(total_timesteps=c['finetune_steps']-model.num_timesteps,reset_num_timesteps=not resumed,
            tb_log_name=f'BC_PPO_seed{seed}',callback=Snapshot(c,folder,seed))
        assert model.num_timesteps==c['finetune_steps']
        assert all(torch.isfinite(v).all() for v in model.policy.state_dict().values())
        path=folder/'final.zip';save_model(model,path)
        result=dict(seed=seed,path=str(path),sha256=sha(path),steps=model.num_timesteps,resumed=resumed,
            initial_path=str(folder/'initial_0.zip'),source_bc=source,elapsed_seconds=time.perf_counter()-start,
            snapshot_steps=sorted(int(p.stem.split('_')[-1]) for p in folder.glob('checkpoint_*.zip')))
        atomic_json(folder/'complete.json',result);return result
    finally:e.close()

def main(c):
    setup(c);require(c,'evaluate_bc')
    if done(c,'finetune').exists():return
    sources=read(out(c)/'models/bc/active_manifest.json');viewer=None
    (out(c)/'live/stop_viewer.json').unlink(missing_ok=True)
    if c['viewer']:
        log=(out(c)/'logs/finetune_viewer.log').open('a',encoding='utf-8')
        viewer=subprocess.Popen([sys.executable,'-u',str(HERE/'watch_bc_ppo_training.py'),
            '--config',c['_config_path'],'--seed','7100','--fps','25'],stdout=log,stderr=subprocess.STDOUT,cwd=HERE)
    models=[]
    try:
        for source in sources:models.append(one(c,source))
        atomic_json(out(c)/'models/bc_ppo/manifest.json',models)
        finish(c,'finetune',dict(models=models))
    finally:
        atomic_json(out(c)/'live/stop_viewer.json',dict(reason='finetuning complete'))
        if viewer:
            try:viewer.wait(timeout=10)
            except subprocess.TimeoutExpired:viewer.terminate()
            log.close()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config');a=p.parse_args();main(config(a.config))
