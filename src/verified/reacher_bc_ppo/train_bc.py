"""Supervised actor-only training, phase-balanced sampling and validation early stopping."""
import argparse
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bc_common import *

def metrics(model,data,scale):
    obs=torch.tensor(data['observation'],dtype=torch.float32);target=torch.tensor(data['expert_action'],dtype=torch.float32)
    with torch.no_grad():pred=torch.cat([actor(model,b) for b in obs.split(4096)]).numpy()
    error=pred-target.numpy();norm=np.linalg.norm(target.numpy(),axis=1)
    result=dict(mse=float(np.mean(error**2)),normalized_mse=float(np.mean((error/scale)**2)),
        action_norm_pred=float(np.linalg.norm(pred,axis=1).mean()),action_norm_expert=float(norm.mean()))
    scores=[]
    for name,mask in [('early_50',data['step']<50),('middle', (data['step']>=50)&(data['step']<150)),
                      ('late_100',data['step']>=150),('large_action',norm>.02),('near_target',np.linalg.norm(data['observation'][:,8:10],axis=1)<.01)]:
        result[name+'_mse']=float(np.mean(error[mask]**2)) if mask.any() else 0.
        if name in ('early_50','middle','late_100'):scores.append(float(np.mean((error[mask]/scale)**2)))
    result['balanced_normalized_mse']=float(np.mean(scores));return result

def train_one(c,seed,train_path,val_path,folder,epochs=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'training_complete.json').exists():return read(folder/'training_complete.json')
    completed=sorted(folder.glob('run_*/training_complete.json'))
    if completed:return read(completed[-1])
    if (folder/'bc_best.zip').exists():
        # Preserve an interrupted run; never repeatedly replace an open Windows ZIP.
        folder=folder/f'run_{time.time_ns()}';folder.mkdir()
    seed_everything(seed);tr=dict(np.load(train_path));va=dict(np.load(val_path))
    mean=tr['observation'].mean(axis=0);std=np.maximum(tr['observation'].std(axis=0),.02)
    scale=np.maximum(tr['expert_action'].std(axis=0),.001)
    model=make_ppo(c,seed,mean,std)
    critic_before={k:v.detach().clone() for k,v in model.policy.state_dict().items() if 'value_net' in k}
    optimizer=torch.optim.Adam(actor_parameters(model),lr=c['bc_learning_rate'])
    obs=torch.tensor(tr['observation'],dtype=torch.float32);actions=torch.tensor(tr['expert_action'],dtype=torch.float32)
    scale_t=torch.tensor(scale);groups=[np.flatnonzero(tr['step']<50),np.flatnonzero((tr['step']>=50)&(tr['step']<150)),np.flatnonzero(tr['step']>=150)]
    rng=np.random.default_rng(seed);history=[];best=float('inf');bad=0;best_epoch=0
    for epoch in range(1,(epochs or c['bc_epochs'])+1):
        model.policy.set_training_mode(True);losses=[]
        sampled=np.concatenate([rng.choice(g,size=int(np.ceil(len(obs)/3)),replace=True) for g in groups]);rng.shuffle(sampled)
        for begin in range(0,len(sampled),c['bc_batch_size']):
            idx=sampled[begin:begin+c['bc_batch_size']]
            prediction=actor(model,obs[idx]);loss=torch.mean(((prediction-actions[idx])/scale_t)**2)
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(actor_parameters(model),5.);optimizer.step();losses.append(loss.item())
        model.policy.set_training_mode(False);vm=metrics(model,va,scale)
        row=dict(epoch=epoch,train_balanced_normalized_loss=float(np.mean(losses)),**vm);history.append(row)
        if vm['balanced_normalized_mse']<best-1e-7:
            best=vm['balanced_normalized_mse'];best_epoch=epoch;bad=0
            best_snapshot=folder/f'epoch_{epoch:03d}.zip'
            save_model(model,best_snapshot)
        else:bad+=1
        if epoch%5==0:print('BC',seed,'epoch',epoch,'validation',best,flush=True)
        if bad>=c['bc_patience']:break
    for k,v in critic_before.items():torch.testing.assert_close(model.policy.state_dict()[k],v,rtol=0,atol=0)
    best_model=load_bc(best_snapshot);save_model(best_model,folder/'bc_best.zip')
    torch.save(dict(actor_state={k:v.cpu() for k,v in best_model.policy.state_dict().items() if 'policy_net' in k or 'action_net' in k},
        observation_mean=mean.tolist(),observation_std=std.tolist(),action_loss_scale=scale.tolist(),
        sb3_file='bc_best.zip',seed=seed,epoch=best_epoch,input_dimension=10),folder/'bc_best.pt')
    vm=metrics(best_model,va,scale);tm=metrics(best_model,tr,scale)
    write_csv(folder/'bc_validation_metrics.csv',history)
    fig,ax=plt.subplots(figsize=(8,4.5))
    ax.plot([r['epoch'] for r in history],[r['train_balanced_normalized_loss'] for r in history],label='Training phase-balanced loss')
    ax.plot([r['epoch'] for r in history],[r['balanced_normalized_mse'] for r in history],label='Validation phase-balanced loss')
    ax.set(xlabel='Epoch',ylabel='Normalized action MSE',title=f'BC actor only, train seed={seed}');ax.set_yscale('log');ax.legend();ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(folder/'bc_training_curve.png',dpi=160);plt.close(fig)
    info=dict(seed=seed,best_epoch=best_epoch,epochs_run=len(history),validation=vm,training=tm,
        path=str(folder/'bc_best.zip'),pt_path=str(folder/'bc_best.pt'),sha256=sha(folder/'bc_best.zip'),
        train_dataset_sha256=sha(train_path),validation_dataset_sha256=sha(val_path),samples=len(obs),
        observation_mean=mean.tolist(),observation_std=std.tolist(),action_loss_scale=scale.tolist(),critic_unchanged=True)
    atomic_json(folder/'training_complete.json',info)
    (folder/'bc_training_report.md').write_text(f'# BC训练报告\n\n仅训练Actor，Critic参数逐张量核对未变。输入只用10维观测；固定标准化参数与监督损失尺度来自训练集。\n\n```json\n{json.dumps(info,ensure_ascii=False,indent=2)}\n```\n\n低MSE不能替代闭环测试，特别需要检查早期/大动作/近目标误差与输出范数。',encoding='utf-8')
    model.get_env().close();return info

def main(c):
    setup(c);require(c,'collect')
    if done(c,'bc').exists():return
    models=[train_one(c,seed,out(c)/'datasets/expert_train.npz',out(c)/'datasets/expert_validation.npz',
        out(c)/f'models/bc/seed{seed}') for seed in c['training_seeds']]
    atomic_json(out(c)/'models/bc/manifest.json',models);finish(c,'bc',dict(models=models))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config');a=p.parse_args();main(config(a.config))
