"""Collect state/action transitions with the unchanged, episode-conditioned IK+PD expert."""
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bc_common import *

def collect(c,seeds,path):
    fields={k:[] for k in ['observation','expert_action','next_observation','reward','terminated','truncated',
        'episode_id','step','seed','distance','qpos','qvel','expert_target_q','ik_projected']}
    episodes=[];e=env();expert=Expert()
    try:
        for episode,seed in enumerate(seeds):
            obs,_=e.reset(seed=int(seed));expert.reset(e,obs);dist=[];cost=[]
            for step in range(250):
                before=obs.copy();qpos=e.unwrapped.data.qpos.copy();qvel=e.unwrapped.data.qvel.copy()
                action=expert.action(before)
                obs,r,term,trunc,_=e.step(action);d=float(np.linalg.norm(obs[8:10]));dist.append(d);cost.append(float(np.sum(action**2)))
                row=dict(observation=before.astype(np.float32),expert_action=action,next_observation=obs.astype(np.float32),
                    reward=r,terminated=term,truncated=trunc,episode_id=episode,step=step,seed=seed,distance=d,
                    qpos=qpos,qvel=qvel,expert_target_q=expert.target_q.copy(),ik_projected=expert.projected)
                for k,v in row.items():fields[k].append(v)
                assert not term and trunc==(step==249)
            episodes.append(dict(seed=int(seed),steps=250,final_cm=dist[-1]*100,
                final_success=int(dist[-1]<.01),stable_success=int(all(d<.01 for d in dist[-5:])),
                ik_projected=int(expert.projected),action_sq_sum=sum(cost)))
            if (episode+1)%100==0:print('COLLECTED',episode+1,'episodes',flush=True)
    finally:e.close()
    arrays={k:np.asarray(v) for k,v in fields.items()}
    for k,a in arrays.items():assert np.isfinite(a).all(),k
    assert arrays['observation'].shape==(len(seeds)*250,10)
    assert arrays['expert_action'].shape==(len(seeds)*250,2) and np.max(np.abs(arrays['expert_action']))<=1
    path=Path(path);path.parent.mkdir(exist_ok=True,parents=True);np.savez_compressed(path,**arrays)
    return arrays,episodes

def main(c):
    setup(c);require(c,'audit')
    if done(c,'collect').exists():return
    train,tre=collect(c,range(c['expert_train_seeds'][0],c['expert_train_seeds'][1]+1),out(c)/'datasets/expert_train.npz')
    val,ve=collect(c,range(c['expert_validation_seeds'][0],c['expert_validation_seeds'][1]+1),out(c)/'datasets/expert_validation.npz')
    write_csv(out(c)/'datasets/expert_episodes.csv',[dict(split='train',**r) for r in tre]+[dict(split='validation',**r) for r in ve])
    stats=[]
    for name,data,eps in [('train',train,tre),('validation',val,ve)]:
        a=data['expert_action'];norm=np.linalg.norm(a,axis=1)
        stats.append(dict(split=name,episodes=len(eps),samples=len(a),final_success_pct=np.mean([r['final_success'] for r in eps])*100,
            stable_success_pct=np.mean([r['stable_success'] for r in eps])*100,projected_episodes=sum(r['ik_projected'] for r in eps),
            action_abs_max=float(np.abs(a).max()),action_norm_mean=float(norm.mean()),
            near_zero_action_pct=float(np.mean(norm<1e-4)*100),early_50_samples=int(np.sum(data['step']<50)),
            late_100_samples=int(np.sum(data['step']>=150)),qpos1_min=float(data['qpos'][:,0].min()),
            qpos1_max=float(data['qpos'][:,0].max()),qvel_abs_max=float(np.abs(data['qvel'][:,:2]).max())))
    write_csv(out(c)/'datasets/expert_dataset_statistics.csv',stats)
    # Nearby states from different episodes: demonstrate ambiguity, without claiming exact duplicate labels.
    early=np.flatnonzero(train['step']<100);idx=early[::max(1,len(early)//2000)]
    obs=train['observation'][idx];scale=np.maximum(train['observation'].std(axis=0),.01)
    with torch.no_grad():
        distances=torch.cdist(torch.tensor(obs/scale),torch.tensor(obs/scale))
        dist,neigh=torch.topk(distances,k=12,largest=False,dim=1)
        dist=dist.numpy();neigh=neigh.numpy()
    pairs=[]
    for i in range(len(idx)):
        for rank,j in enumerate(neigh[i,1:],start=1):
            if train['seed'][idx[i]]!=train['seed'][idx[j]]:
                opposite=np.sign(train['expert_target_q'][idx[i],1])!=np.sign(train['expert_target_q'][idx[j],1])
                pairs.append(dict(normalized_observation_distance=float(dist[i,rank]),
                    action_difference=float(np.linalg.norm(train['expert_action'][idx[i]]-train['expert_action'][idx[j]])),
                    opposite_ik_branch=int(opposite),seed1=int(train['seed'][idx[i]]),seed2=int(train['seed'][idx[j]])));break
    if pairs:write_csv(out(c)/'datasets/nearby_label_pairs.csv',pairs)
    fig,axs=plt.subplots(1,2,figsize=(11,4.5))
    for j in (0,1):axs[0].hist(train['expert_action'][:,j],bins=100,histtype='step',density=True,label=f'Joint {j+1}')
    axs[0].set(xlabel='Clipped expert action',ylabel='Density',title='Expert training actions');axs[0].set_yscale('log');axs[0].legend()
    for seed in range(c['expert_train_seeds'][0],c['expert_train_seeds'][0]+5):
        ds=train['distance'][train['seed']==seed];axs[1].plot(np.arange(1,251)*.02,ds*100,label=f'seed {seed}')
    axs[1].axhline(1,color='gray',ls='--');axs[1].set(xlabel='Time (s)',ylabel='Distance (cm)',title='First 5 fixed training episodes');axs[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(out(c)/'plots/expert_data_distribution.png',dpi=170);plt.close(fig)
    ambiguity=sum(p['opposite_ik_branch'] for p in pairs)
    report=f'''# 专家数据质量与标签分析

训练{len(tre)}回合、{len(train['reward'])}样本；验证{len(ve)}回合、{len(val['reward'])}样本。每回合250步，全部数值有限，动作维度2、观测维度10，动作已限幅。详细成功率、投影及状态范围见expert_dataset_statistics.csv。distance保存动作后XY距离，observation/qpos/qvel为动作前状态，next_observation为动作后状态。

学生只接收10维observation，不接收expert_target_q、初始构型或episode_id。记录的目标关节角仅用于数据审计。专家按初始状态选择IK分支并在全回合保持该分支：因此专家实际带有回合记忆，10维瞬时观测未必能唯一确定其所选分支。不能保证这是严格无记忆的单值标签函数。

对固定采样的早期状态构建标准化10维最近邻，在不同回合间找到{len(pairs)}个邻居对，其中{ambiguity}对的目标q2符号不同。相近状态并非完全相同；动作差也会受到剩余状态差异影响。因此这项统计是多模态风险诊断，不是证明相同输入必有冲突标签。保留nearby_label_pairs.csv供复核，不把回归误差简单归因于网络容量。

训练后段近零动作占比与早期样本占比已统计。BC按0–49、50–149、150–249三个阶段等权采样，验证损失仍另外报告原始均匀MSE及分段误差，防止仅学习近零输出。输入标准化和动作尺度仅从训练集计算。验证、7000测试与8000最终数据不用于监督训练。
'''
    (out(c)/'datasets/expert_dataset_report.md').write_text(report,encoding='utf-8')
    finish(c,'collect',dict(train_samples=len(train['reward']),validation_samples=len(val['reward']),statistics=stats,
        train_sha256=sha(out(c)/'datasets/expert_train.npz'),validation_sha256=sha(out(c)/'datasets/expert_validation.npz')))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config');a=p.parse_args();main(config(a.config))
