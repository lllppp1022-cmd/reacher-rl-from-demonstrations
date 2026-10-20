"""Independent C-compatible PPO with fixed-BC mean anchoring and gradient-strength probes.

Rollout collection, time-limit bootstrapping and GAE remain the installed SB3 implementation.
The PPO objective/gradient clipping match SB3; additions are explicitly logged.
"""
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance
from anchor_common import *
from bc_common import InputNormalizer

def anchoring_loss(model,observations):
    """Raw Gaussian mean MSE in control units; teacher has no autograd graph."""
    with torch.no_grad():target=actor(model.bc_reference,observations)
    return F.mse_loss(actor(model,observations),target)

def gradient_vector(loss,parameters):
    grads=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten() for p,g in zip(parameters,grads)])

class AnchorPPO(PPO):
    def __init__(self,*args,group_settings=None,**kwargs):
        self.group_settings=group_settings or dict(actor_lr=3e-4,critic_lr=3e-4,std=.005,warmup_steps=0,target_kl=None,anchor_beta=0)
        self.on_update=None;self.bc_reference=None
        super().__init__(*args,**kwargs)

    def _setup_model(self):
        super()._setup_model()
        groups={'actor':[],'critic':[]};names={'actor':[],'critic':[]}
        for name,param in self.policy.named_parameters():
            key='actor' if name=='log_std' or 'policy_net' in name or 'action_net' in name else 'critic'
            assert key=='actor' or 'value_net' in name,f'Unclassified trainable parameter {name}'
            groups[key].append(param);names[key].append(name)
        ids=[id(p) for group in groups.values() for p in group]
        assert len(ids)==len(set(ids))==len(list(self.policy.parameters()))
        self.parameter_group_names=names
        self.policy.optimizer=torch.optim.Adam([
            dict(params=groups['actor'],lr=self.group_settings['actor_lr'],name='actor'),
            dict(params=groups['critic'],lr=self.group_settings['critic_lr'],name='critic')],eps=1e-5)

    def _excluded_save_params(self):
        return super()._excluded_save_params()+['on_update','bc_reference','reference_tensor','bc_fixed_means',
            'initial_actorvec','initial_meanvec','previous_meanvec','teacher_snapshot']

    def attach_reference(self,bc,observations,reference_std):
        self.bc_reference=bc
        bc.policy.set_training_mode(False)
        for p in bc.policy.parameters():p.requires_grad_(False)
        self.teacher_snapshot={k:v.detach().clone() for k,v in bc.policy.state_dict().items()}
        self.reference_tensor=torch.as_tensor(observations,dtype=torch.float32,device=self.device)
        with torch.no_grad():self.bc_fixed_means=actor(bc,self.reference_tensor).detach()
        self.reference_std=reference_std
        self.initial_actorvec=actor_vector(bc);self.initial_meanvec=mean_actor_vector(bc)
        self.previous_meanvec=mean_actor_vector(self)

    def beta_value(self):
        return effective_beta(self.group_settings,self.num_timesteps,self.n_steps)

    def diagnostics(self):
        with torch.no_grad():
            mu=actor(self,self.reference_tensor);old=self.bc_fixed_means
            delta=mu-old;std=self.policy.log_std.exp();refstd=self.reference_std
            # Analytic forward KL N(mu_BC,refstd) || N(mu_PPO,std), before actuator clipping.
            kl=(torch.log(std/refstd)+(refstd**2+delta.square())/(2*std.square())-.5).sum(-1)
            clipped=mu.clamp(-1,1);oldclipped=old.clamp(-1,1)
            values=self.policy.predict_values(self.reference_tensor).flatten()
            row=dict(bc_action_mse=float(delta.square().mean()),bc_action_max_error=float(delta.abs().max()),
                bc_executed_action_mse=float((clipped-oldclipped).square().mean()),
                bc_gaussian_kl_mean=float(kl.mean()),bc_gaussian_kl_max=float(kl.max()),bc_reference_std=refstd,
                actor_parameter_drift=float(torch.linalg.vector_norm(mean_actor_vector(self)-self.initial_meanvec)),
                actor_parameter_update=float(torch.linalg.vector_norm(mean_actor_vector(self)-self.previous_meanvec)),
                fixed_critic_mean=float(values.mean()),fixed_critic_std=float(values.std()),
                executed_action_norm_mean=float(clipped.norm(dim=-1).mean()),
                action_clipped_fraction=float((mu.abs()>1).any(-1).float().mean()))
            for j in range(2):
                row.update({f'actor_mean_{j+1}':float(mu[:,j].mean()),f'actor_state_std_{j+1}':float(mu[:,j].std()),
                    f'gaussian_std_{j+1}':float(std[j]),f'bc_mean_bias_{j+1}':float(delta[:,j].mean())})
        return row

    def train(self):
        self.policy.set_training_mode(True)
        settings=self.group_settings
        beta=self.beta_value()
        warmup=self.num_timesteps<=settings['warmup_steps']
        for group in self.policy.optimizer.param_groups:
            group['lr']=settings[group['name']+'_lr']
        for name,param in self.policy.named_parameters():
            param.requires_grad_(not warmup or name in self.parameter_group_names['critic'])
        before=actor_vector(self).clone()
        previous_mu=self.previous_meanvec.clone()
        losses=[];minibatches=[];probes=[];update_count=0;stop=False;start=time.perf_counter()
        clip_range=self.clip_range(self._current_progress_remaining)
        pre_values=self.rollout_buffer.values.copy();returns=self.rollout_buffer.returns.copy()
        advantages_all=self.rollout_buffer.advantages.copy()
        for epoch in range(self.n_epochs):
            for batch_i,data in enumerate(self.rollout_buffer.get(self.batch_size)):
                values,log_prob,entropy=self.policy.evaluate_actions(data.observations,data.actions)
                values=values.flatten();adv=data.advantages
                if self.normalize_advantage and len(adv)>1:adv=(adv-adv.mean())/(adv.std()+1e-8)
                ratio=torch.exp(log_prob-data.old_log_prob)
                policy_loss=-torch.min(adv*ratio,adv*torch.clamp(ratio,1-clip_range,1+clip_range)).mean()
                value_loss=F.mse_loss(data.returns,values)
                entropy_loss=-torch.mean(entropy) if entropy is not None else torch.mean(log_prob)
                clip_fraction=float(((ratio-1).abs()>clip_range).float().mean())
                with torch.no_grad():
                    log_ratio=log_prob-data.old_log_prob
                    approx_kl=float(((torch.exp(log_ratio)-1)-log_ratio).mean())
                anchor_loss=torch.zeros((),device=self.device)
                if beta and not warmup:anchor_loss=anchoring_loss(self,data.observations)
                ppo_loss=policy_loss+self.ent_coef*entropy_loss+self.vf_coef*value_loss
                loss=self.vf_coef*value_loss if warmup else (ppo_loss+beta*anchor_loss if beta else ppo_loss)
                trigger=not warmup and self.target_kl is not None and approx_kl>1.5*self.target_kl
                entry=dict(step=self.num_timesteps,epoch=epoch+1,minibatch=batch_i+1,approx_kl=approx_kl,
                    kl_stop=trigger,updated=not trigger,warmup=warmup)
                minibatches.append(entry)
                losses.append(dict(policy_gradient_loss=float(policy_loss.detach()),value_loss=float(value_loss.detach()),
                    entropy_loss=float(entropy_loss.detach()),anchor_loss=float(anchor_loss.detach()),
                    weighted_anchor_loss=beta*float(anchor_loss.detach()),clip_fraction=clip_fraction))
                if trigger:stop=True;break
                if epoch in (0,self.n_epochs-1) and batch_i in (0,self.n_steps//self.batch_size-1):
                    parameters=[p for n,p in self.policy.named_parameters() if ('policy_net' in n or 'action_net' in n)]
                    pg=gradient_vector(policy_loss+self.ent_coef*entropy_loss,parameters)
                    ag=gradient_vector(beta*anchor_loss,parameters) if beta else torch.zeros_like(pg)
                    pg_norm=float(pg.norm());ag_norm=float(ag.norm())
                    probe=dict(ppo_actor_grad_norm=pg_norm,weighted_anchor_grad_norm=ag_norm,
                        anchor_to_ppo_grad_ratio=ag_norm/pg_norm if pg_norm else 0.)
                    entry.update(probe);probes.append(probe)
                if not torch.isfinite(loss):raise FloatingPointError('Non-finite loss before optimizer step')
                self.policy.optimizer.zero_grad();loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.policy.parameters()):raise FloatingPointError('Non-finite gradient before optimizer step')
                actor_grad=torch.sqrt(sum(p.grad.detach().square().sum() for n,p in self.policy.named_parameters()
                    if p.grad is not None and n in self.parameter_group_names['actor'])) if not warmup else torch.tensor(0.)
                critic_grad=torch.sqrt(sum(p.grad.detach().square().sum() for n,p in self.policy.named_parameters()
                    if p.grad is not None and n in self.parameter_group_names['critic']))
                preclip=torch.nn.utils.clip_grad_norm_(self.policy.parameters(),self.max_grad_norm,error_if_nonfinite=True)
                entry.update(global_grad_norm_before_clip=float(preclip),clip_coefficient=min(1.,self.max_grad_norm/(float(preclip)+1e-6)),log_std_grad_norm=float(self.policy.log_std.grad.norm()) if self.policy.log_std.grad is not None else 0.)
                self.policy.optimizer.step();update_count+=1
                entry.update(actor_grad_norm=float(actor_grad),critic_grad_norm=float(critic_grad))
            self._n_updates+=1
            if stop:break
        assert all(torch.isfinite(v).all() for v in self.policy.state_dict().values()),'Non-finite policy parameters'
        assert all(p.grad is None and not p.requires_grad for p in self.bc_reference.policy.parameters())
        assert all(torch.equal(value,self.bc_reference.policy.state_dict()[key]) for key,value in self.teacher_snapshot.items()),'Fixed BC teacher changed'
        if warmup:assert torch.equal(before,actor_vector(self)),'Warmup changed Actor or log_std'
        # Restore flags for subsequent rollout collection and checkpoint loading.
        for p in self.policy.parameters():p.requires_grad_(True)
        self.previous_meanvec=previous_mu
        report=self.diagnostics()
        report.update(actor_sha256=mean_actor_hash(self),step=self.num_timesteps,rollout_index=self.num_timesteps//self.n_steps,
            phase='critic_warmup' if warmup else 'ppo',warmup_actor_exactly_unchanged=bool(torch.equal(before,actor_vector(self))) if warmup else None,
            update_epochs=epoch+1,optimizer_steps=update_count,kl_early_stop=stop,
            update_kl_mean=float(np.mean([r['approx_kl'] for r in minibatches])),update_kl_max=max(r['approx_kl'] for r in minibatches),
            explained_variance=finite_number(explained_variance(pre_values.flatten(),returns.flatten())),
            rollout_value_mean=float(pre_values.mean()),rollout_return_mean=float(returns.mean()),
            advantage_mean=float(advantages_all.mean()),advantage_std=float(advantages_all.std()),
            advantage_min=float(advantages_all.min()),advantage_max=float(advantages_all.max()),
            actor_lr=settings['actor_lr'],critic_lr=settings['critic_lr'],anchor_beta=settings['anchor_beta'],
            beta_effective=beta,beta_schedule=settings['beta_schedule'],teacher_exactly_unchanged=True,
            actor_grad_norm_mean=float(np.mean([r.get('actor_grad_norm',0) for r in minibatches])),
            global_grad_norm_before_clip_mean=float(np.mean([r.get('global_grad_norm_before_clip',0) for r in minibatches])),
            clip_coefficient_mean=float(np.mean([r.get('clip_coefficient',1) for r in minibatches])),
            log_std_grad_norm_mean=float(np.mean([r.get('log_std_grad_norm',0) for r in minibatches])),
            critic_grad_norm_mean=float(np.mean([r.get('critic_grad_norm',0) for r in minibatches])),update_seconds=time.perf_counter()-start)
        for key in losses[0]:report[key]=float(np.mean([r[key] for r in losses]))
        for key in ['ppo_actor_grad_norm','weighted_anchor_grad_norm','anchor_to_ppo_grad_ratio']:
            report[key+'_mean']=float(np.mean([r[key] for r in probes])) if probes else None
        with torch.no_grad():
            obs=torch.as_tensor(self.rollout_buffer.observations.reshape(-1,10),device=self.device)
            after=self.policy.predict_values(obs).cpu().numpy().flatten()
        report['value_target_mse_before']=float(np.mean((pre_values.flatten()-returns.flatten())**2))
        report['value_target_mse_after']=float(np.mean((after-returns.flatten())**2))
        for key,val in report.items():
            if isinstance(val,(int,float)) and val is not None:self.logger.record('diagnostics/'+key,val)
        self.logger.dump(self.num_timesteps)
        if self.on_update:self.on_update(self,report,minibatches)
        self.previous_meanvec=mean_actor_vector(self)

def build(c,group,seed,environment=None,settings=None):
    settings=settings or phase_settings(c,group,'short');src=source(c,seed);bc=load_bc(src['path'])
    seed_everything(seed)
    model=AnchorPPO('MlpPolicy',environment or env(),seed=seed,device='cpu',verbose=0,
        learning_rate=settings['critic_lr'],group_settings=settings,target_kl=settings['target_kl'],
        policy_kwargs=dict(net_arch=dict(pi=[64,64],vf=[64,64]),activation_fn=torch.nn.Tanh,
            features_extractor_class=InputNormalizer,features_extractor_kwargs=dict(mean=src['observation_mean'],std=src['observation_std'])),
        **c['ppo'])
    model.policy.mlp_extractor.policy_net.load_state_dict(bc.policy.mlp_extractor.policy_net.state_dict())
    model.policy.action_net.load_state_dict(bc.policy.action_net.state_dict())
    with torch.no_grad():model.policy.log_std.fill_(np.log(settings['std']))
    samples=reference(c);expected=bc.predict(samples,deterministic=True)[0]
    assert np.max(np.abs(model.predict(samples,deterministic=True)[0]-expected))<1e-7
    model.attach_reference(bc,samples,c['bc_reference_std'])
    return model
