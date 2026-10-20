"""Opt-in observation of the actual SB3 sampler; inherited collection and optimization stay intact."""
import csv, json, time, hashlib, os
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
import mujoco
from stable_baselines3.common.callbacks import BaseCallback
from anchor_common import atomic_json,sha,save_model,mean_actor_hash
from anchor_ppo import AnchorPPO

STATE_SPEC=mujoco.mjtState.mjSTATE_INTEGRATION

def physics_snapshot(e):
    u=e.unwrapped;state=np.empty(mujoco.mj_stateSize(u.model,STATE_SPEC))
    mujoco.mj_getState(u.model,u.data,state,STATE_SPEC)
    return dict(integration_state=state,qpos=u.data.qpos.copy(),qvel=u.data.qvel.copy(),ctrl=u.data.ctrl.copy(),
        tip=u.get_body_com('fingertip').copy(),target=u.get_body_com('target').copy(),sim_time=float(u.data.time))

def rows_csv(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r));path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v.tolist() if isinstance(v,np.ndarray) else v,ensure_ascii=False) if isinstance(v,(np.ndarray,list,dict,tuple)) else v for k,v in r.items()})

class ActualTrainingEnv(gym.Wrapper):
    """Exactly one wrapped step per PPO call; capture terminal state before DummyVecEnv autoreset."""
    def __init__(self,environment,observer):
        super().__init__(environment);self.observer=observer;self.global_step=0;self.episode=0;self.episode_step=0;self.total_reward=0.;self.previous_obs=None;self.latest=None
        if observer.human:
            root=self.unwrapped;original=root.render
            def guarded_render():
                try:return original()
                except Exception as exc:
                    observer.errors.append(dict(where='human_render_guard',error=repr(exc),step=self.global_step))
                    root.render_mode=None;observer.human=False;return None
            root.render=guarded_render
    def reset(self,**kwargs):
        obs,info=self.env.reset(**kwargs)  # Pass SB3 seed/reset options through unchanged.
        self.episode+=1;self.episode_step=0;self.total_reward=0.;self.previous_obs=obs.copy()
        self.observer.reset_calls.append(dict(episode=self.episode,seed=kwargs.get('seed'),global_step=self.global_step))
        return obs,info
    def step(self,action):
        before=self.observer.safe(lambda:physics_snapshot(self.env),'snapshot_before')
        old=self.previous_obs.copy()
        obs,reward,terminated,truncated,info=self.env.step(action)  # THE ONLY physics step.
        self.global_step+=1;self.episode_step+=1;self.total_reward+=float(reward)
        after=self.observer.safe(lambda:physics_snapshot(self.env),'snapshot_after')
        self.latest=dict(global_step=self.global_step,episode_id=self.episode,episode_step=self.episode_step,
            observation_env=old,next_observation=obs.copy(),executed_action=np.array(action).copy(),reward=float(reward),
            cumulative_reward=self.total_reward,terminated=bool(terminated),truncated=bool(truncated),before=before,after=after)
        self.previous_obs=obs.copy()
        self.observer.safe(lambda:self.observer.environment_step(self,self.latest),'observe_environment_step')
        return obs,reward,terminated,truncated,info

class ObservedAnchorPPO(AnchorPPO):
    """Only phase notifications surround the unchanged historical train() implementation."""
    def _excluded_save_params(self):return super()._excluded_save_params()+['training_observer']
    def train(self):
        observer=getattr(self,'training_observer',None)
        if observer:observer.safe(lambda:observer.update_start(self),'update_start_notification')
        super().train()
        if observer:observer.safe(lambda:observer.update_end(self),'update_end_notification')

class SamplerObserver(BaseCallback):
    def __init__(self,destination,record_rgb=False,every=4,maximum=4,human=False,fps=50,disable_after=None,inject_rgb_failure=False):
        super().__init__(verbose=0);self.dest=Path(destination);self.dest.mkdir(parents=True,exist_ok=True)
        for name in ['checkpoints','rgb','videos','plots','evaluation','live']:(self.dest/name).mkdir(exist_ok=True)
        self.record_rgb=record_rgb;self.every=every;self.maximum=maximum;self.human=human;self.fps=fps;self.disable_after=disable_after
        self.inject_rgb_failure=inject_rgb_failure;self.rgb_disabled=False;self.rows=[];self.executions=[];self.events=[];self.errors=[];self.reset_calls=[]
        self.frames={};self.frame_steps={};self.rollout_id=0;self.version=0;self.policy_sha=None;self.last_report=None;self.active_wrapper=None
        self.buffer_checks=[];self.render_seconds=0.;self.sample_start=0;self.human_visible=False;self.closed_render_at=None;self.phase='Starting';self.render_released=False
    def safe(self,fn,where):
        try:return fn()
        except Exception as exc:
            self.errors.append(dict(where=where,error=repr(exc),step=self.active_wrapper.global_step if self.active_wrapper else 0))
            # Observability failure is never raised into PPO physics or optimizer code.
            return None
    def checkpoint(self,model):
        p=self.dest/'checkpoints'/f'step{model.num_timesteps}.zip'
        if not p.exists():save_model(model,p)
        self.policy_sha=sha(p);return p
    def status(self,model=None,**extra):
        atomic_json(self.dest/'live/progress.json',dict(source='Actual Training Rollout',phase=self.phase,rollout_id=self.rollout_id,
            policy_update_id=self.version,global_step=self.active_wrapper.global_step if self.active_wrapper else 0,
            sb3_n_updates=model._n_updates if model else None,checkpoint_sha256=self.policy_sha,
            render_enabled=self.human,errors=len(self.errors),**extra))
    def selected(self,episode):return self.record_rgb and (episode-1)%self.every==0 and (episode-1)//self.every<self.maximum
    def environment_step(self,wrapper,r):
        self.active_wrapper=wrapper
        # Independent execution trace is captured from the action passed into the real Gym env.
        self.executions.append(dict(global_step=r['global_step'],episode_id=r['episode_id'],episode_step=r['episode_step'],
            action_1=float(r['executed_action'][0]),action_2=float(r['executed_action'][1]),reward=r['reward'],
            actuator_ctrl_1=float(r['after']['ctrl'][0]),actuator_ctrl_2=float(r['after']['ctrl'][1])))
        if self.selected(r['episode_id']) and not self.rgb_disabled:
            start=time.perf_counter()
            try:
                if self.inject_rgb_failure:raise RuntimeError('Injected RGB failure; optimizer must continue')
                assert wrapper.unwrapped.render_mode=='rgb_array','RGB is captured directly from the real training env'
                frame=wrapper.render().copy()
                self.frames.setdefault(r['episode_id'],[]).append(frame)
                self.frame_steps.setdefault(r['episode_id'],[]).append(r['global_step'])
                if r['terminated'] or r['truncated']:
                    self.flush_episode(r['episode_id'])
            except Exception as exc:
                self.errors.append(dict(where='rgb_capture',error=repr(exc),step=r['global_step']));self.rgb_disabled=True
            self.render_seconds+=time.perf_counter()-start
        if self.human:
            self.safe(lambda:self.human_draw(wrapper,r),'human_render')
            if self.disable_after and r['global_step']>=self.disable_after:self.disable_human(wrapper,'scheduled close test')
        if self.closed_render_at is not None and not self.render_released:
            self.safe(lambda:wrapper.unwrapped.mujoco_renderer.close(),'close_render_context');self.render_released=True
        if r['global_step']%25==0:self.safe(lambda:self.status(),'status')
    def human_draw(self,wrapper,row=None):
        import glfw
        u=wrapper.unwrapped;viewer=u.mujoco_renderer.viewer
        if not viewer:
            wrapper.render();viewer=u.mujoco_renderer.viewer
        if not self.human_visible:
            assert glfw.get_window_attrib(viewer.window,glfw.VISIBLE)
            self.human_visible=True;viewer._render_every_frame=True;viewer._create_overlay=lambda:None
            def close(window):
                glfw.set_window_should_close(window,False);self.disable_human(wrapper,'window closed')
            glfw.set_window_close_callback(viewer.window,close)
            original=viewer._key_callback
            def key(window,k,scancode,action,mods):
                if action==glfw.RELEASE and k in (glfw.KEY_R,glfw.KEY_ESCAPE):self.disable_human(wrapper,'R/Esc pressed')
                else:original(window,k,scancode,action,mods)
            glfw.set_key_callback(viewer.window,key)
        viewer.add_overlay(mujoco.mjtGridPos.mjGRID_TOPRIGHT,'Actual Training Rollout',self.phase)
        viewer.add_overlay(mujoco.mjtGridPos.mjGRID_TOPRIGHT,'Rollout / policy updates',f'{self.rollout_id} / {self.version}')
        viewer.add_overlay(mujoco.mjtGridPos.mjGRID_TOPRIGHT,'Step / rollout budget',f'{wrapper.global_step} / 1024 per rollout')
        if row:
            viewer.add_overlay(mujoco.mjtGridPos.mjGRID_TOPRIGHT,'Executed stochastic action',str(row['executed_action'].round(6)))
            viewer.add_overlay(mujoco.mjtGridPos.mjGRID_TOPRIGHT,'Actual reward / return',f'{row["reward"]:.6f} / {row["cumulative_reward"]:.4f}')
            viewer.add_overlay(mujoco.mjtGridPos.mjGRID_TOPRIGHT,'Episode / step / XY cm',f'{row["episode_id"]} / {row["episode_step"]} / {100*np.linalg.norm(row["next_observation"][8:10]):.3f}')
        viewer.add_overlay(mujoco.mjtGridPos.mjGRID_TOPRIGHT,'R or Esc / close window','Disable rendering; PPO continues')
        wrapper.render()
        if self.phase=='Updating Actor/Critic' or (row and row['global_step']==100):
            pixels=np.zeros((viewer.viewport.height,viewer.viewport.width,3),np.uint8)
            # WindowViewer swaps buffers. Read the displayed FRONT buffer, rather than the
            # previous BACK frame returned by mjr_readPixels after that swap.
            import ctypes
            gl=ctypes.WinDLL('opengl32')
            gl.glReadBuffer.argtypes=[ctypes.c_uint];gl.glReadBuffer.restype=None
            gl.glReadPixels.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_uint,ctypes.c_uint,ctypes.c_void_p];gl.glReadPixels.restype=None
            gl.glPixelStorei.argtypes=[ctypes.c_uint,ctypes.c_int];gl.glPixelStorei.restype=None
            gl.glPixelStorei(0x0D05,1);gl.glReadBuffer(0x0404);gl.glReadPixels(0,0,viewer.viewport.width,viewer.viewport.height,0x1907,0x1401,pixels.ctypes.data_as(ctypes.c_void_p));gl.glReadBuffer(0x0405);gl.glPixelStorei(0x0D05,4)
            from PIL import Image
            Image.fromarray(np.flipud(pixels)).save(self.dest/'live'/('updating.png' if self.phase.startswith('Updating') else 'collecting.png'))
        if row:time.sleep(max(0.,1/self.fps))
    def disable_human(self,wrapper,reason):
        # Called after a step or GLFW event; release the context after render returns, not inside callback.
        wrapper.unwrapped.render_mode=None;self.human=False;self.closed_render_at=wrapper.global_step
        atomic_json(self.dest/'live/render_disabled.json',dict(step=wrapper.global_step,reason=reason,training_continues=True))
    def flush_episode(self,episode):
        frames=self.frames.get(episode,[])
        if frames:
            np.savez_compressed(self.dest/'rgb'/f'episode{episode}.npz',frames=np.asarray(frames),global_steps=self.frame_steps[episode],source=np.asarray('Actual Training Rollout'))
            self.frames[episode]=[]
    def _on_rollout_start(self):
        self.rollout_id+=1;self.sample_start=len(self.rows);self.phase='Collecting Experience'
        self.safe(lambda:self.status(self.model),'rollout_start_status')
    def _on_step(self):
        self.safe(self.capture_callback,'sampler_callback');return True
    def capture_callback(self):
        wrapper=self.active_wrapper;transition=wrapper.latest
        assert self.training_env.envs[0].unwrapped is wrapper.unwrapped,'Observer is not attached to the PPO training env'
        raw=self.locals['actions'][0].copy();clipped=self.locals['clipped_actions'][0].copy()
        distribution=self.model.policy.action_dist.distribution
        mu=distribution.mean[0].detach().cpu().numpy().copy();std=distribution.stddev[0].detach().cpu().numpy().copy()
        assert np.array_equal(clipped,transition['executed_action'])
        assert np.array_equal(transition['after']['ctrl'],clipped)
        assert np.array_equal(np.clip(raw,-1,1),clipped)
        before=transition['before'];after=transition['after']
        self.rows.append(dict(source='Actual Training Rollout',rollout_id=self.rollout_id,policy_update_id=self.version,
            sb3_n_updates=self.model._n_updates,global_step=transition['global_step'],episode_id=transition['episode_id'],episode_step=transition['episode_step'],
            observation=self.model._last_obs[0].copy(),observation_env=transition['observation_env'],actor_mean=mu,raw_action=raw,
            executed_action=clipped,gaussian_std=std,target_before=before['target'],target_after=after['target'],
            state_before=before['integration_state'],state_after=after['integration_state'],qpos_before=before['qpos'],qpos_after=after['qpos'],
            qvel_before=before['qvel'],qvel_after=after['qvel'],tip_before=before['tip'],tip_after=after['tip'],
            reward=transition['reward'],cumulative_reward=transition['cumulative_reward'],next_observation=transition['next_observation'],
            terminated=transition['terminated'],truncated=transition['truncated'],episode_complete=transition['terminated'] or transition['truncated'],
            policy_checkpoint_sha256=self.policy_sha,sim_time_before=before['sim_time'],sim_time_after=after['sim_time']))
    def _on_rollout_end(self):
        def check():
            recorded=np.array([r['raw_action'] for r in self.rows[self.sample_start:]])
            original=self.model.rollout_buffer.actions.reshape(-1,2)
            np.testing.assert_array_equal(recorded,original)
            obs=np.array([r['observation'] for r in self.rows[self.sample_start:]])
            np.testing.assert_array_equal(obs,self.model.rollout_buffer.observations.reshape(-1,10))
            self.buffer_checks.append(dict(rollout_id=self.rollout_id,steps=len(recorded),raw_actions_exact=True,observations_exact=True,
                action_sha256=hashlib.sha256(original.tobytes()).hexdigest()))
        self.safe(check,'rollout_buffer_consistency')
    def update_start(self,model):
        self.phase='Updating Actor/Critic';self.update_started=time.perf_counter()
        self.events.append(dict(event='update_start',update_id=self.version+1,global_step=model.num_timesteps,rollout_id=self.rollout_id,
            version_before=self.version,version_after=None,wall_time=time.time()))
        self.status(model)
        if self.human:
            self.human_draw(self.active_wrapper)
            if not self.human:self.active_wrapper.unwrapped.mujoco_renderer.close()
        print('UPDATING ACTOR/CRITIC',self.version+1,'after real env step',model.num_timesteps,flush=True)
    def update_end(self,model):
        self.version+=1;path=self.checkpoint(model);r=self.last_report
        self.events.append(dict(event='update_end',update_id=self.version,global_step=model.num_timesteps,rollout_id=self.rollout_id,
            version_before=self.version-1,version_after=self.version,wall_time=time.time(),duration_seconds=time.perf_counter()-self.update_started,
            policy_loss=r['policy_gradient_loss'],value_loss=r['value_loss'],approx_kl=r['update_kl_mean'],max_kl=r['update_kl_max'],
            clip_fraction=r['clip_fraction'],anchor_loss=r['anchor_loss'],weighted_anchor_loss=r['weighted_anchor_loss'],
            actor_bc_action_mse=r['bc_action_mse'],optimizer_steps=r['optimizer_steps'],sb3_n_updates=model._n_updates,
            actor_sha256=mean_actor_hash(model),checkpoint=str(path),checkpoint_sha256=self.policy_sha))
        rows_csv(self.dest/'ppo_update_events.csv',self.events)
        self.phase='Collecting Experience';self.status(model)
    def finish(self,model):
        rows_csv(self.dest/'training_rollouts.csv',self.rows);rows_csv(self.dest/'executed_actions.csv',self.executions)
        arrays={k:np.asarray([r[k] for r in self.rows]) for k in self.rows[0]} if self.rows else {}
        np.savez_compressed(self.dest/'training_rollouts.npz',**arrays)
        summaries=[]
        for episode in sorted({r['episode_id'] for r in self.rows}):
            rr=[r for r in self.rows if r['episode_id']==episode];complete=bool(rr[-1]['episode_complete']);d=np.array([np.linalg.norm(r['next_observation'][8:10]) for r in rr])
            summaries.append(dict(episode_id=episode,start_step=rr[0]['global_step'],end_step=rr[-1]['global_step'],steps=len(rr),
                complete=complete,stable_success_1cm=bool(np.all(d[-5:]<.01)) if complete else None,
                outcome='success' if complete and np.all(d[-5:]<.01) else 'failure' if complete else 'partial_at_training_stop',
                final_distance_cm=float(d[-1]*100),reward_sum=float(sum(r['reward'] for r in rr)),
                action_cost=float(sum(np.square(r['executed_action']).sum() for r in rr)),
                rollout_ids=sorted({r['rollout_id'] for r in rr}),policy_versions=sorted({r['policy_update_id'] for r in rr}),
                recorded_rgb=episode in self.frames))
        rows_csv(self.dest/'training_episode_summary.csv',summaries)
        for episode in self.frames:self.flush_episode(episode)
        self.phase='Complete';self.status(model)
        atomic_json(self.dest/'observation_integrity.json',dict(errors=self.errors,reset_calls=self.reset_calls,buffer_checks=self.buffer_checks,
            physics_steps=self.active_wrapper.global_step,logged_steps=len(self.rows),policy_updates=self.version,
            training_environment_identity=id(self.active_wrapper.unwrapped),human_visible=self.human_visible,
            rendering_disabled_at=self.closed_render_at,render_seconds=self.render_seconds,
            state_spec='mjSTATE_INTEGRATION',state_spec_integer=int(STATE_SPEC),
            replay_design='Actual replay uses captured RGB ONLY; full integration states are archived, never assumed to reproduce cached rendering by themselves.',
            sb3_timeout_note='reward is raw environment reward BEFORE SB3 time-limit bootstrap correction; terminal next_observation captured BEFORE vector autoreset'))
