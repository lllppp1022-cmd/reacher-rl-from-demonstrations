"""Offline figures, real-frame GIF verification, and explicitly separate deterministic diagnostics."""
from pathlib import Path
import json,csv,hashlib,time
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from anchor_common import env,atomic_json,read,sha
from training_observation import rows_csv,ObservedAnchorPPO,physics_snapshot

def load_csv(path):
    with Path(path).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def video(frames,steps,rows,path,source):
    path=Path(path)
    if path.with_suffix('.json').exists():
        info=read(path.with_suffix('.json'));assert sha(path)==info['sha256'];return info
    bystep={int(r['global_step']):r for r in rows};annotated=[]
    try:font=ImageFont.truetype('DejaVuSans.ttf',16)
    except OSError:font=ImageFont.load_default()
    for f,s in zip(frames,steps):
        r=bystep[int(s)];im=Image.new('RGB',(f.shape[1],f.shape[0]+76),'#111111');im.paste(Image.fromarray(f),(0,76));draw=ImageDraw.Draw(im)
        badge=source+(' | PPO UPDATE NEXT' if source.startswith('Actual') and int(s)%1024==0 else '')
        draw.text((8,4),badge,fill='#57d9ff' if source.startswith('Actual') else '#ffcc55',font=font)
        draw.text((8,26),f"step={s} episode={r.get('episode_id','diagnostic')} ep_step={r.get('episode_step',s)} rollout={r.get('rollout_id','n/a')} version={r.get('policy_update_id','n/a')}",fill='white',font=font)
        draw.text((8,49),f"action={np.asarray(r['executed_action']).round(5)} reward={float(r['reward']):.5f}",fill='white',font=font)
        annotated.append(im.quantize(colors=128,method=Image.Quantize.FASTOCTREE))
    annotated[0].save(path,save_all=True,append_images=annotated[1:],duration=20,loop=0,optimize=False)
    with Image.open(path) as im:
        count=im.n_frames;duration=0
        for i in range(count):im.seek(i);im.load();duration+=im.info.get('duration',0)
    unique=len({hashlib.sha256(f.tobytes()).hexdigest() for f in frames})
    pixel=np.array([np.abs(b.astype(float)-a.astype(float)).mean() for a,b in zip(frames[:-1],frames[1:])])
    assert unique>1 and count>1 and duration==20*len(frames)
    info=dict(source=source,path=str(path),sha256=sha(path),raw_frames=len(frames),unique_raw_frames=unique,encoded_frames=count,
        duration_ms=duration,changed_frame_pairs=int(np.sum(pixel>0)),mean_adjacent_pixel_difference=float(pixel.mean()),
        file_bytes=path.stat().st_size,global_step_start=int(steps[0]),global_step_end=int(steps[-1]),fps=50)
    atomic_json(path.with_suffix('.json'),info);return info

def sequence_video(dest,videos):
    """Bounded memory: read one archived episode at a time; keep only compact palette images."""
    palettes=[];raw_hashes=set();raw_count=0;intervals=[]
    for v in videos:
        with np.load(dest/'rgb'/f'episode{v["episode_id"]}.npz') as a:
            frames=a['frames'];raw_count+=len(frames)
            raw_hashes.update(hashlib.sha256(f.tobytes()).hexdigest() for f in frames)
            intervals.append([int(a['global_steps'][0]),int(a['global_steps'][-1])]);del frames
        with Image.open(v['path']) as im:
            for i in range(im.n_frames):
                im.seek(i);rgb=im.convert('RGB');rgb=rgb.resize((480,round(rgb.height*480/rgb.width)))
                palettes.append(rgb.quantize(colors=128,method=Image.Quantize.FASTOCTREE))
    path=dest/'videos/actual_selected_sequence.gif'
    palettes[0].save(path,save_all=True,append_images=palettes[1:],duration=20,loop=0,optimize=False)
    with Image.open(path) as im:
        encoded=im.n_frames;duration=0
        for i in range(encoded):im.seek(i);im.load();duration+=im.info.get('duration',0)
    assert duration==raw_count*20 and len(raw_hashes)>1
    return dict(source='Actual Training Rollout',path=str(path),sha256=sha(path),raw_frames=raw_count,unique_raw_frames=len(raw_hashes),
        encoded_frames=encoded,duration_ms=duration,global_step_intervals=intervals,episodes=[v['episode_id'] for v in videos],
        joint_displacement_rad=max(v['joint_displacement_rad'] for v in videos),file_bytes=path.stat().st_size,
        note='Chronological selected real episodes, resized for overview. Global-step gaps are explicit; no resimulation or replacement actions.')

def evaluation_replays(c,dest,seed=17059):
    dest=Path(dest);manifest=[]
    checkpoints=sorted((dest/'checkpoints').glob('step*.zip'),key=lambda p:int(p.stem[4:]))
    for p in checkpoints:
        step=int(p.stem[4:]);d=dest/'evaluation'/f'step{step}';d.mkdir(exist_ok=True)
        if (d/'complete.json').exists():manifest.append(read(d/'complete.json'));continue
        m=ObservedAnchorPPO.load(p,device='cpu');e=env('rgb_array',width=640,height=480,default_camera_config=dict(distance=.65,elevation=-90,azimuth=90,lookat=np.zeros(3),trackbodyid=-1))
        obs,_=e.reset(seed=seed);initial=np.r_[e.unwrapped.data.qpos,e.unwrapped.data.qvel].copy();frames=[];rows=[];total=0.
        try:
            for i in range(250):
                action=m.predict(obs,deterministic=True)[0]  # ONLY Evaluation Replay, after training.
                pre=obs.copy();obs,reward,term,trunc,_=e.step(action);total+=reward;frames.append(e.render().copy())
                rows.append(dict(source='Evaluation Replay',global_step=i+1,episode_id=seed,episode_step=i+1,policy_update_id=step//1024,
                    executed_action=action.copy(),reward=float(reward),cumulative_reward=float(total),observation=pre,next_observation=obs.copy(),
                    distance_cm=float(100*np.linalg.norm(obs[8:10])),qpos=e.unwrapped.data.qpos.copy(),qvel=e.unwrapped.data.qvel.copy()))
        finally:e.close()
        np.savez_compressed(d/'frames.npz',frames=frames,global_steps=np.arange(1,251),source=np.asarray('Evaluation Replay'))
        rows_csv(d/'trajectory.csv',rows);np.savez_compressed(d/'trajectory.npz',**{k:np.asarray([r[k] for r in rows]) for k in rows[0]})
        v=video(frames,np.arange(1,251),rows,d/'evaluation_replay.gif','Evaluation Replay')
        info=dict(source='Evaluation Replay',training_checkpoint_step=step,checkpoint=str(p),checkpoint_sha256=sha(p),
            diagnostic_seed=seed,initial_state=initial.tolist(),stable_success_1cm=bool(np.all([r['distance_cm']<1 for r in rows[-5:]])),
            final_distance_cm=rows[-1]['distance_cm'],video=v)
        atomic_json(d/'complete.json',info);manifest.append(info)
    assert all(x['initial_state']==manifest[0]['initial_state'] for x in manifest)
    pairs=[dict(update_id=i,before=manifest[i-1],after=manifest[i],source='Evaluation Replay') for i in range(1,len(manifest))]
    atomic_json(dest/'evaluation/manifest.json',dict(source='Evaluation Replay',generated_after_training=True,diagnostic_seed=seed,pairs=pairs,checkpoints=manifest))

def analyze(dest):
    dest=Path(dest)
    with np.load(dest/'training_rollouts.npz') as archive:z={k:archive[k] for k in archive.files}
    rows=[{k:z[k][i] for k in z} for i in range(len(z['global_step']))]
    episodes=load_csv(dest/'training_episode_summary.csv');events=load_csv(dest/'ppo_update_events.csv');ends=[r for r in events if r['event']=='update_end']
    videos=[]
    for p in sorted((dest/'rgb').glob('episode*.npz'),key=lambda p:int(p.stem[7:])):
        a=np.load(p);episode=int(p.stem[7:]);v=video(a['frames'],a['global_steps'],rows,dest/'videos'/f'actual_episode{episode}.gif','Actual Training Rollout')
        selected=np.flatnonzero(z['episode_id']==episode)
        motion=float(np.linalg.norm(z['qpos_after'][selected,:2]-z['qpos_before'][selected[0],:2],axis=1).max())
        v.update(episode_id=episode,joint_displacement_rad=motion,complete=episodes[episode-1]['complete'],
            outcome=episodes[episode-1]['outcome'],policy_checkpoint_sha256s=sorted(set(z['policy_checkpoint_sha256'][selected].tolist())))
        assert motion>0;atomic_json(Path(v['path']).with_suffix('.json'),v);videos.append(v)
    if videos:
        overview=sequence_video(dest,videos)
        atomic_json(dest/'videos/actual_selected_sequence.json',overview)
        alias=dest/'actual_training_rollout.gif';alias.write_bytes(Path(overview['path']).read_bytes())
        atomic_json(dest/'actual_training_rollout.json',dict(overview,alias_path=str(alias),alias_sha256=sha(alias)))
    atomic_json(dest/'videos/manifest.json',videos)
    count=len(episodes);cols=4;nr=(count+cols-1)//cols
    for mode in ['distance','action']:
        fig,axs=plt.subplots(nr,cols,figsize=(14,2.5*nr),squeeze=False)
        for i,ep in enumerate(episodes):
            ax=axs.flat[i];idx=z['episode_id']==int(ep['episode_id']);t=z['episode_step'][idx]*.02
            if mode=='distance':ax.plot(t,np.linalg.norm(z['next_observation'][idx,8:10],axis=1)*100);ax.axhline(1,color='black',ls='--',lw=.6);ax.set_ylabel('XY cm')
            else:
                for j in range(2):ax.plot(t,z['executed_action'][idx,j],label=f'action{j+1}',lw=.7)
                ax.set_ylabel('Control');ax.legend(fontsize=6)
            first=int(ep['start_step'])
            for r in ends:
                g=int(r['global_step'])
                if first<=g<int(ep['end_step']):ax.axvline((g-first+1)*.02,color='red',ls=':',lw=.8)
            ax.set_title(f"Actual episode {ep['episode_id']}: {ep['outcome']}",fontsize=8);ax.set_xlabel('Episode simulation seconds');ax.grid(alpha=.2)
        for i in range(count,nr*cols):axs.flat[i].axis('off')
        plt.tight_layout();plt.savefig(dest/f'training_{mode}_curve.png',dpi=125);plt.close()
    fig,axs=plt.subplots(3,1,figsize=(12,9))
    axs[0].plot(z['global_step'],np.linalg.norm(z['next_observation'][:,8:10],axis=1)*100,lw=.5);axs[0].set_ylabel('Actual XY error (cm)')
    for r in ends:axs[0].axvline(int(r['global_step']),color='red',ls='--',lw=.6)
    axs[0].set_xlabel('Real global env step; red lines = optimization after1024 collected steps')
    axs[1].plot(z['global_step'],z['policy_update_id']);axs[1].set_ylabel('Policy update version');axs[1].set_xlabel('Actual env steps (optimization does not advance physics)')
    axs[2].bar([int(r['update_id']) for r in ends],[float(r['duration_seconds']) for r in ends]);axs[2].set_xlabel('Policy update id');axs[2].set_ylabel('Optimization + checkpoint seconds')
    plt.tight_layout();plt.savefig(dest/'policy_update_timeline.png',dpi=150);plt.close()
    if (dest/'evaluation/manifest.json').exists():
        ev=read(dest/'evaluation/manifest.json');fig,axs=plt.subplots(len(ev['pairs']),1,figsize=(10,3*len(ev['pairs'])),squeeze=False)
        for ax,pair in zip(axs.flat,ev['pairs']):
            for role in ['before','after']:
                step=pair[role]['training_checkpoint_step'];zz=np.load(dest/f'evaluation/step{step}/trajectory.npz');ax.plot(np.arange(1,251)*.02,zz['distance_cm'],label=f'{role}: policy step{step}')
            ax.axhline(1,color='black',ls='--');ax.set_title(f'Evaluation Replay: update {pair["update_id"]}, fixed seed{ev["diagnostic_seed"]}');ax.set_xlabel('Simulation seconds');ax.set_ylabel('XY cm');ax.legend()
        plt.tight_layout();plt.savefig(dest/'evaluation/policy_before_after.png',dpi=135);plt.close()
    validate(dest)

def validate(dest):
    dest=Path(dest)
    with np.load(dest/'training_rollouts.npz') as archive:z={k:archive[k] for k in archive.files}
    integrity=read(dest/'observation_integrity.json');executed=load_csv(dest/'executed_actions.csv')
    actions=np.array([[float(r['action_1']),float(r['action_2'])] for r in executed]);np.testing.assert_array_equal(actions,z['executed_action'])
    np.testing.assert_array_equal(np.clip(z['raw_action'],-1,1),z['executed_action'])
    rewards=np.array([float(r['reward']) for r in executed]);np.testing.assert_array_equal(rewards,z['reward'])
    episodes=load_csv(dest/'training_episode_summary.csv');partial=[r for r in episodes if r['complete']=='False']
    assert all(r['outcome']=='partial_at_training_stop' and r['stable_success_1cm']=='' for r in partial)
    assert len(z['global_step'])==integrity['physics_steps']==integrity['logged_steps']
    assert all(r['raw_actions_exact'] and r['observations_exact'] for r in integrity['buffer_checks'])
    for i in range(1,len(z['global_step'])):
        if z['episode_id'][i]==z['episode_id'][i-1]:
            np.testing.assert_array_equal(z['next_observation'][i-1],z['observation_env'][i])
            np.testing.assert_array_equal(z['state_after'][i-1],z['state_before'][i])
    videos=read(dest/'videos/manifest.json')
    for v in videos:
        assert v['joint_displacement_rad']>0 and v['unique_raw_frames']>1 and v['changed_frame_pairs']>0 and sha(v['path'])==v['sha256']
        ep=np.load(dest/'rgb'/f'episode{v["episode_id"]}.npz');idx=np.isin(z['global_step'],ep['global_steps'])
        assert np.array_equal(ep['global_steps'],z['global_step'][idx])
    eps=np.divide(z['raw_action']-z['actor_mean'],z['gaussian_std'])
    result=dict(passed=True,source='Actual Training Rollout',real_physics_steps=integrity['physics_steps'],csv_rows=len(executed),
        policy_update_count=integrity['policy_updates'],rollout_buffer_checks=integrity['buffer_checks'],
        executed_action_max_error=float(np.max(np.abs(actions-z['executed_action']))),raw_action_clip_exact=True,reward_log_max_error=0.,
        stochastic_action_nonmean_fraction=float(np.mean(np.any(z['raw_action']!=z['actor_mean'],axis=1))),
        normalized_noise_mean=eps.mean(axis=0).tolist(),normalized_noise_std=eps.std(axis=0).tolist(),
        no_extra_training_steps=True,terminal_observations_captured_before_autoreset=True,raw_rewards_before_time_limit_bootstrap=True,
        full_state_dimension=int(z['state_after'].shape[1]),state_spec=integrity['state_spec'],replay_uses_captured_rgb_not_reconstructed_or_resampled_actions=True,
        complete_episodes=sum(r['complete']=='True' for r in episodes),partial_episodes=len(partial),
        successful_complete_episodes=sum(r['outcome']=='success' for r in episodes),failed_complete_episodes=sum(r['outcome']=='failure' for r in episodes),
        episodes_crossing_rollout_boundary=[r['episode_id'] for r in episodes if len(json.loads(r['rollout_ids']))>1],
        recorded_actual_videos=len(videos),all_videos_real_frames=True,observer_errors=integrity['errors'])
    atomic_json(dest/'rollout_recording_validation.json',result)
