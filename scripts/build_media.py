"""Compose real archived frames; do not simulate or sample a replacement trajectory."""
from pathlib import Path
import sys,json,hashlib,shutil,csv,importlib.util
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageSequence
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.runtime import read,sha,atomic_json
H=ROOT.parent/'reacher_anchor_calibration';V=ROOT/'assets/videos';F=ROOT/'assets/figures'
try:FONT=ImageFont.truetype('DejaVuSans.ttf',15)
except OSError:FONT=ImageFont.load_default()
def frames(path,size=(320,240),fps=20):
    """Decode merged GIF durations before temporal resampling; no fabricated poses."""
    result=[];next_t=0.;t=0.
    with Image.open(path) as im:
        for frame in ImageSequence.Iterator(im):
            duration=frame.info.get('duration',20)
            while next_t<t+duration-1e-6:
                result.append(frame.convert('RGB').resize(size));next_t+=1000/fps
            t+=duration
    return result
def card(lines,seconds=3):
    im=Image.new('RGB',(640,540),'#152030');d=ImageDraw.Draw(im)
    for i,line in enumerate(lines):d.text((25,60+i*40),line,font=FONT,fill='white')
    return [im]*int(seconds*20)
def save(name,ims,sources,kind):
    path=V/(name+'.gif');pal=[im.quantize(colors=128,method=Image.Quantize.FASTOCTREE) for im in ims]
    pal[0].save(path,save_all=True,append_images=pal[1:],duration=50,loop=0,optimize=False)
    previous=None;changes=0;duration=0;count=0;hashes=set()
    with Image.open(path) as im:
        for fr in ImageSequence.Iterator(im):
            b=fr.convert('RGB').tobytes();h=hashlib.sha256(b).hexdigest();hashes.add(h);changes+=int(previous is not None and b!=previous);previous=b;count+=1;duration+=fr.info.get('duration',0)
    assert changes>0 and count>1 and abs(duration-len(ims)*50)<100
    info=dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),content=kind,source_records=sources,resampled_input_frames=len(ims),encoded_frames=count,duration_seconds=duration/1000,unique_decoded_frames=len(hashes),changed_frame_pairs=changes,temporal_sampling='20fps from real source GIF durations; holds/titles explicit; no physics interpolation',mp4_status='not generated: no installed ffmpeg/PyAV encoding backend')
    return info
def main():
    V.mkdir(parents=True,exist_ok=True);F.mkdir(parents=True,exist_ok=True);manifest=read(H/'videos/manifest.json')
    sources=[];panels=[];beforeafter=[]
    for seed,case in [(7059,'typical'),(7012,'difficult')]:
        items=[]
        for role in ['ikpd','bc','control','best']:
            options=[x for x in manifest if x['seed']==seed and ((role in ['ikpd','bc'] and x['role']==role) or (role=='control' and x.get('actual_training_steps')==100352 and 'long_control' in x['path']) or (role=='best' and x.get('actual_training_steps')==100352 and 'long_decay' in x['path'] and 'strong' in x['path']))]
            if options:entry=options[0];p=Path(entry['path'])
            else:
                entry=read(ROOT/'results/control7012/recording.json');p=Path(entry['path'])
                assert role=='control' and seed==7012
            assert sha(p)==entry['sha256'];assert entry['joint_displacement_rad']>0
            sources.append(dict(role=role,seed=seed,source_project='reacher_anchor_calibration' if options else 'portfolio_repeat_diagnostic',source_file=p.name,sha256=sha(p),model_sha256=entry.get('model_sha256'),raw_frames=entry['raw_frames'],source_encoded_frames=entry['encoded_frames'],source_duration_ms=entry['duration_ms'],joint_displacement_rad=entry['joint_displacement_rad']))
            items.append(frames(p))
        for i in range(100):
            im=Image.new('RGB',(640,540),'#152030');d=ImageDraw.Draw(im)
            for j,ims in enumerate(items):
                x=(j%2)*320;y=(j//2)*270;im.paste(ims[i],(x,y+30));d.text((x+5,y+5),['IK+PD','BC+DAgger','No anchor 100352','Strong decay 100352'][j],font=FONT,fill='white')
            d.text((5,520),f'Evaluation Replay | demo seed {seed} ({case}); not final statistics',font=FONT,fill='yellow');panels.append(im)
            im2=Image.new('RGB',(640,540),'#152030');im2.paste(items[1][i].resize((320,400)),(0,80));im2.paste(items[3][i].resize((320,400)),(320,80));dr=ImageDraw.Draw(im2)
            dr.text((10,15),f'Evaluation Replay | same start / seed {seed}',font=FONT,fill='yellow');dr.text((10,50),'BC initialization',font=FONT,fill='white');dr.text((330,50),'Strong decay 100352',font=FONT,fill='white');beforeafter.append(im2)
    panels[70].save(F/'robot_control.png')
    actual=ROOT/'results/online_demo';assert read(actual/'rollout_recording_validation.json')['passed']
    liveframes=frames(actual/'actual_training_rollout.gif',size=(640,540))
    live_source=read(actual/'actual_training_rollout.json')
    with np.load(actual/'training_rollouts.npz') as z:
        data={k:z[k] for k in ['global_step','episode_id','executed_action','next_observation','qpos_after','policy_checkpoint_sha256','actor_mean','raw_action','policy_update_id']}
    # Add mean/raw/distance overlays from the actual row, never call predict or env.step.
    rgbsteps=np.concatenate([np.load(actual/'rgb'/f'episode{ep}.npz')['global_steps'] for ep in [1,5,9,13]])
    recent_update=None
    for i,im in enumerate(liveframes):
        j=int(rgbsteps[min(len(rgbsteps)-1,int(i*2.5))])-1;d=ImageDraw.Draw(im)
        d.rectangle((0,480,640,540),fill='#152030');d.text((8,483),f"Actual Training Rollout | XY {100*np.linalg.norm(data['next_observation'][j,8:10]):.3f} cm",font=FONT,fill='yellow')
        d.text((8,505),f"mu {data['actor_mean'][j].round(4)} raw {data['raw_action'][j].round(4)}",font=FONT,fill='white')
        if i and int(data['policy_update_id'][j])!=int(data['policy_update_id'][int(rgbsteps[min(len(rgbsteps)-1,int((i-1)*2.5))])-1]):
            recent_update=(i,int(data['policy_update_id'][j]))
        if recent_update and i-recent_update[0]<12:
            d.rectangle((0,440,640,477),fill='#5c3210');d.text((8,447),f'Recorded Updating Actor/Critic #{recent_update[1]} after step {1024*recent_update[1]}',font=FONT,fill='white')
    idx=data['episode_id']==13;distance=np.linalg.norm(data['next_observation'][idx,8:10],axis=1)*100
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,1,figsize=(8,6));t=np.arange(1,251)*.02;axs[0].plot(t,distance);axs[0].axhline(1,color='red',ls='--');axs[0].set(ylabel='XY error (cm)',title='Actual failure: episode 13 (not an evaluation episode)');axs[1].plot(t,data['executed_action'][idx]);axs[1].set(xlabel='Episode time (s)',ylabel='Executed actions');fig.tight_layout();fig.savefig(F/'failure_episode13.png',dpi=140);plt.close(fig)
    fail=frames(actual/'videos/actual_episode13.gif',size=(640,540));fail+=card(['Actual Training Rollout: episode 13','Complete 250-step failure retained',f'Final XY error = {distance[-1]:.4f} cm','Crosses PPO rollout boundary at step 3072','Observed control change; cause not proven'],5)
    fail+=panels[100:200]
    real_record=dict(raw_frames=live_source['raw_frames'],source_encoded_frames=live_source['encoded_frames'],source_duration_ms=live_source['duration_ms'],sha256=sha(actual/'actual_training_rollout.gif'),model_sha256s=sorted(set(data['policy_checkpoint_sha256'].tolist())),joint_displacement_rad=float(np.linalg.norm(data['qpos_after'][idx,:2]-data['qpos_after'][idx,:2][0],axis=1).max()),source='Actual Training Rollout')
    videos=[save('01_same_start_comparison',panels,sources,'Evaluation Replay; two demo seeds'),save('02_actual_online_learning',liveframes,[real_record],'Actual Training Rollout; episodes 1,5,9,13 with gaps'),save('03_bc_vs_final_policy',beforeafter,sources,'Evaluation Replay: BC vs frozen 100352-step model'),save('04_failure_cases',fail,[real_record]+sources,'Actual failure episode13 + difficult Evaluation Replay')]
    overview=card(['Stable Online RL from Demonstrations','MuJoCo Reacher | simulation only','Expert -> BC -> DAgger -> PPO + BC anchor','3 training seeds; 100 frozen evaluation seeds'],5)+panels+card(['Actual Training Rollout follows','Gaussian actions, real rewards and policy versions','Each 1024 environment steps -> Actor/Critic update','Recorded episodes are separated by unrecorded gaps'],5)+liveframes+beforeafter+fail+card(['Results: historical CSV recomputation','BC: 58.67% / 2.3029 cm','Strong decay: 64.67% / 1.9986 cm','Descriptive results, not statistical significance','No real-robot or Sim-to-Real validation'],5)
    videos.append(save('00_project_overview',overview,sources+[real_record],'70-second portfolio montage; source labels preserved'))
    # Static scientific figures generated directly from recomputed CSV.
    with (ROOT/'reports/final_results.csv').open(encoding='utf-8-sig') as f:r=list(csv.DictReader(f))
    fig,ax=plt.subplots(1,2,figsize=(11,4));labels=['BC','No anchor','Fixed medium','Fixed strong','Decay strong'];x=np.arange(5)
    for a,k,sd,title in [(ax[0],'stable_success_pct','stable_success_pct_seed_sd','Stable success (%)'),(ax[1],'mean_final_cm','mean_final_cm_seed_sd','Final XY error (cm)')]:
        a.bar(x,[float(v[k]) for v in r],yerr=[float(v[sd]) for v in r],capsize=4,color=['#777777','#d97668','#6395ac','#7b8cba','#409b88']);a.set_xticks(x,labels,rotation=20);a.set_ylabel(title);a.grid(axis='y',alpha=.2)
    fig.suptitle('Frozen seeds16000-16099; mean +/- sample SD over3 training seeds');fig.tight_layout();fig.savefig(F/'final_comparison.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(14,2.6));ax.axis('off');names=['IK+PD\nExpert','Demonstration\nDataset','BC','DAgger','PPO\nFine-tuning','BC\nAnchoring','Frozen\nEvaluation']
    for i,name in enumerate(names):
        ax.text(i,0,name,ha='center',va='center',bbox=dict(boxstyle='round,pad=.5',fc='#e5edf2',ec='#395970'),fontsize=11)
        if i<6:ax.annotate('',xy=(i+.67,0),xytext=(i+.34,0),arrowprops=dict(arrowstyle='->',color='#395970'))
    ax.set(xlim=(-.6,6.7),ylim=(-1,1));fig.tight_layout();fig.savefig(F/'pipeline.png',dpi=160);plt.close(fig)
    for name in ['training_distance_curve.png','training_action_curve.png','policy_update_timeline.png']:shutil.copy2(actual/name,F/name)
    atomic_json(ROOT/'reports/video_validation.json',dict(passed=True,videos=videos,overview_duration_seconds=videos[-1]['duration_seconds'],mp4_backend_available=bool(importlib.util.find_spec('imageio_ffmpeg') or importlib.util.find_spec('av')),new_experiment_statistics=False))
    print('MEDIA COMPLETE',[(x['path'],x['duration_seconds']) for x in videos])
if __name__=='__main__':main()
