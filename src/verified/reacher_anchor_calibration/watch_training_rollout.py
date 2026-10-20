"""真实录制帧回放 / 明确标注的Evaluation Replay / 实时PPO更新进度；实际回放不调用策略或env.step。"""
import argparse,json,time,os
from pathlib import Path
import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import Image,ImageTk,ImageDraw

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def run(args):
    dest=Path(args.run).resolve();root=tk.Tk();root.title('Reacher — Actual Training Rollout / Evaluation Replay')
    root.geometry('1320x800');mode=tk.StringVar(value=args.mode);selection=tk.StringVar();playing=[True];frame=[0];displayed=[0];photos=[];datasets=[];versions=set();chosen=[None]
    top=ttk.Frame(root);top.pack(fill='x');ttk.Label(top,text='数据来源 / Source:').pack(side='left')
    modes=ttk.Combobox(top,textvariable=mode,values=['actual','compare','progress'],state='readonly',width=14);modes.pack(side='left')
    selector=ttk.Combobox(top,textvariable=selection,state='readonly',width=25);selector.pack(side='left')
    title=ttk.Label(root,text='',font=('Arial',15,'bold'));title.pack(fill='x')
    details=ttk.Label(root,text='',font=('Consolas',10),justify='left');details.pack(fill='x')
    panels=ttk.Frame(root);panels.pack(fill='both',expand=True);left=ttk.Label(panels);right=ttk.Label(panels);left.pack(side='left',expand=True);right.pack(side='left',expand=True)
    controls=ttk.Frame(root);controls.pack(fill='x');ttk.Button(controls,text='Pause / Continue',command=lambda:playing.__setitem__(0,not playing[0])).pack(side='left')
    speed=tk.DoubleVar(value=args.speed);ttk.Label(controls,text='Speed:').pack(side='left');ttk.Scale(controls,from_=.25,to=4,variable=speed).pack(side='left',fill='x',expand=True)
    slider=tk.IntVar();scale=ttk.Scale(root,from_=0,to=249,variable=slider,command=lambda v:frame.__setitem__(0,int(float(v))));scale.pack(fill='x')
    footer=ttk.Label(root,text='Space pause/continue; Left/Right previous/next frame; Esc close. Recorded frames only.');footer.pack(fill='x')
    data=None;row_by_step={};events={};manifest=None
    if (dest/'training_rollouts.npz').exists():
        with np.load(dest/'training_rollouts.npz') as archive:data={k:archive[k] for k in archive.files}
        row_by_step={int(s):i for i,s in enumerate(data['global_step'])}
    if (dest/'ppo_update_events.csv').exists():
        import csv
        with (dest/'ppo_update_events.csv').open(encoding='utf-8-sig') as f:
            events={int(r['global_step']):r for r in csv.DictReader(f) if r['event']=='update_end'}
    if (dest/'evaluation/manifest.json').exists():manifest=read(dest/'evaluation/manifest.json')
    def options(*unused):
        nonlocal manifest
        if manifest is None and (dest/'evaluation/manifest.json').exists():manifest=read(dest/'evaluation/manifest.json')
        if mode.get()=='actual':
            choices=[p.stem for p in sorted((dest/'rgb').glob('episode*.npz'),key=lambda p:int(p.stem[7:]))]
            if not choices:raise FileNotFoundError('该run没有真实RGB记录；请使用--record-training-rollouts，或选择progress查看轻量日志')
            if f'episode{args.episode}' not in choices:raise FileNotFoundError(f'回合{args.episode}没有预录RGB。实际可选：{choices}；其他回合可查看CSV曲线，不能替换成另一条轨迹')
            selector['values']=choices;selection.set(f'episode{args.episode}')
        elif mode.get()=='compare':
            if not manifest:raise FileNotFoundError('Evaluation Replay尚未生成；它与实际训练轨迹是两种不同来源')
            choices=[f'update{x["update_id"]}' for x in manifest['pairs']];selector['values']=choices;selection.set(f'update{args.update}' if f'update{args.update}' in choices else choices[0])
        else:selector['values']=[];selection.set('Live status')
        load()
    def load(*unused):
        nonlocal datasets,data,row_by_step
        datasets=[];frame[0]=0
        if mode.get()=='actual':
            if data is None:
                with np.load(dest/'training_rollouts.npz') as archive:data={k:archive[k] for k in archive.files}
                row_by_step={int(s):i for i,s in enumerate(data['global_step'])}
            path=dest/'rgb'/(selection.get()+'.npz');a=np.load(path);assert str(a['source'])=='Actual Training Rollout';datasets=[(a['frames'],a['global_steps'],'Actual Training Rollout')]
        elif mode.get()=='compare':
            pair=next(x for x in manifest['pairs'] if f'update{x["update_id"]}'==selection.get())
            for role in ['before','after']:
                step=pair[role]['training_checkpoint_step'];a=np.load(dest/f'evaluation/step{step}/frames.npz');assert str(a['source'])=='Evaluation Replay'
                datasets.append((a['frames'],a['global_steps'],f'Evaluation Replay — {role}, policy step{step}'))
            chosen[0]=pair
        scale.configure(to=len(datasets[0][0])-1 if datasets else 1)
    modes.bind('<<ComboboxSelected>>',options);selector.bind('<<ComboboxSelected>>',load)
    root.bind('<space>',lambda e:playing.__setitem__(0,not playing[0]));root.bind('<Left>',lambda e:frame.__setitem__(0,max(0,frame[0]-1)))
    root.bind('<Right>',lambda e:frame.__setitem__(0,min(len(datasets[0][0])-1,frame[0]+1)) if datasets else None);root.bind('<Escape>',lambda e:root.destroy())
    root.update();assert root.winfo_viewable();options()
    def tick():
        if not root.winfo_exists():return
        photos.clear()
        if mode.get()=='progress':
            p=dest/'live/progress.json'
            if p.exists():
                status=read(p);title.configure(text=f"Actual Training Rollout — {status['phase']}")
                details.configure(text=json.dumps(status,ensure_ascii=False,indent=2));left.configure(image='');right.configure(image='')
            else:title.configure(text='Waiting for training progress.json…')
        else:
            i=min(frame[0],len(datasets[0][0])-1);slider.set(i)
            title.configure(text='Actual Training Rollout — 真实训练已发生的RGB画面' if mode.get()=='actual' else 'Evaluation Replay — 独立固定起点、确定性策略对比')
            for widget,(frames,steps,label) in zip([left,right],datasets):
                photo=ImageTk.PhotoImage(Image.fromarray(frames[i]));photos.append(photo);widget.configure(image=photo)
            if mode.get()=='actual':
                right.configure(image='');g=int(datasets[0][1][i]);k=row_by_step[g];versions.add(int(data['policy_update_id'][k]))
                phase='Updating Actor/Critic follows this recorded step (no physics step during update)' if g in events else 'Collecting Experience'
                details.configure(text=f"{phase}\nrollout={data['rollout_id'][k]} policy_version={data['policy_update_id'][k]} global_step={g} episode={data['episode_id'][k]} ep_step={data['episode_step'][k]}\n"
                    f"mean={data['actor_mean'][k].round(6)} sampled={data['raw_action'][k].round(6)} executed={data['executed_action'][k].round(6)} std={data['gaussian_std'][k].round(6)}\n"
                    f"reward={data['reward'][k]:.6f} return={data['cumulative_reward'][k]:.4f} XY_cm={100*np.linalg.norm(data['next_observation'][k,8:10]):.4f}")
            else:
                pair=chosen[0];details.configure(text=f"Evaluation Replay ONLY | fixed diagnostic seed={manifest['diagnostic_seed']} | update={pair['update_id']} | frame={i+1}/250\n"
                    f"BEFORE step{pair['before']['training_checkpoint_step']} finalXY={pair['before']['final_distance_cm']:.3f}cm  | AFTER step{pair['after']['training_checkpoint_step']} finalXY={pair['after']['final_distance_cm']:.3f}cm\n"
                    'Both start at identical saved initial states. These deterministic trajectories were NOT used for PPO rollout updates.')
            if playing[0]:frame[0]=(i+1)%len(datasets[0][0])
        displayed[0]+=1
        if displayed[0]==10 and datasets:
            ims=[Image.fromarray(d[0][min(frame[0],len(d[0])-1)]) for d in datasets];im=Image.new('RGB',(sum(x.width for x in ims),max(x.height for x in ims)+45),'#222222');draw=ImageDraw.Draw(im);x=0
            for picture,ds in zip(ims,datasets):im.paste(picture,(x,45));draw.text((x+6,8),ds[2],fill='white');x+=picture.width
            im.save(dest/'live'/f'{mode.get()}_replay_preview.png')
        if args.max_frames and displayed[0]>=args.max_frames:root.destroy();return
        root.after(max(1,int(1000/(args.fps*max(.25,speed.get())))),tick)
    root.after(1,tick)
    try:root.mainloop()
    finally:
        path=dest/'live'/f'replay_{os.getpid()}.json';path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(dict(native_window_visible=True,mode=mode.get(),frames_displayed=displayed[0],policy_versions_displayed=sorted(versions),
            actual_replay_calls_policy_predict=False,actual_replay_calls_env_step=False,source='Actual Training Rollout' if mode.get()!='compare' else 'Evaluation Replay'),indent=2),encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True,help='record_training_rollouts.py的独立输出目录')
    p.add_argument('--mode',choices=['actual','compare','progress'],default='actual');p.add_argument('--episode',type=int,default=1);p.add_argument('--update',type=int,default=1)
    p.add_argument('--fps',type=float,default=50);p.add_argument('--speed',type=float,default=1);p.add_argument('--max-frames',type=int,help='命令验证时自动关闭；正常观看省略')
    run(p.parse_args())
