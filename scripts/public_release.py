"""Reviewed GIF/publication tooling. Does not run or change RL training."""
from pathlib import Path
import sys,os,re,json,csv,hashlib,subprocess,io,time,urllib.request,urllib.parse
from html.parser import HTMLParser
from PIL import Image,ImageSequence,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from src.runtime import read,atomic_json,sha
from github_remote import credential,api,OWNER,FULL,URL
NAMES=['00_project_overview.gif','01_same_start_comparison.gif','02_actual_online_learning.gif','03_bc_vs_final_policy.gif','04_failure_cases.gif']
PREVIEWS=['01_same_start_comparison_preview.gif','02_actual_online_learning_preview.gif']
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)
def decode(data):
    frames=0;duration=0;changed=0;last=None;unique=set()
    with Image.open(io.BytesIO(data)) as im:
        size=list(im.size);comment=im.info.get('comment',b'')
        for frame in ImageSequence.Iterator(im):
            digest=hashlib.sha256(frame.convert('RGB').tobytes()).hexdigest();unique.add(digest)
            changed+=int(last is not None and digest!=last);last=digest;frames+=1;duration+=frame.info.get('duration',0)
    assert frames>1 and changed>0 and duration>0
    return dict(width=size[0],height=size[1],frames=frames,duration_ms=duration,unique_decoded_frames=len(unique),changed_pairs=changed,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),comment_bytes=len(comment))
def preview(source,target):
    frames=[];t=0;next_t=0
    with Image.open(source) as im:
        size=(480,round(im.height*480/im.width))
        for frame in ImageSequence.Iterator(im):
            end=t+frame.info.get('duration',50)
            while next_t<end:
                frames.append(frame.convert('RGB').resize(size,Image.Resampling.LANCZOS).quantize(colors=96,method=Image.Quantize.FASTOCTREE,dither=Image.Dither.NONE));next_t+=100
            t=end
    frames[0].save(target,save_all=True,append_images=frames[1:],duration=100,loop=0,optimize=True)
    result=decode(target.read_bytes());assert abs(result['duration_ms']-t)<=100
    result.update(source=source.relative_to(ROOT).as_posix(),source_sha256=sha(source),transformation='Decoded existing GIF;480px width,10fps,96 colors; full duration preserved; no new actions or synthesized physics poses',input_sampled_frames=len(frames))
    return result
def prepare_media():
    old={Path(r['path']).name:r for r in read(ROOT/'reports/video_validation.json')['videos']};records=[]
    contact=Image.new('RGB',(960,5*295),'white');draw=ImageDraw.Draw(contact)
    for row,name in enumerate(NAMES):
        p=ROOT/'assets/videos'/name;assert sha(p)==old[name]['sha256'],'Original GIF changed since provenance validation'
        data=p.read_bytes();result=decode(data);assert result['duration_ms']==old[name]['duration_seconds']*1000
        result.update(path=p.relative_to(ROOT).as_posix(),original_preserved=True,source_content=old[name]['content'],source_records=old[name]['source_records'])
        assert all(r.get('joint_displacement_rad',0)>0 for r in result['source_records'])
        for forbidden in [b'C:\\Users\\',str(ROOT.parent).encode('utf-8'),b'github_pat_']:
            assert forbidden not in data,'Unexpected identity/credential byte sequence in GIF metadata'
        records.append(result)
        with Image.open(p) as im:
            for col,index in enumerate([0,im.n_frames//2,im.n_frames-1]):
                im.seek(index);frame=im.convert('RGB');frame.thumbnail((320,270));contact.paste(frame,(col*320,row*295+25))
        draw.text((5,row*295+5),name,fill='black')
    contact.save(ROOT/'tests/local_public_media_contact.png')
    for source,target in zip(NAMES[1:3],PREVIEWS):
        r=preview(ROOT/'assets/videos'/source,ROOT/'assets/videos'/target);r.update(path='assets/videos/'+target,preview=True);records.append(r)
    atomic_json(ROOT/'reports/public_media_validation.json',dict(passed=True,records=records,source='Existing verified GIFs only; Actual Training Rollout remains original stochastic sampling',public_distribution_authorized_by_user=True,weights_data_raw_rgb_authorized=False,sensitive_visual_review='contact sheet prepared; complete original telemetry frames contain simulator/task labels only',home_page_preview_bytes=sum(r['bytes'] for r in records if r.get('preview'))))
    print('MEDIA',[(r['path'],r['bytes'],r['frames'],r['duration_ms']) for r in records],flush=True)
def prepare_readme():
    with (ROOT/'reports/final_results.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    keep=['BC-DAgger','long_control_b0','long_decay_strong'];labels=['BC + DAgger','PPO without anchoring','PPO with decaying strong anchoring']
    table='| Method | Additional PPO steps / seed | Stable success | Mean final XY error |\n|---|---:|---:|---:|\n'
    for name,label in zip(keep,labels):
        r=next(r for r in rows if r['method']==name);table+=f"| {label} | {r['steps_per_seed']} | {float(r['stable_success_pct']):.2f}% | {float(r['mean_final_cm']):.4f} cm |\n"
    shared='''```powershell
python scripts/portfolio.py --help
python scripts/portfolio.py results
python scripts/portfolio.py evaluate --method ikpd --first-seed 17080 --episodes 2 --output results/expert_diagnostic
python scripts/portfolio.py pipeline --profile smoke --output results/my_smoke --ppo-method anchor
```
'''
    conditioned='''```powershell
python scripts/portfolio.py import-assets --archive-root ..
python scripts/portfolio.py init
python scripts/portfolio.py watch --policy compare --seed 7059
python scripts/portfolio.py train --output results/my_demo --steps 4096 --record-training-rollouts
python scripts/portfolio.py replay --run results/my_demo --mode actual --episode 13
python scripts/portfolio.py replay --run results/my_demo --mode compare --update 1
```
'''
    text=f'''# Stable Online Reinforcement Learning from Demonstrations for Robotic Control

Investigating policy degradation and reproducible online fine-tuning in MuJoCo Reacher with expert demonstrations, DAgger and a frozen-BC action-mean constraint.

[中文](README_CN.md) · [Protocol](docs/02_实验协议与结果核对.md) · [Source provenance](reports/source_provenance.csv)

## Robot control demonstration

![Four methods from the same starting state](assets/videos/01_same_start_comparison_preview.gif)

**Evaluation Replay:** IK+PD, BC+DAgger, unanchored PPO and decaying strong anchoring; each case uses the same initial state and target. Seeds7059/7012 are demonstrations, not final-test statistics. [Full comparison](assets/videos/01_same_start_comparison.gif) · [70-second overview](assets/videos/00_project_overview.gif).

## Pipeline

![IK+PD to BC, DAgger, PPO, anchoring and evaluation](assets/figures/pipeline.png)

## Frozen evaluation results

{table}
Three training seeds **42,123,2026**; each model uses **100 environment seeds16000–16099**. Long fine-tuning uses **100352 additional online PPO steps per seed**, after BC+DAgger. BC's zero PPO steps does not mean zero demonstration cost. Stable success requires all last five XY errors <1cm;250-step episodes,dt0.02s,distance/control reward weights1/1. These project settings differ from some benchmark defaults.

Numbers are recomputed from archived episode CSV and cross-checked against trajectories; [all five methods and seed SD](reports/final_results.csv), [per-seed results](reports/per_seed_results.csv), [model hashes](reports/model_manifest.csv). The64.67% vs58.67% result is a descriptive **+6.00 percentage-point** difference, not a claim of statistical significance. The method does not outperform the IK+PD expert.

![Final comparison with training-seed standard deviations](assets/figures/final_comparison.png)

## Core engineering

- **Control and imitation:**10-dimensional observations,2-dimensional bounded actions, IK+PD expert, actor-only BC and one-round DAgger.
- **Degradation diagnosis:** compare independent BC restarts, actor/critic learning rates, action drift, KL and frozen final checkpoints; retain failures and between-seed variation.
- **Behavior anchoring:** original PPO plus batch-and-coordinate mean MSE against a frozen BC Gaussian mean. Teacher has no gradients or optimizer membership; actor/critic Adam rates1e-5/3e-4 and global clipping0.5. Fixed and decaying schedules retain the verified implementation.
- **Reproducibility:** fixed validation/test roles, actual1024-step rollout boundaries, checkpoint SHA256, original source provenance and staged-only isolation tests. Algorithms are established methods, not claimed inventions.

## Install and quick start

Validated with an existing Windows Python3.11.17 environment. Fresh-machine installation and other operating systems have not been tested.

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
```

The following commands need **no historical weights or datasets**:

{shared}
The smoke pipeline collects500 training expert labels,250 validation labels and250 DAgger labels/online steps, runs two supervised epochs per stage and1024 PPO steps using its newly trained BC teacher. It is a plumbing check, not a new research result. Use a new output directory per run. [Reproduction costs and full-budget entry points](docs/03_完整训练复现与工程结构.md).

**Pretrained weights, diagnostic NPZ and raw training trajectories are not distributed.** For pretrained viewing or BC-initialized online recording, separately obtain authorized assets:15 ZIPs in configs/models.json plus configs/diagnostic_observations.npz with its hash in reports/asset_provenance.csv. Import below assumes that you already have the authorized original archive layout; it does not download weights. Missing assets cause an explicit path-and-SHA error. After provision:

{conditioned}
## Actual online trial and error

![Actual stochastic PPO training rollout](assets/videos/02_actual_online_learning_preview.gif)

**Actual Training Rollout:** these RGB frames were captured from the environment PPO actually stepped, with sampled actions, raw rewards, distances and policy versions. No replacement predict calls or extra environment steps generated this recording. Preselected episodes1/5/9/13 include failures and explicit gaps;1024-step updates can occur within an episode. Reducing the preview to10fps changes only display sampling, not the source trajectory. [Full recording](assets/videos/02_actual_online_learning.gif) · [BC vs final](assets/videos/03_bc_vs_final_policy.gif) · [Failure cases](assets/videos/04_failure_cases.gif) · [Frame/hash validation](reports/public_media_validation.json).

`train --render-training` opens the actual training environment; run it separately from RGB recording mode. `replay --mode compare` is deterministic **Evaluation Replay**, generated after training, and is never labeled actual sampling. [Chinese operation guide](docs/真实PPO试错过程_运行与可视化教程.md). TensorBoard: `python -m tensorboard.main --logdir results/my_demo/tensorboard --port 6008`.

## Limitations, provenance and references

Single-task simulation, three training seeds, no real-robot deployment, cross-robot validation, Sim-to-Real experiment or statistical-significance claim. Low action drift does not guarantee accurate positioning. Full100k jobs were not rerun for publication. GIFs are reviewed simulator recordings; previews are smaller derivatives and originals remain unchanged. Model weights, training datasets and large raw state/RGB archives stay excluded.

The owner authorized public distribution of independently developed project source, documents, results, figures and reviewed GIFs. Third-party library license/NOTICE text remains in third_party with [dependency records](reports/dependency_licenses.csv) and [provenance review](docs/04_来源与发布许可核对.md). No MIT/Apache license for the project's own code has been selected; public visibility does not create an unspecified open-source license grant.

References: [PPO](https://arxiv.org/abs/1707.06347), [DAgger](https://proceedings.mlr.press/v15/ross11a.html), [Stable-Baselines3](https://jmlr.org/papers/v22/20-1364.html), [MuJoCo](https://mujoco.org/), [Gymnasium Reacher](https://gymnasium.farama.org/environments/mujoco/reacher/).
'''
    (ROOT/'README.md').write_text(text,encoding='utf-8')
    cn=f'''# 基于模仿学习与策略约束的机器人在线强化学习控制

在MuJoCo Reacher中结合专家示教、BC、DAgger和固定BC动作均值约束，诊断PPO在线微调退化，并记录真实随机采样与网络更新。

[English](README.md) · [实验协议](docs/02_实验协议与结果核对.md) · [源码来源](reports/source_provenance.csv)

## 控制效果演示

![同起点四方法动态对比](assets/videos/01_same_start_comparison_preview.gif)

**Evaluation Replay：**同一初态和目标下对比IK+PD、BC+DAgger、无锚定PPO和Strong衰减PPO。7059/7012仅为演示种子，不是最终统计。[原始完整对比](assets/videos/01_same_start_comparison.gif) · [70秒综合演示](assets/videos/00_project_overview.gif)。

## 总体流程

![IK+PD到模仿学习、在线PPO及冻结评价](assets/figures/pipeline.png)

## 最终冻结结果

{table}
统一3训练种子42/123/2026，每模型100个测试初态16000～16099。PPO长微调每种子新增100352在线环境步；BC的0步仅表示无在线PPO，不表示没有示教成本。250步回合、dt.02、距离/控制奖励权重1/1，稳定成功要求最后连续5步XY距离均严格<1cm，不直接与其他环境默认协议比较。

逐回合CSV重算并与NPZ核对：[完整五组统计](reports/final_results.csv)、[种子差异](reports/per_seed_results.csv)、[模型SHA](reports/model_manifest.csv)。64.67%对58.67%的差异为描述性的+6.00个百分点，不宣称统计显著，也未超越IK+PD专家。

![三训练种子结果与标准差](assets/figures/final_comparison.png)

## 核心实现

- 控制与模仿：10维输入、2维动作，IK+PD专家、仅Actor监督BC及一次DAgger状态聚合。
- 策略退化诊断：独立BC重启、分组Actor/Critic学习率、均值偏差/KL、梯度与实际Adam步审计，保留失败和种子差异。
- BC Anchoring：真实rollout状态上的原始高斯动作均值MSE，教师冻结且不进优化器；Actor/Critic学习率1e-5/3e-4、全局裁剪.5，固定与衰减日程保留验证过的原实现。
- 工程复现：验证/最终测试分离、1024实际采样步边界、完整checkpoint哈希、CSV/NPZ校验和仅上传文件隔离测试，不把既有算法组合称为原创算法。

## 安装与快速开始

已在现有Windows Python3.11.17环境验证，未声称全新电脑独立安装或其他系统验证成功。标准安装模板：

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
```

以下不依赖历史模型、数据：

{shared}
烟雾流程500训练专家标签、250验证标签、250DAgger标签/在线步，监督阶段各2epoch，最后1024PPO步；使用本次新训练教师，不是新正式研究结果。每次用新输出目录。[完整复现入口与成本](docs/03_完整训练复现与工程结构.md)。

预训练权重、诊断NPZ和原始轨迹仍不分发。需要预训练演示/真实在线录制时，单独取得授权的15个ZIP及诊断NPZ，路径见configs/models.json，SHA见reports/asset_provenance.csv。以下导入假设已持有原归档，不会下载权重；缺失时提示路径和哈希：

{conditioned}
## 真正的PPO在线试错

![实际PPO随机训练录像](assets/videos/02_actual_online_learning_preview.gif)

**Actual Training Rollout：**RGB来自PPO实际调用的训练环境，显示随机动作、原始奖励、定位误差与策略版本；没有重新predict生成替代动作，也没有多step。预选episode1/5/9/13包含失败且明确显示录制间隔；1024步更新可能发生在回合中间。10fps预览只减少显示采样，不改变原轨迹。[完整实际录像](assets/videos/02_actual_online_learning.gif)、[BC到最终策略](assets/videos/03_bc_vs_final_policy.gif)、[失败案例](assets/videos/04_failure_cases.gif)、[动画校验](reports/public_media_validation.json)。

--render-training直接打开真实训练env；与RGB录制分开运行。compare是训练后独立确定性Evaluation Replay，不能冒充真实采样。[详细中文教程](docs/真实PPO试错过程_运行与可视化教程.md)。TensorBoard：`python -m tensorboard.main --logdir results/my_demo/tensorboard --port 6008`。

## 局限与来源

单任务仿真、仅3训练种子，没有实机部署、跨机器人或Sim-to-Real实验，没有统计显著性声明。低漂移不保证高定位精度；本次发布没有100k重训或调beta。GIF均为已核验的真实仿真录像，预览来自原GIF压缩，5份原件不变。权重、训练数据、大型原始状态/RGB继续排除。

用户已明确授权独立开发源码、文档、结果、图片和本次核验GIF公开。third_party保留适用许可证及NOTICE，[依赖声明](reports/dependency_licenses.csv)与[来源审计](docs/04_来源与发布许可核对.md)保留。未擅自为项目选择MIT/Apache等开源许可证，Public可见性不等同于另行授予未确定的开源许可。

参考：[PPO](https://arxiv.org/abs/1707.06347)、[DAgger](https://proceedings.mlr.press/v15/ross11a.html)、[SB3](https://jmlr.org/papers/v22/20-1364.html)、[MuJoCo](https://mujoco.org/)、[Gymnasium Reacher](https://gymnasium.farama.org/environments/mujoco/reacher/)。
'''
    (ROOT/'README_CN.md').write_text(cn,encoding='utf-8')
    p=ROOT/'.gitignore';s=p.read_text(encoding='utf-8')
    for name in NAMES+PREVIEWS:
        rule='!assets/videos/'+name
        if rule not in s:s+='\n'+rule
    # Final receipt records its own latest HEAD and stays local, avoiding a self-SHA cycle.
    for rule in ['/docs/GitHub公开发布完成报告.md','/reports/public_release_runtime.json']:
        if rule not in s:s+='\n'+rule
    p.write_text(s.rstrip()+'\n',encoding='utf-8')
    p=ROOT/'docs/04_来源与发布许可核对.md';s=p.read_text(encoding='utf-8')
    if '## 正式公开授权（替代此前私有阶段限制）' not in s:s+='''

## 正式公开授权（替代此前私有阶段限制）

用户在正式公开任务中明确确认：src/verified中的项目源码为用户与Codex独立开发或允许分发的部分，授权将源码、README、实验结果、图片与5份核验演示GIF公开，并明确授权将目标仓库由Private改为Public。此前“仅Private/视频待确认”的段落为历史阶段记录，不再限制本次已明确授权的资源。19个迁移源码来源/哈希继续保留，third_party库许可证和NOTICE不改变；没有据此虚构项目级MIT/Apache授权。

模型权重、专家训练/诊断数据、未发布的原始轨迹/RGB和视频ZIP不在本次公开范围；保持Git排除。预览只对既有真实GIF进行显示采样/尺寸压缩，5份原GIF不覆盖。公开前核查所有可达提交和当前文件、真实动画帧与元数据；结果见tests/public_release_audit.json与reports/public_media_validation.json。
'''
    p.write_text(s,encoding='utf-8')
    (ROOT/'assets/videos/README.md').write_text('''# Verified simulator animations

All five original GIFs and two small previews are explicitly authorized for public distribution. Original files are preserved byte-for-byte. Preview GIFs are derived only by resizing, temporal display sampling and palette quantization; they do not resimulate a robot or sample new actions.

- [Overview,70s](00_project_overview.gif)
- [Four-method same-start comparison,10s](01_same_start_comparison.gif) / [small preview](01_same_start_comparison_preview.gif): Evaluation Replay
- [Real online experience,20s](02_actual_online_learning.gif) / [small preview](02_actual_online_learning_preview.gif): Actual Training Rollout
- [BC vs final policy,10s](03_bc_vs_final_policy.gif): Evaluation Replay
- [Failure cases,15s](04_failure_cases.gif): actual episode13 plus labeled difficult evaluation replay

Raw model weights, datasets, state/RGB trajectories and any video ZIP remain excluded. See reports/public_media_validation.json for resolution, decoded frames, durations, hashes, compression provenance and original source models.
''',encoding='utf-8')
def audit():
    paths=[p for p in git('ls-files','-z').decode().split('\0') if p];findings=[];objects={};commits=git('rev-list','--all').decode().splitlines()
    for commit in commits:
        raw=git('ls-tree','-r','-z',commit).decode()
        for row in raw.split('\0'):
            if row:
                meta,name=row.split('\t',1);objects[meta.split()[2]]=name
    patterns={
        'github_token':rb'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b',
        'api_key':rb'\bsk-[A-Za-z0-9_-]{32,}\b','aws_key':rb'\bAKIA[A-Z0-9]{16}\b',
        'private_key':rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
        'machine_identity':rb'(?i)(?:[A-Z]:[/\\]+Users[/\\]+HUST|D:[/\\]+mujoco_learning|D:[/\\]+miniconda)',
    }
    allowed_gifs={'assets/videos/'+n for n in NAMES+PREVIEWS}
    def scan(name,data,where):
        if Path(name).suffix.lower() in ['.zip','.pt','.npz','.pkl','.pem','.key','.mp4'] or name.startswith(('results/','runtime/','tests/local_','.venv/')):findings.append(dict(path=name,kind='excluded_resource',where=where))
        if Path(name).suffix.lower()=='.gif' and name not in allowed_gifs:findings.append(dict(path=name,kind='unreviewed_gif',where=where))
        if len(data)>100*1024*1024:findings.append(dict(path=name,kind='git_file_limit',where=where))
        if Path(name).suffix.lower() in ['.py','.json','.csv','.md','.txt']:
            for kind,pat in patterns.items():
                for m in re.finditer(pat,data):findings.append(dict(path=name,kind=kind,line=data.count(b'\n',0,m.start())+1,where=where))
    for obj,name in objects.items():scan(name,git('cat-file','blob',obj),'reachable_history')
    for name in paths:scan(name,(ROOT/name).read_bytes(),'working_candidate')
    provenance=list(csv.DictReader((ROOT/'reports/source_provenance.csv').open(encoding='utf-8-sig')))
    for row in provenance:
        assert sha(ROOT/row['destination'])==row['destination_sha256'],'Verified RL source changed'
    root_commit=commits[-1]
    third_party=[p for p in paths if p.startswith('third_party/')]
    # Compare the actual staged bytes: working-tree line endings can differ on Windows.
    for p in third_party:assert git('show',':'+p)==git('show',root_commit+':'+p),'Staged third-party notice modified'
    emails=git('log','--all','--format=%ae').decode().splitlines()
    assert all(email.endswith('@users.noreply.github.com') for email in emails),'Review personal commit email before publication'
    readme_links=[]
    for name in ['README.md','README_CN.md']:
        text=(ROOT/name).read_text(encoding='utf-8')
        for target in re.findall(r'!?\[[^\]]+\]\(([^)]+)\)',text):
            if re.match('^[a-z]+:',target,re.I):continue
            target=urllib.parse.unquote(target.split('#')[0]);assert target in paths,(name,target)
            readme_links.append(dict(readme=name,target=target))
        for number in ['58.67%','2.3029','32.33%','2.7527','64.67%','1.9986']:assert number in text
    result=dict(passed=not findings,commits_audited=commits,unique_history_blobs=len(objects),current_files=len(paths),findings=findings,verified_source_files_unchanged=len(provenance),third_party_notice_files_preserved=len(third_party),readme_links=readme_links,reviewed_public_gifs=sorted(allowed_gifs),user_public_authorization=True,project_license='not independently selected',history_author_email='GitHub noreply only',sensitive_content_review='heuristic history/content scanning plus visual GIF contact-sheet review; not a universal credential-detection guarantee',excluded=['weights','expert/diagnostic datasets','raw rollout/RGB archives','video ZIP'])
    atomic_json(ROOT/'tests/public_release_audit.json',result);assert not findings,findings
    print('AUDIT PASSED',len(commits),'commits',len(objects),'history blobs',len(paths),'current files',flush=True)
def publish():
    review=read(ROOT/'tests/public_release_audit.json');assert review['passed']
    token=credential();user=api(token,'/user');assert user['login'].lower()==OWNER.lower()
    repo=api(token,'/repos/'+FULL);assert repo['full_name'].lower()==FULL.lower()
    head=git('rev-parse','HEAD').decode().strip();remote=api(token,f'/repos/{FULL}/git/ref/heads/main')['object']['sha'];assert head==remote,'Push reviewed content before visibility change'
    # Compare every remote blob to the reviewed local commit before changing visibility.
    tree=api(token,f'/repos/{FULL}/git/trees/{head}?recursive=1');assert not tree['truncated'];remote_files={r['path']:r['sha'] for r in tree['tree'] if r['type']=='blob'}
    local={}
    for row in git('ls-tree','-r','-z','HEAD').decode().split('\0'):
        if row:
            meta,name=row.split('\t',1);local[name]=meta.split()[2]
    assert remote_files==local
    assert all('assets/videos/'+n in remote_files for n in NAMES+PREVIEWS)
    assert not any(Path(p).suffix.lower() in ['.zip','.pt','.npz'] for p in remote_files)
    before=repo['private']
    if before:repo=api(token,'/repos/'+FULL,'PATCH',dict(private=False))
    assert repo['private'] is False and repo['visibility']=='public'
    state=dict(url=repo['html_url'],public=True,previously_private=before,owner=user['login'],head=head,default_branch=repo['default_branch'],remote_files=len(remote_files),visibility_changed_under_explicit_user_authorization=True,anonymous_verified=False)
    atomic_json(ROOT/'reports/public_release_runtime.json',state);print(json.dumps(state),flush=True)
class Page(HTMLParser):
    def __init__(self):super().__init__();self.images=[];self.article=False
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='article' and 'markdown-body' in a.get('class',''):self.article=True
        if tag=='img':self.images.append(a.get('src',''))
def anonymous_get(url):
    request=urllib.request.Request(url,headers={'User-Agent':'reacher-public-anonymous-verifier','Accept':'*/*'})
    assert not request.has_header('Authorization') and not request.has_header('Cookie')
    with urllib.request.urlopen(request,timeout=90) as response:
        assert response.status==200;return response.read(),dict(response.headers)
def verify():
    state=read(ROOT/'reports/public_release_runtime.json');head=git('rev-parse','HEAD').decode().strip();assert head==state['head']
    repo=json.loads(anonymous_get('https://api.github.com/repos/'+FULL)[0]);assert repo['private'] is False and repo['default_branch']=='main'
    remote=json.loads(anonymous_get(f'https://api.github.com/repos/{FULL}/git/ref/heads/main')[0]);assert remote['object']['sha']==head
    tree=json.loads(anonymous_get(f'https://api.github.com/repos/{FULL}/git/trees/{head}?recursive=1')[0]);assert not tree['truncated']
    files={r['path']:r['sha'] for r in tree['tree'] if r['type']=='blob'}
    local={}
    for row in git('ls-tree','-r','-z','HEAD').decode().split('\0'):
        if row:
            meta,name=row.split('\t',1);local[name]=meta.split()[2]
    assert files==local
    html=anonymous_get(URL)[0].decode('utf-8');page=Page();page.feed(html)
    assert page.article,'GitHub did not serve a rendered README article'
    for name in PREVIEWS:assert any(name in s for s in page.images),'README preview not rendered'
    media=[]
    for name in NAMES+PREVIEWS:
        path='assets/videos/'+name;data,headers=anonymous_get(f'https://raw.githubusercontent.com/{FULL}/{head}/{path}')
        assert hashlib.sha256(data).hexdigest()==sha(ROOT/path);v=decode(data);v.update(path=path,anonymous_http_status=200);media.append(v)
        print('ANONYMOUS GIF OK',name,v['frames'],v['duration_ms'],flush=True)
    targets=set()
    for name in ['README.md','README_CN.md']:
        text=(ROOT/name).read_text(encoding='utf-8')
        for target in re.findall(r'!?\[[^\]]+\]\(([^)]+)\)',text):
            if not re.match('^[a-z]+:',target,re.I):targets.add(urllib.parse.unquote(target.split('#')[0]))
    # Every link is in the anonymous remote tree; additionally fetch each displayed PNG.
    for path in targets:assert path in files,path
    for path in sorted(p for p in targets if p.endswith('.png')):
        data,_=anonymous_get(f'https://raw.githubusercontent.com/{FULL}/{head}/{path}');assert hashlib.sha256(data).hexdigest()==sha(ROOT/path)
        with Image.open(io.BytesIO(data)) as im:im.load()
    code,_=anonymous_get(f'https://raw.githubusercontent.com/{FULL}/{head}/scripts/portfolio.py');assert hashlib.sha256(code).hexdigest()==sha(ROOT/'scripts/portfolio.py')
    for name in ['README.md','README_CN.md']:
        text=anonymous_get(f'https://raw.githubusercontent.com/{FULL}/{head}/{name}')[0].decode()
        for value in ['58.67%','2.3029','32.33%','2.7527','64.67%','1.9986']:assert value in text
    assert not any(Path(p).suffix.lower() in ['.zip','.pt','.npz','.pem','.key'] for p in files)
    state.update(anonymous_verified=True,anonymous_api_public=True,remote_tree_matches_local=True,rendered_readme_article=True,readme_preview_images_rendered=True,readme_relative_links_in_remote_tree=len(targets),anonymous_media=media,code_verified=True,models_datasets_absent=True,authorization_header_used=False,cookies_used=False)
    atomic_json(ROOT/'reports/public_release_runtime.json',state);report(state);print('PUBLIC ANONYMOUS VERIFICATION COMPLETE',head,len(files),flush=True)
def report(state):
    records=read(ROOT/'reports/public_media_validation.json')['records']
    table='| GIF | Resolution | Frames | Seconds | Bytes |\n|---|---|---:|---:|---:|\n'
    for r in records:table+=f"| {Path(r['path']).name} | {r['width']}×{r['height']} | {r['frames']} | {r['duration_ms']/1000:g} | {r['bytes']} |\n"
    body=f'''# GitHub公开发布完成报告

仓库：[reacher-rl-from-demonstrations]({URL})。实际可见性：**Public**。默认分支：main。最新提交：`{state['head']}`。远程文件数：{state['remote_files']}。公开与匿名验收：{'已完成' if state.get('anonymous_verified') else '尚待完成'}。

用户已明确授权Public转换，以及独立开发源码、README、结果、图片、演示GIF的公开。既有19个迁移核心源码未修改，第三方库许可证与NOTICE逐文件保留；未自行选择项目MIT/Apache许可证。公开可见性不替代未另行确定的项目级开源授权。

## 安全与历史审计

全部可达提交与当前候选文件已做Token/常见API密钥/私钥/本机敏感路径及禁止资源扫描，未发现命中；提交者邮箱为GitHub noreply。源码来源清单与原SHA一致，third_party与首个提交的原文一致。视觉检查GIF接触表，画面为仿真机械臂及任务/动作/奖励标注，无发现个人信息；GIF元数据与注释长度也已核查。扫描为启发式，不声称识别所有凭据格式。检查记录：tests/public_release_audit.json。

模型ZIP/PT、专家和诊断NPZ、大型原始状态/RGB轨迹及视频ZIP仍被忽略，历史和当前公开树均没有这些文件。五个原GIF逐字节保留；首页仅嵌入两个较小480px/10fps/96色预览，时长保留、不产生新动作或仿真画面。

{table}

## 真实录像与README

首页按任务要求排列：标题/介绍、同起点四方法动画、流程图、最终表、核心实现、安装/快速运行、实际随机PPO动画、局限/参考。双语内容与CSV核对。3训练种子，每模型100冻结环境种子；长PPO新增100352步，BC监督/DAgger成本另计。结果58.67%/2.3029cm、32.33%/2.7527cm、64.67%/1.9986cm不变。

四方法是Evaluation Replay；真实试错GIF为PPO实际训练环境捕获RGB，保持随机样本/奖励/版本及录制间隔标注，没有predict重新生成或额外step。预览仅改变显示帧率。失败仍保留。原素材/模型SHA见reports/video_validation.json与reports/public_media_validation.json。

## 匿名远程验收

验证请求未发送Authorization或Cookie：公开GitHub API返回Public，main的SHA等于本地；全部远程Git blob与本地提交一致；GitHub网页包含实际渲染README文章与两张动态预览；7个GIF匿名HTTP200、内容SHA与本地相同、逐帧解码确认多帧/时长/变化；README所有相对资源存在于匿名远程树，展示PNG额外匿名下载解码，无404；主入口源码和中英文结果数字额外核对。记录保存在本地reports/public_release_runtime.json（含最终HEAD，Git忽略，以免自引用SHA循环）。

## 依赖与剩余限制

依赖固定版本见requirements.txt、requirements-lock.txt；包许可证/NOTICE见third_party和reports/dependency_licenses.csv。无历史权重可运行help、指标、专家诊断及从头小流程，之前已在仅拟上传文件的隔离目录实际验证；预训练观看和BC初始化在线录制仍需另行取得模型及诊断数据。本次仅公开发布/媒体优化，没有修改RL算法、重调beta或开展100k训练；没有将旧小流程结果说成新训练。

新机器安装与跨平台尚未验证，环境记录为Windows Python3.11.17。单任务仿真、3训练种子，没有原创算法、统计显著、超越IK+PD或实机/Sim-to-Real声明。未提供MP4；GIF原件较大，首页采用小预览，完整原件通过链接下载。权重和训练数据的公开仍未授权。完整项目级开源许可证待另行选择，不伪称已授予。

本报告为本地最终核验回执，Git忽略，以准确记录包含发布提交本身的最新SHA；公开仓库已包含媒体校验和历史/来源审计文件。
'''
    (ROOT/'docs/GitHub公开发布完成报告.md').write_text(body,encoding='utf-8')
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','audit','publish','verify']);a=p.parse_args()
    if a.mode=='prepare':prepare_media();prepare_readme()
    elif a.mode=='audit':audit()
    elif a.mode=='publish':publish()
    else:verify()
