"""Audit a private-repository candidate and test exactly its staged file set."""
from pathlib import Path
import sys,json,re,csv,subprocess,shutil,time,urllib.parse,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.runtime import read,sha,atomic_json
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)
def staged():return [p for p in git('ls-files','-z').decode('utf-8').split('\0') if p]
def prepare_docs():
    append='''

## GitHub私有托管授权更新

本次上传任务中，用户已明确确认有权把本项目源码（包含src/verified的19个来源文件）、文档、脱敏指标和7张PNG托管到本人GitHub私有仓库。该确认覆盖本次Private远程存储，不构成Public发布许可或开放源码许可证。源码继续保留source_provenance.csv，第三方许可证/NOTICE保持原文。

模型ZIP/PT、专家示教与诊断NPZ、训练轨迹/RGB以及5个GIF不在本次授权范围内，继续保留本地，不上传。GitHub仓库不得自行改为Public。将来如需公开，须再次确认适用的权利和许可证。
'''
    p=ROOT/'docs/04_来源与发布许可核对.md';text=p.read_text(encoding='utf-8')
    if '## GitHub私有托管授权更新' not in text:p.write_text(text+append,encoding='utf-8')
    for name in ['README.md','README_CN.md']:
        p=ROOT/name;text=p.read_text(encoding='utf-8')
        if name=='README.md':
            a=text.index('## Quick start');b=text.index('## Reproduction and scope')
            text=text[:a]+'''## Quick start

Tested on Windows with an existing Python 3.11.17 / MuJoCo environment. Exact installed versions are in requirements.txt and requirements-lock.txt. Fresh-machine package installation and other operating systems have not been tested. Standard installation template: `python -m venv .venv`, activate it, then `python -m pip install -r requirements.txt`.

**No pretrained assets required:** the repository can import its modules, display archived metrics, run expert-only diagnostics, and train a small expert→BC→DAgger→PPO chain from scratch. The default smoke run uses its own newly trained BC teacher and training observations; it does not read historical weights.

```powershell
python scripts/portfolio.py --help
python scripts/portfolio.py results
python scripts/portfolio.py evaluate --method ikpd --first-seed 17080 --episodes 2 --output results/expert_diagnostic
python scripts/portfolio.py pipeline --profile smoke --output results/my_smoke --ppo-method anchor
```

These small diagnostics are engineering checks, not new final-test results. Smoke costs500 expert training labels,250 expert validation labels,250 DAgger labels/online steps, two supervised epochs per stage and1024 PPO steps. Use a new output directory per run.

**Pretrained demonstration, evaluation and online visualization require separately authorized assets.** No model weights, expert/diagnostic NPZ, historical rollout/RGB or GIF is included in this Git repository. Required files:15 ZIP models listed in configs/models.json, plus configs/diagnostic_observations.npz with the SHA256 in reports/asset_provenance.csv. The repository is not a self-contained pretrained distribution. On the authorized original archive layout, use `python scripts/portfolio.py import-assets --archive-root ..`; otherwise provision the same authorized files at the listed relative paths. Missing models produce a path-and-SHA error; never substitute an unverified checkpoint.

After authorized asset provisioning:

```powershell
python scripts/portfolio.py init
python scripts/portfolio.py watch --policy bc
python scripts/portfolio.py watch --policy best
python scripts/portfolio.py watch --policy compare --seed 7012
python scripts/portfolio.py evaluate --method long_decay_strong --training-seed 42 --episodes 100
python scripts/portfolio.py train --output results/my_demo --steps 4096 --record-training-rollouts
python scripts/portfolio.py replay --run results/my_demo --mode actual --episode 13
python scripts/portfolio.py replay --run results/my_demo --mode compare --update 1
python scripts/portfolio.py record-video --policy best --seed 7059 --output results/my_video
python -m tensorboard.main --logdir results/my_demo/tensorboard --port 6008
```

Native GUI tests use bounded frame/step limits for automatic exit. `train --render-training` displays the actual PPO environment; use it separately from RGB recording. Actual replay reads captured RGB/actions without calling predict or env.step. The compare mode is explicitly **Evaluation Replay**, generated after training, never a replacement for **Actual Training Rollout**. See the [Chinese operation guide](docs/真实PPO试错过程_运行与可视化教程.md).

The five validated GIFs remain local-only because their remote-hosting permission is unconfirmed. GitHub displays the authorized PNG control screenshot, pipeline and result plot above. [Media inventory](assets/videos/README.md) and [validation metadata](reports/video_validation.json) record filenames, source/encoded frames, durations and model hashes without linking to absent binaries. No MP4 was produced. Demo seeds7059/7012 are not final statistics.

'''+text[b:]
            text=text[:text.index('## Publication status')]+'''## Publication status

The owner explicitly authorized private GitHub hosting of source (including src/verified), documents, anonymized metrics and seven PNG figures for this task. **Public release is not authorized.** No open-source license has been invented. Model/data/rollout/video assets remain local and excluded from Git. [Publication review](docs/04_来源与发布许可核对.md), [dependency licenses](reports/dependency_licenses.csv) and verbatim third_party notices preserve provenance. Repository creation, authentication and push verification status are recorded separately in the local upload report; local tests alone do not prove an upload succeeded.
'''
        else:
            a=text.index('快速运行使用' if '快速运行使用' in text else '使用已有Python3.11.17环境运行');b=text.index('已有算法的工程实现')
            text=text[:a]+'''使用已有Python3.11.17环境运行，精确版本见requirements.txt与requirements-lock.txt。安装模板为新建并激活venv后执行 `python -m pip install -r requirements.txt`；没有把现有环境测试说成全新电脑独立安装成功。

**无需历史权重也可运行**：入口帮助、归档指标、专家独立诊断、从头采样→BC→DAgger→PPO小流程。默认Anchoring烟雾流程使用本次新训练的BC教师和训练观测，不依赖历史模型或NPZ。

```powershell
python scripts/portfolio.py --help
python scripts/portfolio.py results
python scripts/portfolio.py evaluate --method ikpd --first-seed 17080 --episodes 2 --output results/expert_diagnostic
python scripts/portfolio.py pipeline --profile smoke --output results/my_smoke --ppo-method anchor
```

每次烟雾测试500训练专家标签、250监督验证标签、250DAgger标签/在线步、各阶段2epoch、1024PPO步。仅为工程诊断，不是新的最终实验。

**预训练效果与真实试错演示需要另行取得许可的资产**：Git中没有15个模型ZIP、诊断NPZ、历史RGB/轨迹及GIF。具体相对路径与SHA见configs/models.json、reports/model_manifest.csv、reports/asset_provenance.csv；公开或私有克隆均不能声称预训练资产自包含。持有授权的历史归档可运行 `python scripts/portfolio.py import-assets --archive-root ..`，或按清单放置已授权文件。缺失时明确提示文件路径与SHA，不替换成其他模型。

取得并导入资产后：

```powershell
python scripts/portfolio.py init
python scripts/portfolio.py watch --policy bc
python scripts/portfolio.py watch --policy best
python scripts/portfolio.py watch --policy compare --seed 7012
python scripts/portfolio.py evaluate --method long_decay_strong --training-seed 42 --episodes 100
python scripts/portfolio.py train --output results/my_demo --steps 4096 --record-training-rollouts
python scripts/portfolio.py replay --run results/my_demo --mode actual --episode 13
python scripts/portfolio.py replay --run results/my_demo --mode compare --update 1
python scripts/portfolio.py record-video --policy best --seed 7059 --output results/my_video
python -m tensorboard.main --logdir results/my_demo/tensorboard --port 6008
```

训练使用新输出目录；真实训练窗口用--render-training，RGB记录用--record-training-rollouts，两者分开运行。actual只读取实际捕获RGB；compare明确标为Evaluation Replay。完整说明见中文教程。已有online_demo目录不随Git上传，应自行训练生成my_demo再回放。

五个已验证GIF继续留在本地，未取得远程托管授权，不放无效下载链接。仓库首页保留已获私有托管授权的真实控制PNG、流程图和实验图；[媒体目录说明](assets/videos/README.md)、[帧验收元数据](reports/video_validation.json)保留文件名和校验。当前无MP4编码后端。7059/7012只是演示案例。

'''+text[b:]
            last=text.index('仅创建本地Git' if '仅创建本地Git' in text else '用户已明确授权本次源码')
            text=text[:last]+'''用户已明确授权本次源码（含src/verified）、文档、脱敏指标和7张PNG托管到本人GitHub私有仓库；未授权Public发布，未擅自添加开源许可证。模型、数据、轨迹和GIF继续排除。实际远程创建/推送/验收状态以本地上传报告为准，不能把本地测试当作远程上传成功。[许可核对](docs/04_来源与发布许可核对.md)与third_party保留来源声明。
'''
        p.write_text(text,encoding='utf-8')
    # Private local delivery report intentionally holds the final HEAD without a self-SHA cycle.
    p=ROOT/'.gitignore';s=p.read_text(encoding='utf-8')
    for line in ['/docs/GitHub上传与发布检查报告.md','/reports/github_upload_runtime.json']:
        if line not in s:s+='\n'+line+'\n'
    p.write_text(s,encoding='utf-8')
    # Turn links to intentionally withheld local assets into plain filenames in every document.
    candidates=set(staged())
    for p in ROOT.rglob('*.md'):
        if any(x in p.relative_to(ROOT).parts for x in ['results','runtime','third_party','.git']) or 'local_' in p.as_posix():continue
        text=p.read_text(encoding='utf-8')
        def link(m):
            target=m.group(2).split('#')[0].strip('<>')
            if not target or re.match(r'^[a-z]+:',target,re.I):return m.group(0)
            dest=(p.parent/urllib.parse.unquote(target)).resolve()
            try:rel=dest.relative_to(ROOT).as_posix()
            except ValueError:return m.group(0)
            if rel not in candidates:return f'{m.group(1)} (`{target}`, local-only / not distributed)'
            return m.group(0)
        text=re.sub(r'(?<!!)\[([^\]]+)\]\(([^)]+)\)',link,text)
        p.write_text(text,encoding='utf-8')
def audit():
    paths=staged();findings=[];images=[];maxbytes=0
    patterns={
        'github_token':r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b',
        'aws_key':r'\bAKIA[A-Z0-9]{16}\b',
        'private_key':r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
        'api_key':r'\bsk-[A-Za-z0-9_-]{32,}\b',
        'local_identity_path':r'(?i)(?:[A-Z]:[/\\]+Users[/\\]+HUST|D:[/\\]+mujoco_learning|D:[/\\]+miniconda)',
    }
    for name in paths:
        p=ROOT/name;size=p.stat().st_size;maxbytes=max(maxbytes,size)
        if p.suffix.lower() in ['.zip','.pt','.npz','.gif','.mp4','.pkl','.pem','.key'] or name.startswith(('results/','runtime/','.venv/','tests/local_')):findings.append(dict(file=name,kind='excluded_binary_or_local_data'))
        if size>5*1024*1024:findings.append(dict(file=name,kind='review_size_limit_5MiB'))
        if p.suffix.lower() in ['.py','.md','.txt','.csv','.json']:
            text=p.read_text(encoding='utf-8-sig',errors='replace')
            for kind,pattern in patterns.items():
                for m in re.finditer(pattern,text):findings.append(dict(file=name,kind=kind,line=text.count('\n',0,m.start())+1))
        if p.suffix.lower()=='.png':
            from PIL import Image
            with Image.open(p) as im:im.load();assert not im.getexif();images.append(dict(path=name,bytes=size,sha256=sha(p),private_hosting_authorized=True))
    links=[]
    for name in ['README.md','README_CN.md']:
        text=(ROOT/name).read_text(encoding='utf-8')
        for target in re.findall(r'!?\[[^\]]+\]\(([^)]+)\)',text):
            target=target.split('#')[0]
            if re.match('^[a-z]+:',target,re.I):continue
            target=urllib.parse.unquote(target);ok=target in paths;links.append(dict(readme=name,target=target,staged=ok))
            if not ok:findings.append(dict(file=name,kind='broken_readme_link',target=target))
    result=dict(passed=not findings,candidate_files=len(paths),max_file_bytes=maxbytes,findings=findings,authorized_pngs=images,readme_links=links,
        secret_scan='heuristic pattern and staged-file audit; not a guarantee against every credential format',
        source_hosting_authorization='User explicitly authorized source incl19 src/verified files, docs, anonymized metrics and7 PNGs in own Private GitHub; not Public',
        excluded=['all model ZIP/PT','expert and diagnostic NPZ','results and raw trajectories/RGB','five GIFs and all videos','local environments/caches','local upload receipt'],
        third_party_notices_preserved=True)
    atomic_json(ROOT/'tests/github_preupload_audit.json',result);assert result['passed'],findings
    return result
def isolated(snapshot_name='local_upload_candidate'):
    assert re.fullmatch(r'local_[a-z0-9_]+',snapshot_name),'Snapshot name must be a local test directory basename'
    dest=ROOT/'tests'/snapshot_name;assert not dest.exists(),'Use a new isolated snapshot, never overlay missing asset tests'
    dest.mkdir();paths=staged()
    for name in paths:
        p=dest/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(git('show',':'+name))
    assert not list(dest.rglob('*.npz')) and not list(dest.rglob('*.zip')) and not list(dest.rglob('*.gif'))
    cases=[]
    def run(args,expected=0,contains=None):
        started=time.perf_counter();p=subprocess.run([sys.executable,'-I']+args,cwd=dest,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=120)
        out=p.stdout+'\n'+p.stderr
        if contains:assert contains in out,out[-1600:]
        assert (p.returncode==0)==(expected==0),out[-1600:]
        cases.append(dict(arguments=args,exit_code=p.returncode,expected_success=expected==0,elapsed_seconds=time.perf_counter()-started,expected_text=contains))
    run(['scripts/portfolio.py','--help'])
    run(['scripts/portfolio.py','results'],contains='64.66666666666667')
    run(['scripts/portfolio.py','init'],expected=1,contains='Provision authorized weights first: models/BC-DAgger/seed42/bc_best.zip; expected SHA256=')
    run(['-c',"import sys;sys.path.insert(0,'.');import src.envs,src.expert,src.imitation,src.rl,src.evaluation,src.visualization;from src.runtime import registry;assert len(registry())==15;print('IMPORTS AND MODEL REGISTRY OK')"],contains='IMPORTS AND MODEL REGISTRY OK')
    run(['scripts/portfolio.py','evaluate','--method','ikpd','--first-seed','17080','--episodes','2','--output','results/expert_diagnostic'],contains='EVALUATED repeat_verification')
    run(['scripts/portfolio.py','pipeline','--profile','smoke','--output','results/from_scratch_smoke','--ppo-method','anchor'],contains='PIPELINE COMPLETE')
    done=read(dest/'results/from_scratch_smoke/complete.json');assert done['ppo_environment_steps']==1024
    versions=read(ROOT/'reports/runtime_environment.json')
    result=dict(passed=True,exact_staged_files=len(paths),no_historical_siblings_or_pretrained_assets=True,isolated_flag='python -I',tests=cases,smoke_ppo_steps=1024,smoke_expert_training_samples=500,smoke_expert_validation_samples=250,smoke_dagger_labels=250,not_new_research=True,installed_python=versions['python'],fresh_machine_install_tested=False)
    atomic_json(ROOT/'tests/github_isolated_candidate_validation.json',result);return result
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--isolate',action='store_true');p.add_argument('--snapshot-name',default='local_upload_candidate');a=p.parse_args()
    if a.isolate:print(json.dumps(isolated(a.snapshot_name),ensure_ascii=False))
    else:prepare_docs();subprocess.run(['git','add','.'],cwd=ROOT,check=True);print(json.dumps(audit(),ensure_ascii=False))
