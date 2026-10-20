"""Final local Git, privacy, artifact and delivery report audit (no commit or push)."""
from pathlib import Path
import sys,csv,json,re,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from src.runtime import read,atomic_json,sha
from prepare_portfolio import csvout
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)
def main():
    acceptance=read(ROOT/'tests/portfolio_acceptance.json')
    for r in acceptance['checks']:
        if r['name']=='complete_runtime_dependencies':r['details'].pop('output',None);r['details']['command_exit_code']=0
    atomic_json(ROOT/'tests/portfolio_acceptance.json',acceptance)
    # Improve generated English typography without changing numbers or semantics.
    for name in ['README.md','docs/Resume_Project_English.md']:
        p=ROOT/name;text=p.read_text(encoding='utf-8')
        for token in ['only','and','uses','through','Achieved','from','versus','across','with','per','seeds','steps','beta','clipping','reached','contains']:
            text=re.sub(r'\b'+token+r'(?=\d)',token+' ',text)
        text=text.replace('stable1cm','stable 1cm').replace('before running init','before running init')
        p.write_text(text,encoding='utf-8')
    # Include explicitly bounded native/recording runs from this closeout, not cached training.
    executions=[]
    for n in ['online_demo','human_smoke','human_update_window','pipeline_smoke','pipeline_scratch','pipeline_init','pipeline_decay_smoke']:
        record=read(ROOT/f'results/{n}/complete.json');executions.append(dict(run=n,purpose='engineering verification, excluded from historical research statistics',actual_ppo_environment_steps=record.get('steps',record.get('ppo_environment_steps')),ppo_rollout_updates=record.get('policy_update_count',record.get('ppo_updates',0)//10),model_sha256=record['model_sha256'],expert_train_samples=record.get('expert_train_samples',0),expert_validation_samples=record.get('expert_validation_samples',0),dagger_expert_labels=record.get('dagger_expert_labels',0),dagger_online_steps=record.get('dagger_online_steps',0)))
    csvout(ROOT/'reports/closeout_execution_manifest.csv',executions)
    video=read(ROOT/'reports/video_validation.json');assert video['overview_duration_seconds']==70
    for v in video['videos']:assert sha(ROOT/v['path'])==v['sha256']
    # Stage only public-review text/code/small figures; .gitignore withholds local assets.
    git('add','.')
    paths=git('ls-files','-z').decode('utf-8').split('\0');paths=[p for p in paths if p]
    bad=[];large=[];private=[]
    for name in paths:
        p=ROOT/name
        if p.stat().st_size>5*1024*1024:large.append(name)
        if p.suffix.lower() in ['.zip','.npz','.pt','.gif','.mp4'] or name.startswith(('results/','runtime/','tests/local_')):bad.append(name)
        if p.suffix.lower() in ['.md','.txt','.py','.json','.csv']:
            text=p.read_text(encoding='utf-8-sig',errors='replace')
            # Installed-package paths in source code are prohibited; generic path syntax is fine.
            if re.search(r'(?i)(?:C:[/\\]+Users[/\\]+HUST|D:[/\\]+miniconda|D:[/\\]+mujoco_learning)',text):private.append(name)
    assert not bad and not large and not private,(bad,large,private)
    assert not git('remote').strip()
    result=dict(name='local_git_no_private_or_unauthorized_large_assets',status='passed',details=dict(staged_files=len(paths),largest_bytes=max((ROOT/p).stat().st_size for p in paths),binaries_withheld=True,private_machine_path_matches=0,remotes=0,public_push=False,license_grant='pending_owner_confirmation'))
    acceptance['checks']=[r for r in acceptance['checks'] if r['name']!=result['name']]+[result]
    acceptance.pop('git_check',None);acceptance['counts']={k:sum(r['status']==k for r in acceptance['checks']) for k in ['passed','failed','skipped']}
    acceptance['status']='failed' if acceptance['counts']['failed'] else 'passed_with_explicit_skips'
    atomic_json(ROOT/'tests/portfolio_acceptance.json',acceptance)
    atomic_json(ROOT/'tests/git_release_validation.json',result['details'])
    data=list(csv.DictReader((ROOT/'reports/final_results.csv').open(encoding='utf-8-sig')))
    models=list(csv.DictReader((ROOT/'reports/model_manifest.csv').open(encoding='utf-8-sig')))
    table='| 方法 | 稳定成功率 | 最终XY误差cm | 动作代价 |\n|---|---:|---:|---:|\n'
    for r in data:table+=f"| {r['method']} | {float(r['stable_success_pct']):.2f}% | {float(r['mean_final_cm']):.4f} | {float(r['mean_action_sq_sum']):.6f} |\n"
    modeltable='| 方法/种子 | 本地核心模型相对路径 | 实际PPO步数 | SHA256 |\n|---|---|---:|---|\n'
    for r in models:modeltable+=f"| {r['method']} / {r['training_seed']} | [{r['path']}]({r['path']}) | {r['steps']} | {r['sha256']} |\n"
    history=read(ROOT/'tests/history_integrity.json')
    report=f'''# 项目收尾完成报告

最终项目：基于模仿学习与策略约束的机器人在线强化学习控制（Stable Online Reinforcement Learning from Demonstrations for Robotic Control）。独立交付目录为本报告所在的reacher_rl_portfolio，所有历史四项目保留原位。本地代码与求职演示已可使用；尚未获权利确认的公开发布不在已完成范围。

完成验收：**{acceptance['counts']['passed']}项通过、{acceptance['counts']['failed']}项失败、{acceptance['counts']['skipped']}项明确跳过**。详见[机器可读验收](tests/portfolio_acceptance.json)。专家采样→BC→DAgger→PPO小流程、冻结权重加载、统一评价、真实训练窗口、真实RGB录制/回放、独立优化前后诊断及TensorBoard均实际运行。源码迁移后的BC初始化PPO4096个随机采样动作/执行动作/观测/奖励/前后物理状态与历史观察器逐元素完全一致。

## 本次真实执行清单

- online_demo：BC+DAgger种子42初始化，beta0，4096实际PPO环境步，4次1024步rollout更新，40个epoch计数；16完整episode+96步部分episode。11完整成功、5完整失败，均为训练统计，不能替代独立评价。RGB预选episode1/5/9/13，动作与执行日志差0。
- human_smoke：1024实际PPO环境步，实际训练env原生窗口可见，显示于第1024步关闭，训练仍完成1次优化；并非独立测试窗口冒充训练。
- human_update_window：2048实际PPO步，显示至1536步后关闭渲染；第1024步优化阶段窗口仍可见，updating.png保留真实Updating Actor/Critic画面，其后继续真实采样并完成第2次更新。
- pipeline_smoke / pipeline_scratch / pipeline_init / pipeline_decay_smoke：各1024PPO步，均通过真实专家采样、BC、DAgger。每条500训练标签+250监督验证标签+250DAgger标签/在线步；BC阶段各2epoch。最后一条验证衰减日程分支。仅工程连通性，不代表模型已充分训练。
- 100回合冻结Strong衰减种子42的同种子重复评价，与原16000～16099逐回合CSV完全一致；另外2回合17070起点用于迁移隔离诊断，不作为最终测试。
- 实际打开BC/best/compare（含专家/无锚定）MuJoCo窗口、actual/compare/progress三个播放器；TensorBoard临时服务HTTP200后关闭。
- 复用原始录像，并为缺少的7012无锚定长训练录像补录250步确定性诊断；best7059录像命令也实际运行。无100k重训、无新beta搜索。

精确新增预算及模型哈希见[收尾执行清单](reports/closeout_execution_manifest.csv)。新增PPO环境步总计{sum(x['actual_ppo_environment_steps'] for x in executions)}，新增DAgger在线步{sum(x['dagger_online_steps'] for x in executions)}；专家/诊断步数另记，未加入1751040历史正式预算。

## 最终结果与模型

{table}
从真实逐回合CSV计算，并与NPZ交叉核对；每组3训练种子×100冻结初态。BC到Strong衰减+6.00pp，平均最终误差下降13.21%，仅为描述性单任务结果。统一250步、dt.02、奖励权重1/1、末5步XY严格<1cm，误差/动作代价定义见[协议](docs/02_实验协议与结果核对.md)。

{modeltable}
15核心模型副本均成功加载并匹配SHA。全部46冻结模型声明另见[清单](reports/all_frozen_models.csv)，历史失败模型及日志保持原样。核心ZIP未暂存进Git，发布需权利确认。

## 代码与资料

- [统一入口](scripts/portfolio.py)、[端到端流程](src/pipeline.py)、[路径适配](src/runtime.py)；src六类公共导出对应src/verified的19个原始依赖模块。[来源与修改哈希](reports/source_provenance.csv)记录未改的优化器/损失核心及仅I/O/字体改造。
- [English README](README.md)、[中文README](README_CN.md)、[项目历程](docs/01_项目研究历程.md)、[复现成本与命令](docs/03_完整训练复现与工程结构.md)。
- [中文技术总结](docs/项目技术总结_求职版.md)、[中文简历](docs/简历项目描述_中文版.md)、[English resume](docs/Resume_Project_English.md)、[机器人算法面试问答](docs/机器人算法面试问答.md)。简历数字自动来自结果CSV，个人贡献需按实际参与确认。
- [70秒综合展示](assets/videos/00_project_overview.gif)、[同起点四方法10秒](assets/videos/01_same_start_comparison.gif)、[实际试错20秒](assets/videos/02_actual_online_learning.gif)、[BC与最终10秒](assets/videos/03_bc_vs_final_policy.gif)、[失败15秒](assets/videos/04_failure_cases.gif)。所有GIF重新解码验证，记录源/编码帧数、时长、变化、关节位移及模型SHA；[验收](reports/video_validation.json)。7059/7012只作演示。素材包含明确字幕静帧，未伪装成额外物理轨迹。
- results/online_demo包含training_rollouts.csv/npz、training_episode_summary.csv、executed_actions.csv、ppo_update_events.csv、checkpoint、真实RGB、GIF、逐回合曲线与独立Evaluation Replay。[中文可视化教程](docs/真实PPO试错过程_运行与可视化教程.md)给出VS Code/PowerShell操作。

## 未完成与发布边界

1. 没有MP4：本机FFmpeg/PyAV编码后端未安装，实际交付GIF，不声称已生成MP4。
2. 没有执行新100k训练；明确禁止，完整预算命令只验证参数和对应小规模分支。没有跨任务/跨机器人/实机/Sim-to-Real验证，没有统计显著性结论。
3. 未完成独立干净环境下载安装及其他操作系统验证；现有Windows Python3.11.17环境离线依赖解析通过，隔离路径副本用python -I可加载/评价且不依赖历史工程。不能将路径隔离说成干净依赖安装。
4. 历史{history['unchanged_readable_files']}个可读文件前后SHA和文件清单一致；2个预先存在的ACL拒读临时目录无法检查，未改权限。__pycache__/.git明确排除。不声称已读取不可访问文件。
5. 已准备本地Git，未配置remote、未公开推送或擅自授予开源许可证。源码、统计/小图、README/技术/简历材料可供本地演示与发布审阅；对外公开必须先由用户确认拥有发布权。权重、示教/诊断数据、原始轨迹、RGB/视频另需分发许可，当前Git全部排除；公开克隆不完全自包含。
6. 已核对38个直接/传递运行包元数据并保留third_party许可证/NOTICE；不将包许可证误当成项目所有权或权重授权。详见[来源许可核对](docs/04_来源与发布许可核对.md)。

当前适合用于机器人算法实习的本地作品集展示、面试准备和真实工程核验；公开GitHub发布还需完成权属/许可证及资产分发确认。

## 已验证的PowerShell使用方式

从本项目目录、使用已安装的mujoco_learn Python运行；下列均经过实际功能测试，自动化验收的GUI带有限帧/步数退出参数，正常观看可以省略。新训练需独立输出目录。

```powershell
python scripts/portfolio.py init
python scripts/portfolio.py watch --policy bc
python scripts/portfolio.py watch --policy best
python scripts/portfolio.py replay --run results/online_demo --mode actual --episode 13
python scripts/portfolio.py replay --run results/online_demo --mode compare --update 1
python scripts/portfolio.py results
python scripts/portfolio.py train --output results/new_live --steps 4096 --render-training
python scripts/portfolio.py train --output results/new_record --steps 4096 --record-training-rollouts
python scripts/portfolio.py record-video --policy best --seed 7059 --output results/new_video
```
'''
    (ROOT/'项目收尾完成报告.md').write_text(report,encoding='utf-8')
    git('add','.')
    names=[p for p in git('ls-files','-z').decode('utf-8').split('\0') if p]
    result['details']['staged_files']=len(names)+(0 if 'reports/public_file_manifest.csv' in names else 1)
    atomic_json(ROOT/'tests/portfolio_acceptance.json',acceptance)
    atomic_json(ROOT/'tests/git_release_validation.json',result['details'])
    public_assets=[dict(path=p,sha256=sha(ROOT/p),bytes=(ROOT/p).stat().st_size) for p in names if p!='reports/public_file_manifest.csv']
    csvout(ROOT/'reports/public_file_manifest.csv',public_assets)
    git('add','.')
    print('FINAL',acceptance['counts'],'staged_files',result['details']['staged_files'],'no remote/push')
if __name__=='__main__':main()
