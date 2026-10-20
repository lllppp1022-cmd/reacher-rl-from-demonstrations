"""Generate portfolio prose and all headline numbers from the audited result CSV."""
from pathlib import Path
import sys,csv,json
ROOT=Path(__file__).resolve().parents[1]
def rows(name):
    with (ROOT/name).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def put(name,body):
    p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(body.strip()+'\n',encoding='utf-8')
def main():
    r=rows('reports/final_results.csv');per=rows('reports/per_seed_results.csv');bc=r[0];best=r[-1]
    gain=float(best['stable_success_pct'])-float(bc['stable_success_pct']);reduction=100*(1-float(best['mean_final_cm'])/float(bc['mean_final_cm']))
    labels=['BC + DAgger','PPO, no anchor','Fixed medium','Fixed strong','Decaying strong']
    table='| Method | Steps / seed | Stable success | Final XY error | Action cost |\n|---|---:|---:|---:|---:|\n'
    for label,x in zip(labels,r):table+=f"| {label} | {x['steps_per_seed']} | {float(x['stable_success_pct']):.2f}% | {float(x['mean_final_cm']):.4f} cm | {float(x['mean_action_sq_sum']):.6f} |\n"
    commands=r'''```powershell
# From the repository root, in the installed Python environment
python scripts/portfolio.py init
python scripts/portfolio.py watch --policy bc
python scripts/portfolio.py watch --policy best
python scripts/portfolio.py watch --policy compare --seed 7012
python scripts/portfolio.py evaluate --method long_decay_strong --training-seed 42 --episodes 100
python scripts/portfolio.py train --output results/my_demo --steps 4096 --record-training-rollouts
python scripts/portfolio.py replay --run results/online_demo --mode actual --episode 13
python scripts/portfolio.py replay --run results/online_demo --mode compare --update 1
python scripts/portfolio.py results
python scripts/portfolio.py record-video --policy best --seed 7059 --output results/my_video
python -m tensorboard.main --logdir results/online_demo/tensorboard --port 6008
```
'''
    put('README.md',f'''# Stable Online Reinforcement Learning from Demonstrations for Robotic Control

[中文](README_CN.md) · [Protocol](docs/02_实验协议与结果核对.md) · [Acceptance](tests/portfolio_acceptance.json)

MuJoCo Reacher control with an IK+PD expert, behavior cloning, one-round DAgger and PPO fine-tuning constrained by a frozen BC actor. This project investigates policy degradation during online learning and makes the actual stochastic sampler observable.

![Robot control](assets/figures/robot_control.png)

Frozen historical evaluation: decaying strong anchoring reached **{float(best['stable_success_pct']):.2f}%** stable success and **{float(best['mean_final_cm']):.4f} cm** final XY error, versus BC's **{float(bc['stable_success_pct']):.2f}% / {float(bc['mean_final_cm']):.4f} cm**. The descriptive difference is **+{gain:.2f} percentage points** and **{reduction:.2f}%** lower mean error. Three training seeds do not establish statistical significance or broad task generality.

{table}
Numbers are generated from archived per-episode CSV and cross-checked against NPZ trajectories; [per-seed results](reports/per_seed_results.csv), [SHA256 model manifest](reports/model_manifest.csv). Training seeds42/123/2026; each model100 frozen environment seeds16000–16099. Stable success requires all final5 XY errors <1cm. Episodes250 steps, dt0.02s, original distance/control reward weights1/1. Action cost is the per-episode sum of squared executed actions. BC's zero PPO steps excludes its supervised and DAgger data costs. Reported long budgets are100352 actual steps. These are project settings, not an unmodified benchmark default.

![Framework](assets/figures/pipeline.png)
![Results; error bars show sample SD across training seeds](assets/figures/final_comparison.png)

The actor uses only10 observation values and2 bounded torque-control actions. PPO uses1024-step rollouts,64-sample minibatches and10 optimization epochs. Anchoring penalizes raw Gaussian mean MSE against a frozen BC policy on actual rollout states. Adam actor/critic rates are1e-5/3e-4, with global gradient clipping0.5. Medium/strong beta2000/10000 were calibrated from gradient diagnostics, not asserted to be universal optima. Strong decay holds10000 through20480 steps, then decreases linearly to1000 at100352 steps.

## Quick start

Tested locally with Python3.11 on Windows and an existing MuJoCo environment; exact installed pins are in requirements.txt and requirements-lock.txt. Provision authorized pretrained assets before running init: models and training-only diagnostic NPZ are local-only and deliberately excluded from Git. This is **not a self-contained public weights/data distribution**. On the original local archive layout, `python scripts/portfolio.py import-assets --archive-root ..` provisions all15 SHA-verified models. For another machine, obtain approved assets matching configs/models.json and the diagnostic-observation hash in reports/asset_provenance.csv. No100k retraining is needed for pretrained demonstrations.

Installation template (fresh isolated download/install has not been tested offline): `python -m venv .venv`, activate the environment, then `python -m pip install -r requirements.txt`. The installed-environment offline dependency-resolution check is recorded separately in acceptance.

{commands}
Use a new output directory for each training run. Native windows were tested with bounded frame/step limits; omit limits for normal viewing. `train --render-training` opens the actual environment used by PPO; keep it separate from RGB recording mode. `replay --mode actual` reads captured RGB and logged actions without calling predict or stepping an environment. `--mode compare` shows deterministic **Evaluation Replay**, generated after training, and never substitutes it for **Actual Training Rollout**. See [complete Chinese operation guide](docs/真实PPO试错过程_运行与可视化教程.md).

[70-second overview](assets/videos/00_project_overview.gif) · [Same-start control](assets/videos/01_same_start_comparison.gif) · [Actual online sampler](assets/videos/02_actual_online_learning.gif) · [BC vs final](assets/videos/03_bc_vs_final_policy.gif) · [Failures](assets/videos/04_failure_cases.gif). GIFs are local optional assets; [validation](reports/video_validation.json) reports source frames, encoded frames, durations and model hashes. MP4 was not produced because no FFmpeg/PyAV backend is installed. Seeds7059/7012 are demonstration cases, not final-test statistics. The70-second montage includes explicit title cards and preserved data-source labels.

## Reproduction and scope

`python scripts/portfolio.py pipeline --profile smoke --output results/new_smoke` runs the preserved expert collection→BC→DAgger→Anchoring PPO chain with small budgets. Full-cost commands and actual expert-label costs are in [reproduction guide](docs/03_完整训练复现与工程结构.md); full100k jobs were not rerun for this closeout. Public facades in src map to19 preserved dependency-resolved modules under src/verified; [source provenance](reports/source_provenance.csv) lists every I/O/font-only change. src/runtime.py handles relative model paths. Historical files were audited read-only.

This is simulation research using established methods, without real-robot deployment, Sim-to-Real validation, cross-task tests or proof of causal superiority. Failures and between-seed variation are retained. No anchoring guarantee prevents all degradation. A new100-episode run on16000–16099 is explicitly **repeat verification**, not a new independent final test.

## Publication status

Local Git only; no remote and no public push. Project-code ownership, weights/data rights and media publication permission require owner confirmation. No open-source license grant has been invented. [Publication review](docs/04_来源与发布许可核对.md), [dependency licenses](reports/dependency_licenses.csv) and third_party notices document the installed dependencies. models/, results/, runtime/, videos and local audit snapshots are excluded from Git; do not silently remove these exclusions before authorization. Optional binary media links require approved asset publication.
''')
    put('README_CN.md',f'''# 基于模仿学习与策略约束的机器人在线强化学习控制

[English](README.md) · [完整收尾报告](项目收尾完成报告.md) · [运行教程](docs/真实PPO试错过程_运行与可视化教程.md)

在MuJoCo Reacher中实现IK+PD示教、BC、DAgger与固定BC参考策略约束的PPO，研究在线微调退化及梯度约束尺度，并记录真实随机探索和网络更新。

![控制展示](assets/figures/robot_control.png)

最终冻结测试复核：Strong衰减方案稳定成功率**{float(best['stable_success_pct']):.2f}%**、平均末端XY误差**{float(best['mean_final_cm']):.4f} cm**；BC对应**{float(bc['stable_success_pct']):.2f}% / {float(bc['mean_final_cm']):.4f} cm**。差异为**+{gain:.2f}个百分点**、平均误差下降**{reduction:.2f}%**，属于三训练种子的描述性结果。

{table}
表格从原始逐回合CSV重新计算，并与NPZ的距离、动作及最后5步成功标志交叉核对。训练种子42/123/2026；每模型测试环境种子16000～16099。250步回合，dt=0.02s，距离/控制惩罚权重均1；成功要求最后连续5步XY误差严格小于1cm。动作代价为整个回合执行动作平方和。BC步数0仅表示未进行在线PPO，不表示无示教成本；长训练实际100352步。不能直接与其他默认环境配置比较。

![流程](assets/figures/pipeline.png)
![结果及三训练种子样本标准差](assets/figures/final_comparison.png)

快速运行使用已安装Python环境，确切版本见requirements。15个核心模型已经放入本地models；Git排除权重、数据及大录像。干净克隆需要取得许可后另行提供匹配SHA的权重和诊断NPZ，不能声称公开仓库完全自包含。在原历史目录同级可运行 `python scripts/portfolio.py import-assets --archive-root ..`。新建虚拟环境与联网安装属于安装模板，尚未完成独立离线干净环境验证；现有环境依赖解析和所有功能入口经过测试。

{commands}
训练请使用新输出目录。真实训练窗口命令增加 `--render-training`，RGB录制用 `--record-training-rollouts`，两者分开运行。自动验收用有限帧/步数关闭窗口，正常观看可省略限制。固定beta与衰减日程、监督训练及从零PPO入口见[复现说明](docs/03_完整训练复现与工程结构.md)。

[70秒综合展示](assets/videos/00_project_overview.gif)、[同起点四方法](assets/videos/01_same_start_comparison.gif)、[真实试错](assets/videos/02_actual_online_learning.gif)、[BC与最终策略](assets/videos/03_bc_vs_final_policy.gif)、[失败案例](assets/videos/04_failure_cases.gif)。视频为本地可选资源，发布前需确认授权。当前输出GIF，未生成MP4；本机缺少FFmpeg/PyAV后端。7059、7012仅为演示，不进入最终统计。Episode13是真实完整失败，Episode17最后96步是未完成回合。

已有算法的工程实现与实验分析，不宣称原创算法、统计显著性、跨机器人普遍有效或实机部署。无新100k训练、无奖励改动、无扩大beta搜索。重复评价明确标记repeat_verification，历史失败与日志仍保留原件。

仅创建本地Git，未公开推送，未擅自指定开源许可证。项目源码权属、模型/数据/媒体发布许可待用户确认；[许可核对](docs/04_来源与发布许可核对.md)与third_party保留依赖声明。
''')
    hist='''# 项目研究历程

1. reacher_bc_ppo：IK+PD以初始关节构型选择逆解分支，全回合保持；采集5000～5999训练示教、6000～6099监督验证，分别250000/25000样本。BC只训练Actor，固定训练集标准化、阶段等权采样；闭环表现不足触发一次DAgger，在学生访问的100个训练回合补充25000专家标签并重新训练。学生始终只看10维观测，不使用专家目标关节角或episode编号。
2. 同阶段比较纯PPO、BC初始化PPO。PPO从零训练未获得良好精定位；BC行为在在线优化后可明显退化。旧结果的最终种子8000～8099，只用于这一阶段，不能当作最新16000测试结果。
3. reacher_safe_finetune：比较原始学习率、Critic预热、低Actor学习率方案C、KL停止及更大探索std。C降低每次Actor位移，但不能保证100k保持BC。最终集10000～10099；上一阶段C筛选不代表后来Anchoring已完成。
4. reacher_bc_anchor：独立beta0/.01/.1/1短实验、独立筛选及后续长实验。冻结参考BC、审计beta0等价与教师梯度。该阶段测试12000～12099。小beta损失数值与PPO梯度相差较大，不能由“损失非零”推断约束有效。
5. reacher_anchor_calibration：在训练/诊断集进行均值Actor梯度尺度与实际Adam位移审计；一次标定200/2000/10000，独立四组×三种子短训练20480步。验证15000～15099筛选，最多两候选。正式长训练100352步：b0对照及medium/strong固定、衰减。weak未通过筛选，其固定/衰减共6个长任务未执行，不能补造结果。
6. 最新最终集16000～16099在模型哈希冻结后评价；46个声明含别名，37个唯一ZIP，评价复用按相同哈希明示。最终五组作品集从已有逐回合数据重新计算；最新正式训练12短+15长=1751040环境步。诊断、试跑、行为检查、可视化不混入此预算。
7. 在线观察系统在SB3真实collect_rollouts和实际env.step边界记录高斯均值/随机样本/执行动作、奖励、完整MuJoCo状态；原生窗口观察训练env本体。事后诊断回放与真实训练录像分开。保留成功、失败和跨rollout回合。
8. 本次收尾：迁移最小必要模块与完整依赖，复核原结果，执行小规模工程测试和录像整理；没有重新进行100k研究或更新历史结论。

四阶段配置、验收文件的快照与SHA见reports/historical_audit；历史根目录逐文件保护清单仅保留本地tests/local_history_before.json，原数据/失败日志不删除。各阶段测试种子不同，不跨阶段横向拼接统计。源码迁移保留原目录关系，只适配输出路径、导入副作用和可选字体，算法核心未重写。
'''
    put('docs/01_项目研究历程.md',hist)
    detail='\n'.join(f"| {x['method']} | {x['training_seed']} | {float(x['stable_success_pct']):.0f}% | {float(x['mean_final_cm']):.4f} |" for x in per)
    put('docs/02_实验协议与结果核对.md',f'''# 实验协议与结果核对

{table}
数据链：历史final_freeze.json声明 → ZIP SHA256 → final_comparison.csv对应job → results/final/job/episodes.csv → trajectories.npz。逐模型重新计算100回合均值，三训练种子再等权平均；标准差为三训练种子样本标准差ddof=1，不把300回合视为300次独立训练。最终动作平方和是每步两个执行动作平方相加，再累加250步；误差为最后动作后的XY距离，奖励使用原3D距离项，二者定义不同。

训练种子42/123/2026，共用同一组16000～16099初始状态、250步、dt.02、动作[-1,1]、距离/控制奖励权重1/1。稳定成功=最后连续5步XY距离均严格<.01m。验证15000～15099只负责候选及共同checkpoint选择，最终测试不参与训练和挑选。作品集展示最终100352步模型；control验证最佳可能是step0，不能偷偷改用step0替代100k退化对照。

衰减Strong：初始beta10000，20480步后线性下降，100352步为1000；固定Medium2000、固定Strong10000。所有方案从同训练种子的BC独立初始化，不从候选中途分叉。PPO1024/64/10，gamma.99，GAE.95，clip.2，vf.5，entropy0，Actor/critic学习率1e-5/3e-4，Adam eps1e-5，全局梯度裁剪.5，无预热、无KL提前停止。约束不直接作用log_std或Critic，教师无梯度且不属于优化器。

| Method | Training seed | Stable success | Final cm |
|---|---:|---:|---:|
{detail}

最新46模型声明的哈希核对见all_frozen_models.csv；核心15个副本路径见model_manifest.csv。CSV成功标志与NPZ最后5步距离逐回合一致；最终误差和动作代价数值一致，详见archive_recalculation.json。BC监督/DAgger成本另计，PPO在线预算的0不代表零成本。

Strong衰减对BC均值+{gain:.2f}pp、误差下降{reduction:.2f}%；只有3训练种子，种子123明显困难。不能宣称统计显著、普遍最优、跨任务有效或Sim-to-Real成功。固定Strong漂移更小也未必成功率更高；这些指标衡量不同性质。最新IK+PD为99%稳定成功，有实际失败，不能引用旧数据的100%替代。

本次统一评价是同一冻结模型/同一最终集的重复核验，输出目录repeat_verification；不是新独立测试。所有短训练属于工程演示，不追加到正式结果表。
''')
    put('docs/03_完整训练复现与工程结构.md',r'''# 完整训练复现与工程结构

src/verified按原兄弟目录保留19个必要模块，AST解析本地import传递依赖；src/{envs,expert,imitation,rl,evaluation,visualization}是原实现导出入口。核心anchor_ppo.py未改。src/runtime.py仅重建相对模型注册表和输出配置；scripts/portfolio.py提供统一CLI。源码逐文件来源和修改见source_provenance.csv。不要直接从旧项目默认配置启动vendored脚本，统一使用适配入口。

```powershell
python scripts/portfolio.py pipeline --profile smoke --output results/new_smoke --ppo-method anchor
python scripts/portfolio.py pipeline --profile smoke --output results/new_scratch --ppo-method scratch
python scripts/portfolio.py pipeline --profile smoke --output results/new_init --ppo-method init
```

三条小流程均已实际运行。每条独立采集2训练回合500专家标签、1验证回合250标签，BC2epoch；1回合DAgger补250标签和250在线步，再2epoch；最后1024环境步PPO。仅验证端到端管线，不代表足量训练或研究性能。scratch不复制BC参数；init仅转移Actor且std=.005；anchor使用固定BC参考与原分组优化器。

完整成本入口（本次未运行，明确未执行）：

```powershell
python scripts/portfolio.py pipeline --profile full --output results/full_scratch --ppo-method scratch
python scripts/portfolio.py pipeline --profile full --output results/full_bc_init --ppo-method init
python scripts/portfolio.py pipeline --profile full --output results/full_fixed --ppo-method anchor --group strong --schedule fixed
python scripts/portfolio.py pipeline --profile full --output results/full_decay --ppo-method anchor --group strong --schedule linear
```

每条full入口当前是训练种子42的单管线，不是三种子完整历史研究的一键重做；其参数/依赖已通过对应smoke分支与help检查，高成本训练未执行。full采集1000训练回合250000标签、100监督验证回合25000标签、DAgger100回合25000在线标签和在线步，BC与DAgger重训各至多60epoch并早停，PPO100352步。DAgger为明确请求的复现流程固定执行，历史研究则按闭环验证阈值触发。scratch模式仍经过示教管线用于依赖验收，但其PPO不读示教参数；纯PPO本身不消耗专家标签。该单管线不能声称重新复现27项筛选实验；冻结网格/筛选/梯度审计完整研究原件在历史归档，作品集以协议、执行清单与来源哈希提供追溯。

历史正式实验：12×20480+15×100352=1751040环境步，不含专家、DAgger、诊断与失败试跑。报告同时记录weak六项长训练未执行。CPU可运行MLP；渲染需支持OpenGL的桌面和MuJoCo。RGB为640×480、每回合250帧，未压缩约230MB，逐录制回合落盘释放。稀疏录制会有额外成本，默认不启用正式训练高频渲染。实际短流程、训练/渲染耗时见results/*/complete.json，仅为本机记录，不是严格跨硬件性能基准。完整历史experiments的elapsed_seconds见reports/experiment_manifest.csv；不能将小流程时间线性外推为保证。

模型ZIP、训练数据、原始RGB、TensorBoard、runtime与结果中间件均Git忽略。15模型及诊断NPZ的路径/哈希见configs/models.json与asset_provenance.csv；原专家数据可由采样代码重新生成，但历史权重准确复现需要历史版本、相同训练数据、随机数与协议。未授予重新分发权利前，公开克隆不包含这些资源。再次创建runtime用init，现有完成的训练目录只做哈希核对，不覆盖；不完整目录需要新路径。
''')
    put('docs/04_来源与发布许可核对.md','''# 来源与发布许可核对

本交付的项目代码来自用户指定的四个历史工程及根目录专家/评价依赖；逐文件SHA和修改见source_provenance.csv。新代码为I/O适配、审计、验收和文档/媒体组合，不把既有算法声称为原创。历史工程缺少明确、可据以再授权的项目许可证或完整作者签署文件，当前没有自行添加MIT/Apache等开源授权。

发布状态：本地整理、运行和Git暂存已获本任务授权；对外发布源码、训练权重、数据及派生媒体的权属仍需用户确认是否涉及实验室、课程、合作方或第三方限制。确认前所有项目内容均为待发布审阅版本，不能标为已获开放许可。依赖已经按安装包元数据递归核对，license原文/NOTICE复制到third_party；dependency_licenses.csv记录38个运行时包的版本与声明。包声明不代表对本项目权重/数据授予许可。

可直接用于本地演示和个人核验：代码、README、报告、图、GIF、模型副本与小规模测试记录。可进入公开仓库的候选：清理后的源码、双语README、技术/简历文本、已脱敏指标和小图，仍以用户具有发布权为前提。需要单独确认与提供的资源：models全部ZIP、configs/diagnostic_observations.npz、历史轨迹NPZ/示教数据、真实训练RGB/视频及GIF。Git默认排除全部这些二进制和本机日志。

第三方方法引用：PPO（Schulman et al., 2017, Proximal Policy Optimization Algorithms, arXiv:1707.06347）；DAgger（Ross et al., 2011, A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning, AISTATS）；MuJoCo（Todorov et al., 2012, MuJoCo: A physics engine for model-based control）；Stable-Baselines3（Raffin et al., 2021, Stable-Baselines3: Reliable Reinforcement Learning Implementations, JMLR22(268)）；Gymnasium的Reacher-v5环境与分发包。项目使用固定参考行为MSE约束，未把算法组合包装为原创方法。

没有创建GitHub仓库、remote或公开推送。发布前确认权利、选定合法许可证、决定允许分发的资产，再补充公开下载入口；这些属于未完成的发布事项，非本地工程阻塞。
''')
    tutorial=r'''# 真实PPO试错过程：运行与可视化教程

在PowerShell中先进入项目文件夹，激活已有mujoco_learn环境；VS Code选择同一Python解释器，终端运行以下命令。命令以仓库根为工作目录，移动仓库后先运行init重新生成本机runtime。已有models来自只读历史模型副本，init核对SHA。原生MuJoCo窗口需要图形桌面。

```powershell
python scripts/portfolio.py init
python scripts/portfolio.py watch --policy compare --seed 7059
python scripts/portfolio.py train --output results/my_live --steps 4096 --render-training
```

watch是Evaluation Replay，使用保存策略的确定性动作；train --render-training的窗口是被SB3调用的训练环境本体，保留高斯随机探索。Actor根据10维观测计算两个高斯均值和std，采样raw_action，再限幅到[-1,1]，唯一一次env.step产生物理反馈和原始奖励。均值不等于真实随机样本。窗口显示Collecting Experience，每1024实际环境步进入Updating Actor/Critic，优化不推进仿真。R/Esc或关窗可恢复无渲染训练；该模式适合低速教学，不用于比较训练吞吐。

```powershell
python scripts/portfolio.py train --output results/my_record --group b0 --steps 4096 --record-training-rollouts --record-every-n-episodes 4 --max-recorded-episodes 4
python scripts/portfolio.py replay --run results/my_record --mode progress
python scripts/portfolio.py replay --run results/online_demo --mode actual --episode 13
python scripts/portfolio.py replay --run results/online_demo --mode compare --update 1
```

progress可在第二个终端监控当前采样/优化状态。my_record完成后把回放命令run参数改为它。actual读训练进程实际捕获的RGB及逐步日志，不加载策略重新采样，也不调用step。默认预选episode1/5/9/13，其他回合仍有轻量日志，不会用新回放替代缺失RGB。GUI支持actual/compare/progress切换、空格暂停、左右逐帧与速度调节。compare使用同起点独立诊断环境的确定性轨迹，在训练完成后生成，明确标为Evaluation Replay；它只能解释策略变化，不能冒充采样轨迹。

PPO先收集1024个随机交互步，通过Critic估值和GAE计算优势/回报目标，再对同一rollout进行10个epoch、每批64样本的优化，即每轮160次Adam更新，同时更新Actor/Critic。不是每动作重新更新网络。4轮4096环境步对应4次PPO优化事件、40个SB3 epoch计数；_n_updates不是环境步，也不是160次优化器step的同义词。

奖励为负3D末端距离减执行动作平方和；训练收益改善可能仅反映动作变小、轨迹早段改善，不能证明最终XY定位成功。独立评价看最后5步严格1cm标准、最终误差和动作代价。BC Anchoring在rollout状态上约束两个原始动作均值的MSE，参考BC冻结；它不直接约束std或Critic，也不保证全部目标成功。

日志training_rollouts.csv/npz包含均值、随机/执行动作、std、前后观测、MuJoCo49维完整integration_state、关节/速度/目标/末端/奖励、策略版本及checkpointSHA。executed_actions.csv在env实际执行点独立记账；rollout_recording_validation.json确认与SB3 rollout buffer完全一致、动作差0、奖励差0，且没有多step。终止后的next_observation在自动reset前捕获；记录原始奖励，不把SB3超时bootstrap混为环境奖励。

250步episode与1024步rollout不对齐：episode5/9/13跨策略更新边界，回合内版本变化是真实事件。4096结束时episode17只有96步，标记partial_at_training_stop，不能当失败。Episode13的250步完整失败和episode15“最后一步小于1cm但最后5步不全满足”的失败都保留。training_distance_curve.png/training_action_curve.png逐回合展示距离与动作，红线表示优化边界。失败可能与策略位移、状态分布和精度要求有关，但日志不能证明单一因果原因。

```powershell
python scripts/portfolio.py evaluate --method long_decay_strong --training-seed 42 --episodes 100
python scripts/portfolio.py record-video --policy best --seed 7059 --output results/my_video
python -m tensorboard.main --logdir results/online_demo/tensorboard --port 6008
```

在浏览器访问localhost:6008查看diagnostics，ppo_update_events.csv记录每次更新前/后步数、policy_loss/value_loss、approx_kl、clip_fraction、anchor loss、BC偏差及模型SHA。policy_update_timeline.png按真实环境步对齐；evaluation/policy_before_after.png是独立诊断，不是训练曲线。4次更新后的version4在4096训练结束产生，本次没有version4真实后续rollout，不能用其诊断轨迹冒充第5轮探索。

CSV/NPZ、PNG和GIF都在指定输出目录。GIF已解码核对帧数、时长、图像变化、关节运动；项目综合GIF为70秒，含字幕静帧与真实片段，源种子7059/7012为演示场景。没有MP4编码后端，未声称生成MP4。

辨别数据来源：窗口标题/文件元数据/真实env步/随机样本/rollout buffer校验均标Actual Training Rollout才是采样；watch和compare始终是Evaluation Replay。录制回合是事先选择而非成功后筛选。RGB绘制异常由原观察器捕获并关闭显示，继续训练；不承诺驱动崩溃或磁盘耗尽时仍能保存。每次新训练使用新目录；完成目录只校验缓存，想真正重新开始必须换路径。不要把缓存返回“ALREADY COMPLETE”计成新训练。
'''
    put('docs/真实PPO试错过程_运行与可视化教程.md',tutorial)
    tech=f'''# 项目技术总结（求职版）

任务是仿真两关节机械臂定位，10维输入为两关节cos/sin4维、目标XY2维、关节速度2维、末端相对目标XY2维；动作2维执行器控制[-1,1]。奖励负3D末端距离与控制平方惩罚，权重均1；250步、dt.02；定位成功是最后5步XY均<1cm。

专家用几何IK选临近初始构型的分支，PD用角度误差/速度补偿和actuator gear生成限幅动作。它保留初始分支记忆，瞬时10维学生可能面对多模态示教风险。BC为64-Tanh-64-Tanh均值Actor，固定训练集标准化，只训练Actor；阶段平衡损失避免末端近零动作淹没早期控制。DAgger让学生执行，在访问状态请专家标注，聚合后重训。

PPO用高斯动作随机试错、GAE优势、clip ratio目标，Actor/critic独立网络。固定BC参考均值MSE锚定项与PPO和价值损失同批优化；教师参数冻结并排除优化器，保存重载分离处理。既有方法组合，工程重点是稳定性诊断及证据链。

1. 纯PPO为何差？历史匹配配置与预算下精定位差，可能受探索、奖励折中及数据效率影响；不能说PPO算法普遍无效。
2. BC初始化为何仍退化？优化目标是带动作代价的累积奖励，初始Critic不准确、优势噪声、访问分布与随机探索均可能推动均值偏离示教；这是诊断解释，不是单变量因果证明。
3. 低Actor学习率作用？Actor1e-5、Critic3e-4让估值适应更快、动作均值每次位移更小，短期更稳定，但长训练仍可能累积退化。
4. beta=.01为什么弱？实际控制动作尺度很小，原始均值MSE梯度相对PPO梯度很弱；存在非零惩罚不表示实际Adam步受影响。
5. 为什么测梯度？损失标量的单位和尺度不同；比较相同参数的均值Actor梯度、夹角、全局clip系数、固定Adam状态的反事实更新位移，再看行为。log_std不在anchor的直接梯度中，不能拿含std的向量冒充均值约束尺度。
6. Strong衰减为何较好？最终三种子描述性均值最高：{float(best['stable_success_pct']):.2f}%、{float(best['mean_final_cm']):.4f}cm。早期保留BC、后期释放适应空间是合理假设；并未证明衰减机制具有普遍因果优势，也未证明当前beta最优。
7. 如何真实采样？SB3高斯采样raw→限幅执行→reward/nextstate→1024rollout→10epoch/64batch更新；日志对照实际env.step和buffer，独立回放明确分开。
8. 为什么奖励不等于定位？奖励同时看全轨迹距离和动作成本，降低动作可能改善某些回报但损失末端精度，必须用相同冻结种子看最后5步成功。
9. 为何不是Sim-to-Real？没有真实机械臂、传感噪声/延迟、动力学差异、硬件执行器及安全实验；OpenGL机械臂运动仍是仿真。

可核实能力：真实依赖迁移、Actor/Critic优化器分组、梯度与Adam审计、验证集筛选、三种子冻结模型SHA、逐回合统计重算、真实采样录制与完整失败保留。相对BC成功率+{gain:.2f}pp、误差下降{reduction:.2f}%；样本少、单任务、未跨机器人验证，不能声称统计显著。失败episode13最后XY1.7395cm，episode17为部分回合。全部数值来源见final_results.csv及真实轨迹日志。
'''
    put('docs/项目技术总结_求职版.md',tech)
    put('docs/简历项目描述_中文版.md',f'''# 基于模仿学习与策略约束的机器人在线强化学习控制

- 搭建MuJoCo Reacher两关节控制管线，结合IK+PD专家示教、Actor行为克隆与DAgger数据聚合，实现10维观测、2维控制的闭环策略。
- 实现BC初始化PPO、Actor/Critic分组学习率与固定参考动作均值约束，审计梯度尺度、Adam更新和策略漂移，比较固定及衰减约束的长期稳定性。
- 在3个训练种子、每模型100个冻结测试初态上，Strong衰减方案稳定1cm成功率达{float(best['stable_success_pct']):.2f}%（BC {float(bc['stable_success_pct']):.2f}%），平均最终误差{float(best['mean_final_cm']):.4f}cm（下降{reduction:.2f}%）；结果为单任务仿真的描述性比较。
- 开发真实随机rollout记录、1024步PPO更新事件及前后策略诊断可视化，提供CSV/NPZ、checkpoint SHA、失败轨迹、双语运行文档和自动验收。

数字自动取自reports/final_results.csv。无实机部署、无原创算法或统计显著性声明；个人职责应按实际参与情况确认后用于简历。
''')
    put('docs/Resume_Project_English.md',f'''# Stable Online Reinforcement Learning from Demonstrations for Robotic Control

- Built a MuJoCo Reacher pipeline combining an IK+PD expert, actor-only behavior cloning and DAgger for closed-loop control from10-dimensional observations and2-dimensional actions.
- Implemented BC-initialized PPO with separate actor/critic learning rates and a frozen-reference action-mean penalty; audited gradient scales, Adam updates and fixed versus decaying regularization.
- Achieved{float(best['stable_success_pct']):.2f}% stable1cm success versus{float(bc['stable_success_pct']):.2f}% for BC, with{float(best['mean_final_cm']):.4f}cm mean final error ({reduction:.2f}% lower), across3 training seeds and100 frozen evaluation starts per model; descriptive, single-task simulation results.
- Instrumented actual stochastic rollouts,1024-step PPO update events and separate before/after diagnostic replays, with checkpoint hashes, failure retention, reproducible commands and automated acceptance checks.

Metrics are generated from reports/final_results.csv. Confirm personal contributions before using these bullets. No original-algorithm, statistical-significance, physical deployment or Sim-to-Real claim.
''')
    put('docs/机器人算法面试问答.md','''# 机器人算法面试问答

**状态/动作是什么？**10维cos/sin、目标XY、两关节速度、末端相对目标XY；2维[-1,1]执行器控制。完整MuJoCo状态不是10维观测，录像保存49维integration_state和RGB。

**IK、PID与PPO区别？**IK由几何求目标构型；项目实际是PD而非带积分PID，KP20/KD10并用gear换算。PPO由交互回报拟合高斯控制器。专家保留IK分支记忆，学生瞬时输入不含该记忆。

**BC为何有效？**专家标签提供合理动作先验和低代价精定位，减少纯在线探索。Actor监督训练、Critic未被示教训练；固定标准化和分段采样来自训练数据，标签MSE不等于闭环成功。

**DAgger解决什么？**学生访问状态可能不在专家轨迹分布，学生执行但专家标注，在100个训练回合补25000标签并聚合，不在测试环境取标签。仍不能保证解决隐藏分支记忆问题。

**PPO clip objective是什么？**最小化负的min(ratio×优势, clip(ratio,1±.2)×优势)；旧log_prob来自真实rollout。限制的是样本相对概率目标，不是动作绝对变化硬约束，不能保证不退化。

**GAE在这里怎样用？**gamma=.99、lambda=.95，对1024步rollout计算优势；时限截断按SB3 bootstrap，记录器保留bootstrap前环境原始奖励与reset前terminal observation，避免日志概念混淆。

**Actor/Critic各做什么？**Actor输出高斯均值和可学习std，Critic估值指导优势；独立64×64网络，Actor1e-5、Critic3e-4，Adam eps1e-5。BC只复制Actor均值与固定归一化，不能说Critic也有专家价值函数。

**BC Anchoring怎么实现？**rollout观测上原始高斯均值与固定BC均值的MSE，平均覆盖batch及两个动作坐标。教师no_grad、requires_grad=False、不在optimizer。惩罚没有直接log_std或Critic梯度。总目标加beta×MSE，KL只是诊断，不把MSE说成KL正则。

**为什么Adam/裁剪重要？**全局梯度裁剪.5会受Critic梯度影响；Adam还有二阶尺度及动量，原始梯度比不等于参数步比例。控制相同权重、batch和深拷贝Adam状态才能比较反事实位移；浅拷贝优化器会共享张量导致错误结论。

**KL与策略漂移？**报告固定BC高斯参考std.005相对PPO的raw Gaussian KL、动作均值MSE、参数漂移和PPO approx_kl，它们不同。执行器clip后动作差也单独记录；低参数漂移不等于低定位误差。

**为什么用验证集？**15000～15099筛选约束并选择共同checkpoint，最终16000～16099只在冻结后评价。相同最终集的本次重跑叫重复核验，不是新独立测试，也不能再用它调beta。

**失败如何分析？**真实episode13完整250步、跨3072更新，最终1.7395cm；记录动作/均值/随机噪声/奖励/关节/目标与距离，避免只发成功片段。episode15最后一步过线但末5步不全过线仍失败；episode17只有96步，不能误判失败。变化时间相关不证明一次更新就是失败唯一原因。

**Sim-to-Real完成了吗？**没有。只做MuJoCo仿真。需要动力学与执行器辨识、噪声/延迟/摩擦评估、硬件接口、部署安全和真实重复实验，才能讨论迁移。

**局限和后续方向？**仅单任务3训练种子，未统计显著性分析；没有跨机器人测试、实机或全局最优beta证明。候选性能保持门槛不保证长期稳定。后续应预先规定新任务/新种子协议，扩大证据后再谈泛化；本次工程收尾没有执行这些新研究。
''')
    put('.gitignore','''__pycache__/
*.py[cod]
.venv/
runtime/
models/*/
*.zip
*.pt
*.npz
results/
assets/videos/*
!assets/videos/README.md
tests/local_*
tests/command_logs/
reports/runtime_environment.json
reports/*local*.json
*.log
''')
    put('models/README.md','Pretrained weights are provisioned locally, SHA256-listed in configs/models.json, and excluded from Git pending distribution permission. Do not substitute a different checkpoint for a missing model.')
    put('assets/videos/README.md','Local GIF assets: 00_project_overview (70s), 01_same_start_comparison (10s), 02_actual_online_learning (20s), 03_bc_vs_final_policy (10s), 04_failure_cases (15s). Publication rights pending; Git intentionally excludes binary videos. See reports/video_validation.json.')
    put('docs/README命令验证说明.md','''快速开始命令均经功能验证：模型初始化、BC/best/compare原生窗口、100回合重复评价、4096实际录制、1024原生训练窗口、actual/compare/progress播放器、GIF录制、结果显示、三种1024步端到端流程及TensorBoard服务。测试用max-steps/max-frames自动关窗，新输出目录名可不同。完整100k命令仅验证help和对应smoke分支，未执行。联网依赖下载与跨平台干净安装未执行；现有环境进行了离线pip解析，不把其声称为全新隔离环境安装。具体命令、退出码、输出日志和耗时见tests/command_validation.json及本地command_logs。''')
    print('DOCS GENERATED',gain,reduction)
if __name__=='__main__':main()
