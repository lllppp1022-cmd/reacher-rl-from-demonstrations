# Stable Online Reinforcement Learning from Demonstrations for Robotic Control

Investigating policy degradation and reproducible online fine-tuning in MuJoCo Reacher with expert demonstrations, DAgger and a frozen-BC action-mean constraint.

[中文](README_CN.md) · [Protocol](docs/02_实验协议与结果核对.md) · [Source provenance](reports/source_provenance.csv)

## Robot control demonstration

![Four methods from the same starting state](assets/videos/01_same_start_comparison_preview.gif)

**Evaluation Replay:** IK+PD, BC+DAgger, unanchored PPO and decaying strong anchoring; each case uses the same initial state and target. Seeds7059/7012 are demonstrations, not final-test statistics. [Full comparison](assets/videos/01_same_start_comparison.gif) · [70-second overview](assets/videos/00_project_overview.gif).

## Pipeline

![IK+PD to BC, DAgger, PPO, anchoring and evaluation](assets/figures/pipeline.png)

## Frozen evaluation results

| Method | Additional PPO steps / seed | Stable success | Mean final XY error |
|---|---:|---:|---:|
| BC + DAgger | 0 | 58.67% | 2.3029 cm |
| PPO without anchoring | 100352 | 32.33% | 2.7527 cm |
| PPO with decaying strong anchoring | 100352 | 64.67% | 1.9986 cm |

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
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The following commands need **no historical weights or datasets**:

```powershell
python scripts/portfolio.py --help
python scripts/portfolio.py results
python scripts/portfolio.py evaluate --method ikpd --first-seed 17080 --episodes 2 --output results/expert_diagnostic
python scripts/portfolio.py pipeline --profile smoke --output results/my_smoke --ppo-method anchor
```

The smoke pipeline collects500 training expert labels,250 validation labels and250 DAgger labels/online steps, runs two supervised epochs per stage and1024 PPO steps using its newly trained BC teacher. It is a plumbing check, not a new research result. Use a new output directory per run. [Reproduction costs and full-budget entry points](docs/03_完整训练复现与工程结构.md).

**Pretrained weights, diagnostic NPZ and raw training trajectories are not distributed.** For pretrained viewing or BC-initialized online recording, separately obtain authorized assets:15 ZIPs in configs/models.json plus configs/diagnostic_observations.npz with its hash in reports/asset_provenance.csv. Import below assumes that you already have the authorized original archive layout; it does not download weights. Missing assets cause an explicit path-and-SHA error. After provision:

```powershell
python scripts/portfolio.py import-assets --archive-root ..
python scripts/portfolio.py init
python scripts/portfolio.py watch --policy compare --seed 7059
python scripts/portfolio.py train --output results/my_demo --steps 4096 --record-training-rollouts
python scripts/portfolio.py replay --run results/my_demo --mode actual --episode 13
python scripts/portfolio.py replay --run results/my_demo --mode compare --update 1
```

## Actual online trial and error

![Actual stochastic PPO training rollout](assets/videos/02_actual_online_learning_preview.gif)

**Actual Training Rollout:** these RGB frames were captured from the environment PPO actually stepped, with sampled actions, raw rewards, distances and policy versions. No replacement predict calls or extra environment steps generated this recording. Preselected episodes1/5/9/13 include failures and explicit gaps;1024-step updates can occur within an episode. Reducing the preview to10fps changes only display sampling, not the source trajectory. [Full recording](assets/videos/02_actual_online_learning.gif) · [BC vs final](assets/videos/03_bc_vs_final_policy.gif) · [Failure cases](assets/videos/04_failure_cases.gif) · [Frame/hash validation](reports/public_media_validation.json).

`train --render-training` opens the actual training environment; run it separately from RGB recording mode. `replay --mode compare` is deterministic **Evaluation Replay**, generated after training, and is never labeled actual sampling. [Chinese operation guide](docs/真实PPO试错过程_运行与可视化教程.md). TensorBoard: `python -m tensorboard.main --logdir results/my_demo/tensorboard --port 6008`.

## Limitations, provenance and references

Single-task simulation, three training seeds, no real-robot deployment, cross-robot validation, Sim-to-Real experiment or statistical-significance claim. Low action drift does not guarantee accurate positioning. Full100k jobs were not rerun for publication. GIFs are reviewed simulator recordings; previews are smaller derivatives and originals remain unchanged. Model weights, training datasets and large raw state/RGB archives stay excluded.

The owner authorized public distribution of independently developed project source, documents, results, figures and reviewed GIFs. Third-party library license/NOTICE text remains in third_party with [dependency records](reports/dependency_licenses.csv) and [provenance review](docs/04_来源与发布许可核对.md). No MIT/Apache license for the project's own code has been selected; public visibility does not create an unspecified open-source license grant.

References: [PPO](https://arxiv.org/abs/1707.06347), [DAgger](https://proceedings.mlr.press/v15/ross11a.html), [Stable-Baselines3](https://jmlr.org/papers/v22/20-1364.html), [MuJoCo](https://mujoco.org/), [Gymnasium Reacher](https://gymnasium.farama.org/environments/mujoco/reacher/).
