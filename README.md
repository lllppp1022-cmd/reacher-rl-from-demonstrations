# Stable Online Reinforcement Learning from Demonstrations for Robotic Control

[中文](README_CN.md) · [Protocol](docs/02_实验协议与结果核对.md) · [Acceptance](tests/portfolio_acceptance.json)

MuJoCo Reacher control with an IK+PD expert, behavior cloning, one-round DAgger and PPO fine-tuning constrained by a frozen BC actor. This project investigates policy degradation during online learning and makes the actual stochastic sampler observable.

![Robot control](assets/figures/robot_control.png)

Frozen historical evaluation: decaying strong anchoring reached **64.67%** stable success and **1.9986 cm** final XY error, versus BC's **58.67% / 2.3029 cm**. The descriptive difference is **+6.00 percentage points** and **13.21%** lower mean error. Three training seeds do not establish statistical significance or broad task generality.

| Method | Steps / seed | Stable success | Final XY error | Action cost |
|---|---:|---:|---:|---:|
| BC + DAgger | 0 | 58.67% | 2.3029 cm | 0.157358 |
| PPO, no anchor | 100352 | 32.33% | 2.7527 cm | 0.157019 |
| Fixed medium | 100352 | 63.33% | 2.0939 cm | 0.161475 |
| Fixed strong | 100352 | 61.00% | 2.2421 cm | 0.154540 |
| Decaying strong | 100352 | 64.67% | 1.9986 cm | 0.162236 |

Numbers are generated from archived per-episode CSV and cross-checked against NPZ trajectories; [per-seed results](reports/per_seed_results.csv), [SHA256 model manifest](reports/model_manifest.csv). Training seeds 42/123/2026; each model100 frozen environment seeds 16000–16099. Stable success requires all final5 XY errors <1cm. Episodes250 steps, dt0.02s, original distance/control reward weights1/1. Action cost is the per-episode sum of squared executed actions. BC's zero PPO steps excludes its supervised and DAgger data costs. Reported long budgets are100352 actual steps. These are project settings, not an unmodified benchmark default.

![Framework](assets/figures/pipeline.png)
![Results; error bars show sample SD across training seeds](assets/figures/final_comparison.png)

The actor uses only 10 observation values and 2 bounded torque-control actions. PPO uses 1024-step rollouts,64-sample minibatches and 10 optimization epochs. Anchoring penalizes raw Gaussian mean MSE against a frozen BC policy on actual rollout states. Adam actor/critic rates are1e-5/3e-4, with global gradient clipping 0.5. Medium/strong beta 2000/10000 were calibrated from gradient diagnostics, not asserted to be universal optima. Strong decay holds10000 through 20480 steps, then decreases linearly to1000 at100352 steps.

## Quick start

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

## Reproduction and scope

`python scripts/portfolio.py pipeline --profile smoke --output results/new_smoke` runs the preserved expert collection→BC→DAgger→Anchoring PPO chain with small budgets. Full-cost commands and actual expert-label costs are in [reproduction guide](docs/03_完整训练复现与工程结构.md); full100k jobs were not rerun for this closeout. Public facades in src map to19 preserved dependency-resolved modules under src/verified; [source provenance](reports/source_provenance.csv) lists every I/O/font-only change. src/runtime.py handles relative model paths. Historical files were audited read-only.

This is simulation research using established methods, without real-robot deployment, Sim-to-Real validation, cross-task tests or proof of causal superiority. Failures and between-seed variation are retained. No anchoring guarantee prevents all degradation. A new100-episode run on16000–16099 is explicitly **repeat verification**, not a new independent final test.

## Publication status

The owner explicitly authorized private GitHub hosting of source (including src/verified), documents, anonymized metrics and seven PNG figures for this task. **Public release is not authorized.** No open-source license has been invented. Model/data/rollout/video assets remain local and excluded from Git. [Publication review](docs/04_来源与发布许可核对.md), [dependency licenses](reports/dependency_licenses.csv) and verbatim third_party notices preserve provenance. Repository creation, authentication and push verification status are recorded separately in the local upload report; local tests alone do not prove an upload succeeded.
