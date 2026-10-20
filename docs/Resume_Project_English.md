# Stable Online Reinforcement Learning from Demonstrations for Robotic Control

- Built a MuJoCo Reacher pipeline combining an IK+PD expert, actor-only behavior cloning and DAgger for closed-loop control from 10-dimensional observations and 2-dimensional actions.
- Implemented BC-initialized PPO with separate actor/critic learning rates and a frozen-reference action-mean penalty; audited gradient scales, Adam updates and fixed versus decaying regularization.
- Achieved 64.67% stable 1cm success versus 58.67% for BC, with 1.9986cm mean final error (13.21% lower), across 3 training seeds and 100 frozen evaluation starts per model; descriptive, single-task simulation results.
- Instrumented actual stochastic rollouts,1024-step PPO update events and separate before/after diagnostic replays, with checkpoint hashes, failure retention, reproducible commands and automated acceptance checks.

Metrics are generated from reports/final_results.csv. Confirm personal contributions before using these bullets. No original-algorithm, statistical-significance, physical deployment or Sim-to-Real claim.
