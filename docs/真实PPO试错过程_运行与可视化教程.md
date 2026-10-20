# 真实PPO试错过程：运行与可视化教程

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
