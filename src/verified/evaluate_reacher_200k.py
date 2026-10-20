
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import PPO

# ==============================
# 1. 实验配置
# ==============================

SEEDS = range(100, 200)
N_STEPS = 250
THRESHOLD = 0.01

# Reacher两段连杆的有效长度（m）
L1, L2 = 0.10, 0.11

# 使用之前已经验证的PD参数
KP = np.array([20.0, 20.0])
KD = np.array([10.0, 10.0])

POLICIES = ("IK+PD", "PPO", "Random", "Zero")

OUT = Path("reacher_eval_results_200k")
# Portfolio: no filesystem side effects during import.


# ==============================
# 2. 角度与逆运动学
# ==============================

def wrap_angle(angle):
    """将角度差转换到[-pi, pi]。"""
    return np.arctan2(np.sin(angle), np.cos(angle))


def solve_ik(target_xy, initial_q):
    """求解两连杆IK，选择离初始角度较近的一组解。"""

    target_xy = np.asarray(target_xy, dtype=float)
    r = np.linalg.norm(target_xy)

    # Reacher第二关节限位为[-3, 3] rad
    min_r = np.sqrt(
        L1**2 + L2**2
        + 2 * L1 * L2 * np.cos(3.0)
    )
    max_r = L1 + L2

    # 超出可达范围时，投影至最近的可达半径
    r_used = np.clip(r, min_r + 1e-6, max_r - 1e-6)
    projected = abs(r_used - r) > 1e-10

    if r > 1e-12:
        xy = target_xy * (r_used / r)
    else:
        xy = np.array([r_used, 0.0])

    x, y = xy

    c2 = (
        (x*x + y*y - L1*L1 - L2*L2)
        / (2 * L1 * L2)
    )
    c2 = np.clip(c2, -1.0, 1.0)

    theta2 = np.arccos(c2)

    # 两组弯曲方向不同的IK解
    candidates = []

    for q2 in (theta2, -theta2):
        q1 = np.arctan2(y, x) - np.arctan2(
            L2 * np.sin(q2),
            L1 + L2 * np.cos(q2)
        )
        candidates.append(np.array([q1, q2]))

    # 选择离初始关节角度更近的解
    selected = min(
        candidates,
        key=lambda q: np.linalg.norm(
            wrap_angle(q - initial_q)
        )
    )

    return selected, projected


# ==============================
# 3. IK + PD控制器
# ==============================

def pd_action(obs, target_q, gear):
    """根据观测计算PD关节控制动作。"""

    # 利用正余弦恢复关节角度
    q = np.arctan2(obs[2:4], obs[0:2])

    # 获取关节角速度
    dq = obs[6:8]

    # 关节角度误差
    error = wrap_angle(target_q - q)

    # PD计算期望关节力矩
    torque = KP * error - KD * dq

    # 根据gear转换为执行器输入
    return torque / gear


# ==============================
# 4. 保存CSV
# ==============================

def write_csv(file_path, rows):
    with open(
        file_path, "w",
        newline="", encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f, fieldnames=list(rows[0])
        )

        writer.writeheader()
        writer.writerows(rows)


# ==============================
# 5. 主评估程序
# ==============================

def main():

    if not Path("ppo_reacher.zip").exists():
        raise FileNotFoundError(
            "找不到ppo_reacher.zip，请检查项目目录。"
        )

    # 加载已经训练好的PPO
    model = PPO.load(
    "ppo_reacher_250_A_200k",
    device="cpu"
)

    # 统一奖励权重
    env = gym.make(
    "Reacher-v5", 
    max_episode_steps=250,
    reward_dist_weight=1.0,
    reward_control_weight=1.0
)

    gear = env.unwrapped.model.actuator_gear[:2, 0].copy()

    assert np.all(gear != 0)
    assert np.isclose(env.unwrapped.dt, 0.02)

    print("控制周期：", env.unwrapped.dt)
    print("动作惩罚权重：",
          env.unwrapped._reward_control_weight)
    print("执行器gear：", gear)

    results = []
    histories = {}

    try:
        # 100组初始条件
        for seed in SEEDS:

            reference_obs = None

            # 每组初始条件都测试四种策略
            for name in POLICIES:

                obs, info = env.reset(seed=seed)
                env.action_space.seed(seed)

                # 检查初始状态是否完全一致
                if reference_obs is None:
                    reference_obs = obs.copy()
                elif not np.allclose(
                    obs, reference_obs,
                    rtol=0, atol=1e-12
                ):
                    raise RuntimeError(
                        f"Seed={seed}初始状态不一致"
                    )

                initial_distance = float(
                    np.linalg.norm(obs[8:10])
                )

                projected = False

                # IK+PD需要先计算目标关节角度
                if name == "IK+PD":

                    initial_q = np.arctan2(
                        obs[2:4], obs[0:2]
                    )

                    goal_xy = env.unwrapped.get_body_com(
                        "target"
                    )[:2].copy()

                    target_q, projected = solve_ik(
                        goal_xy, initial_q
                    )

                distances = []
                action_norms = []
                action_costs = []
                rewards = []

                # 同样的50个控制步
                for step in range(N_STEPS):

                    if name == "IK+PD":
                        action = pd_action(
                            obs, target_q, gear
                        )

                    elif name == "PPO":
                        action, _ = model.predict(
                            obs, deterministic=True
                        )

                    elif name == "Random":
                        action = env.action_space.sample()

                    else:
                        action = np.zeros(2)

                    # 统一执行器限幅
                    action = np.asarray(
                        action, dtype=np.float32
                    )
                    action = np.clip(
                        action,
                        env.action_space.low,
                        env.action_space.high
                    )

                    # 所有方法统一调用env.step()
                    obs, reward, terminated, truncated, info = (
                        env.step(action)
                    )

                    # 计算并记录评价指标
                    distance = float(
                        np.linalg.norm(obs[8:10])
                    )

                    distances.append(distance)
                    action_norms.append(
                        float(np.linalg.norm(action))
                    )
                    action_costs.append(
                        float(np.sum(action**2))
                    )
                    rewards.append(float(reward))

                    if terminated or truncated:
                        break

                if len(distances) != N_STEPS:
                    raise RuntimeError(
                        f"{name}, seed={seed}未执行完整50步"
                    )

                # 保存该回合所有指标
                results.append({
                    "policy": name,
                    "seed": seed,
                    "initial_distance_m": initial_distance,
                    "trajectory_mean_m": float(
                        np.mean(distances)
                    ),
                    "final_distance_m": distances[-1],
                    "closest_distance_m": float(
                        np.min(distances)
                    ),
                    "action_sq_sum": float(
                        np.sum(action_costs)
                    ),
                    "action_norm_mean": float(
                        np.mean(action_norms)
                    ),
                    "reward_sum": float(
                        np.sum(rewards)
                    ),
                    "final_success_1cm": int(
                        distances[-1] < THRESHOLD
                    ),
                    "stable_success_1cm": int(
                        all(d < THRESHOLD for d in distances[-5:])
                    ),
                    "ever_success_1cm": int(
                        any(d < THRESHOLD for d in distances)
                    ),
                    "ik_goal_projected": int(projected)
                })

                # 保存一组代表性轨迹
                if seed == 123:
                    histories[name] = distances.copy()

    finally:
        env.close()

    # 保存每个回合的结果
    write_csv(OUT / "episodes.csv", results)

    # ==============================
    # 6. 计算四种策略的平均结果
    # ==============================

    summary = []

    for name in POLICIES:

        group = [
            row for row in results
            if row["policy"] == name
        ]

        def mean(key):
            return float(np.mean([
                row[key] for row in group
            ]))

        summary.append({
            "policy": name,
            "n_episodes": len(group),
            "mean_trajectory_cm":
                100 * mean("trajectory_mean_m"),
            "mean_final_cm":
                100 * mean("final_distance_m"),
            "mean_action_sq_sum":
                mean("action_sq_sum"),
            "mean_reward_sum":
                mean("reward_sum"),
            "final_success_pct":
                100 * mean("final_success_1cm"),
            "stable_success_pct":
                100 * mean("stable_success_1cm"),
            "ever_success_pct":
                100 * mean("ever_success_1cm"),
            "ik_projected_count":
                sum(x["ik_goal_projected"] for x in group)
        })

    write_csv(OUT / "summary.csv", summary)

    # 打印评价结果
    print("\n========= 公平对比结果 =========")

    for x in summary:
        print(f"\n策略：{x['policy']}")
        print("平均轨迹距离(cm)：",
              round(x["mean_trajectory_cm"], 3))
        print("平均最终距离(cm)：",
              round(x["mean_final_cm"], 3))
        print("平均动作平方和：",
              round(x["mean_action_sq_sum"], 3))
        print("平均累计奖励：",
              round(x["mean_reward_sum"], 3))
        print("最终1cm成功率(%)：",
              round(x["final_success_pct"], 1))
        print("稳定1cm成功率(%)：",
              round(x["stable_success_pct"], 1))

    print(
        "\nIK目标投影次数：",
        summary[0]["ik_projected_count"]
    )

    # ==============================
    # 7. 绘制代表性轨迹
    # ==============================

    plt.figure(figsize=(9, 5))

    for name in POLICIES:
        plt.plot(
             range(1, N_STEPS + 1),
            histories[name],
            label=name
        )

    plt.axhline(
        THRESHOLD,
        color="gray",
        linestyle="--",
        label="1 cm threshold"
    )

    plt.xlabel("Control step")
    plt.ylabel("End-effector distance (m)")
    plt.title("Reacher: Four Policies (seed=123)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        OUT / "trajectory_seed123.png",
        dpi=160
    )

    # ==============================
    # 8. 绘制平均最终距离柱状图
    # ==============================

    plt.figure(figsize=(8, 5))

    plt.bar(
        [x["policy"] for x in summary],
        [x["mean_final_cm"] for x in summary]
    )

    plt.ylabel("Mean final distance (cm)")
    plt.title("Reacher: 100-Episode Comparison")
    plt.tight_layout()

    plt.savefig(
        OUT / "mean_final_distance.png",
        dpi=160
    )

    print("\n实验完成，结果保存至：")
    print(OUT.resolve())

    plt.show()


if __name__ == "__main__":
    main()
