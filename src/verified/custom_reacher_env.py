
import gymnasium as gym
import numpy as np


class CustomReacherEnv(gym.Env):
    """自定义Reacher末端定位强化学习环境。"""

    def __init__(
        self,
        distance_weight=1.0,
        velocity_weight=0.0,
        render_mode=None
    ):
        super().__init__()

        if distance_weight <= 0 or velocity_weight < 0:
            raise ValueError("奖励权重设置不合理")

        # 1. 创建底层MuJoCo机器人
        self.base_env = gym.make(
            "Reacher-v5",
            max_episode_steps=250,
            reward_dist_weight=1.0,
            reward_control_weight=1.0,
            render_mode=render_mode
        )

        # 2. 保留原始观测空间和动作空间
        self.observation_space = self.base_env.observation_space
        self.action_space = self.base_env.action_space
        self.metadata = self.base_env.metadata

        # 3. 自定义奖励参数
        self.distance_weight = distance_weight
        self.velocity_weight = velocity_weight

        # 4. 物理信息
        self.dt = float(self.base_env.unwrapped.dt)
        self.previous_error = None
        self.step_count = 0

    def reset(self, *, seed=None, options=None):
        """每个新回合开始时调用。"""

        super().reset(seed=seed)

        obs, info = self.base_env.reset(
            seed=seed,
            options=options
        )

        # 保存初始末端相对目标的位置
        self.previous_error = np.array(
            obs[8:10], dtype=np.float64
        )

        self.step_count = 0

        return obs, info

    def step(self, action):
        """执行控制动作，计算自定义奖励。"""

        # 1. 让MuJoCo计算下一时刻的运动
        obs, base_reward, terminated, truncated, info = (
            self.base_env.step(action)
        )

        self.step_count += 1

        # 2. 计算末端距离
        current_error = np.asarray(
            obs[8:10], dtype=np.float64
        )

        distance = float(np.linalg.norm(current_error))

        # 3. 估计末端平面速度
        tip_velocity = (
            current_error - self.previous_error
        ) / self.dt

        speed_sq = float(np.sum(tip_velocity**2))

        self.previous_error = current_error.copy()

        # 4. 自定义距离奖励
        # 原环境的距离权重已经是1.0
        original_dist_reward = float(info["reward_dist"])

        extra_distance_reward = (
            self.distance_weight - 1.0
        ) * original_dist_reward

        # 5. 仅在目标附近增强速度惩罚
        near_target_weight = np.exp(
            -0.5 * (distance / 0.04)**2
        )

        velocity_penalty = (
            -self.velocity_weight
            * near_target_weight
            * speed_sq
        )

        # 6. 组合新奖励
        reward = float(
            base_reward
            + extra_distance_reward
            + velocity_penalty
        )

        # 7. 返回额外诊断信息
        info = dict(info)

        info["distance_m"] = distance
        info["tip_speed_m_s"] = float(np.sqrt(speed_sq))
        info["reward_velocity_extra"] = float(
            velocity_penalty
        )
        info["reward_distance_extra"] = float(
            extra_distance_reward
        )
        info["is_success"] = bool(distance < 0.01)
        info["step_count"] = self.step_count

        # 不因第一次接近目标就结束回合
        return obs, reward, terminated, truncated, info

    def render(self):
        return self.base_env.render()

    def close(self):
        self.base_env.close()
