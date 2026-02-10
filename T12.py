import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ================= HIGHWAY ENVIRONMENT =================
class HighwayEnv(gym.Env):
    def __init__(self):
        super(HighwayEnv, self).__init__()

        self.observation_space = spaces.Box(
            low=0, high=100, shape=(4,), dtype=np.float32
        )

        self.action_space = spaces.Discrete(3)  # 0: stay, 1: left, 2: right
        self.max_steps = 100
        self.reset()

    def reset(self):
        self.speed = np.random.uniform(20, 30)
        self.front_distance = np.random.uniform(10, 60)
        self.front_speed = np.random.uniform(15, 25)
        self.lane = 1
        self.steps = 0
        return self._get_state()

    def _get_state(self):
        return np.array([
            self.speed,
            self.front_distance,
            self.front_speed,
            self.lane
        ], dtype=np.float32)

    def step(self, action):
        reward = 0
        done = False
        self.steps += 1

        # Lane change logic
        if action == 1 and self.lane > 0:
            self.lane -= 1
            reward -= 1
        elif action == 2 and self.lane < 2:
            self.lane += 1
            reward -= 1

        # Speed control
        if self.front_distance < 15:
            self.speed = max(10, self.speed - 5)
            reward -= 10
        else:
            self.speed = min(35, self.speed + 2)
            reward += self.speed

        # Collision
        if self.front_distance < 5:
            reward -= 100
            done = True

        # Update traffic
        self.front_distance = np.random.uniform(5, 60)
        self.front_speed = np.random.uniform(15, 30)

        if self.steps >= self.max_steps:
            done = True

        return self._get_state(), reward, done, {}

# ================= PPO AGENT =================
class PPOAgent:
    def __init__(self, env):
        self.env = env
        self.lr = 0.001
        self.policy = np.random.randn(4, 3) * 0.01

    def choose_action(self, state):
        logits = np.dot(state, self.policy)

        # -------- STABLE SOFTMAX --------
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits)

        return np.random.choice(3, p=probs)

    def train(self, episodes=500):
        rewards = []

        for ep in range(episodes):
            state = self.env.reset()
            total_reward = 0

            for _ in range(100):
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)

                # -------- PPO-like policy update --------
                self.policy[:, action] += self.lr * reward * state

                # Clip policy to avoid explosion
                self.policy = np.clip(self.policy, -5, 5)

                state = next_state
                total_reward += reward

                if done:
                    break

            rewards.append(total_reward)

            if ep % 50 == 0:
                print(f"Episode {ep}, Total Reward: {total_reward:.2f}")

        return rewards

# ================= RUN TRAINING =================
env = HighwayEnv()
agent = PPOAgent(env)
training_rewards = agent.train()
