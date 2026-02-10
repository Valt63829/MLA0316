import numpy as np
import gym
from gym import spaces

# ---------------- SMART GRID ENVIRONMENT ----------------
class SmartGridEnv(gym.Env):
    def __init__(self):
        super(SmartGridEnv, self).__init__()

        self.observation_space = spaces.Box(
            low=0, high=100, shape=(4,), dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-10, high=10, shape=(1,), dtype=np.float32
        )

        self.state = None

    def reset(self):
        demand = np.random.uniform(20, 80)
        production = np.random.uniform(10, 60)
        battery = np.random.uniform(0, 50)
        price = np.random.uniform(1, 5)
        self.state = np.array([demand, production, battery, price])
        return self.state

    def step(self, action):
        demand, production, battery, price = self.state
        energy_from_grid = action[0]

        supply = production + energy_from_grid
        imbalance = abs(demand - supply)

        cost = energy_from_grid * price
        reward = - (cost + imbalance)

        demand = np.random.uniform(20, 80)
        production = np.random.uniform(10, 60)
        price = np.random.uniform(1, 5)

        self.state = np.array([demand, production, battery, price])
        done = False

        return self.state, reward, done, {}

# ---------------- SIMPLE TRPO AGENT (POLICY GRADIENT) ----------------
class TRPOAgent:
    def __init__(self, env):
        self.env = env
        self.lr = 0.01
        self.policy = np.random.randn(4)

    def select_action(self, state):
        action = np.dot(self.policy, state) * 0.01
        return np.array([action])

    def train(self, episodes=500):
        rewards = []

        for episode in range(episodes):
            state = self.env.reset()
            episode_reward = 0

            for _ in range(50):
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)

                # Policy update (simplified)
                self.policy += self.lr * reward * state

                state = next_state
                episode_reward += reward

            rewards.append(episode_reward)
            if episode % 50 == 0:
                print(f"Episode {episode}, Reward: {episode_reward:.2f}")

        return rewards

# ---------------- RUN TRAINING ----------------
env = SmartGridEnv()
agent = TRPOAgent(env)
training_rewards = agent.train()
