import numpy as np
import matplotlib.pyplot as plt
import random

# Environment Parameters
GRID_SIZE = 10
START = (0, 0)
GOAL = (9, 9)
OBSTACLES = [(3,3), (3,4), (4,3), (6,7), (7,7)]

ACTIONS = [(0,1), (0,-1), (1,0), (-1,0)]  # Right, Left, Down, Up

# Q-table for low-level controller
Q = np.zeros((GRID_SIZE, GRID_SIZE, 4))

alpha = 0.1
gamma = 0.9
epsilon = 0.2
episodes = 300

def valid_state(state):
    x, y = state
    if x < 0 or x >= GRID_SIZE or y < 0 or y >= GRID_SIZE:
        return False
    if state in OBSTACLES:
        return False
    return True

def choose_action(state):
    if random.uniform(0,1) < epsilon:
        return random.randint(0,3)
    return np.argmax(Q[state[0], state[1]])

def step(state, action):
    dx, dy = ACTIONS[action]
    new_state = (state[0] + dx, state[1] + dy)

    if not valid_state(new_state):
        return state, -100

    if new_state == GOAL:
        return new_state, 100

    return new_state, -1

# Training
for ep in range(episodes):
    state = START

    while state != GOAL:
        action = choose_action(state)
        next_state, reward = step(state, action)

        best_next = np.max(Q[next_state[0], next_state[1]])
        Q[state[0], state[1], action] += alpha * (
            reward + gamma * best_next - Q[state[0], state[1], action]
        )

        state = next_state

# Visualization
grid = np.zeros((GRID_SIZE, GRID_SIZE))
for obs in OBSTACLES:
    grid[obs] = -1
grid[GOAL] = 2

plt.imshow(grid)
plt.title("Urban Grid Environment")
plt.colorbar()
plt.show()
