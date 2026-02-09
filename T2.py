import numpy as np
import random

# Grid Environment
GRID_SIZE = 5
FOOD_POS = (4, 4)
GHOST_POS = (2, 2)
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']

# Q-learning parameters
alpha = 0.1       # Learning rate
gamma = 0.9       # Discount factor
epsilon = 1.0     # Exploration rate
epsilon_decay = 0.995
min_epsilon = 0.01
episodes = 500

# Initialize Q-table: (states x actions) -> 25x4
Q = np.zeros((GRID_SIZE * GRID_SIZE, len(ACTIONS)))

def state_to_index(pos):
    return pos[0] * GRID_SIZE + pos[1]

def get_next_position(pos, action):
    x, y = pos
    if action == 'UP':
        x = max(x - 1, 0)
    elif action == 'DOWN':
        x = min(x + 1, GRID_SIZE - 1)
    elif action == 'LEFT':
        y = max(y - 1, 0)
    elif action == 'RIGHT':
        y = min(y + 1, GRID_SIZE - 1)
    return (x, y)

def get_reward(pos):
    if pos == FOOD_POS:
        return 10
    elif pos == GHOST_POS:
        return -10
    else:
        return 0

# Training
for episode in range(episodes):
    pos = (0, 0)  # start position
    done = False
    while not done:
        state = state_to_index(pos)
        
        # Epsilon-greedy action
        if random.uniform(0,1) < epsilon:
            action_idx = random.randint(0, len(ACTIONS)-1)
        else:
            action_idx = np.argmax(Q[state])
        action = ACTIONS[action_idx]
        
        next_pos = get_next_position(pos, action)
        reward = get_reward(next_pos)
        next_state = state_to_index(next_pos)
        
        # Q-learning update
        Q[state, action_idx] = Q[state, action_idx] + alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action_idx])
        
        pos = next_pos
        
        if pos == FOOD_POS or pos == GHOST_POS:
            done = True
    
    # Reduce exploration
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

print("Training finished.\n")

# Evaluation
pos = (0, 0)
path = [pos]
total_reward = 0
done = False
while not done:
    state = state_to_index(pos)
    action_idx = np.argmax(Q[state])
    pos = get_next_position(pos, ACTIONS[action_idx])
    path.append(pos)
    total_reward += get_reward(pos)
    if pos == FOOD_POS or pos == GHOST_POS:
        done = True

print("Agent Path:", path)
print("Total Reward Collected:", total_reward)
