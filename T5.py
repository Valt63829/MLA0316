import numpy as np

# Grid size
ROWS, COLS = 4, 4

# Pickup location
PICKUP = (3, 3)

# Actions
ACTIONS = {
    'U': (-1, 0),
    'D': (1, 0),
    'L': (0, -1),
    'R': (0, 1)
}

gamma = 0.9
theta = 0.001

# Initialize value function
V = np.zeros((ROWS, COLS))

def reward(state):
    if state == PICKUP:
        return 10
    return -1

def next_state(state, action):
    r, c = state
    dr, dc = ACTIONS[action]
    nr, nc = r + dr, c + dc
    if 0 <= nr < ROWS and 0 <= nc < COLS:
        return (nr, nc)
    return state  # hit wall

# Value Iteration
while True:
    delta = 0
    for r in range(ROWS):
        for c in range(COLS):
            state = (r, c)
            if state == PICKUP:
                continue

            v = V[r][c]
            action_values = []

            for a in ACTIONS:
                ns = next_state(state, a)
                rwd = reward(ns)
                action_values.append(rwd + gamma * V[ns])

            V[r][c] = max(action_values)
            delta = max(delta, abs(v - V[r][c]))

    if delta < theta:
        break

# Extract policy
policy = np.empty((ROWS, COLS), dtype=str)
for r in range(ROWS):
    for c in range(COLS):
        state = (r, c)
        if state == PICKUP:
            policy[r][c] = 'P'
            continue

        best_action = None
        best_value = -np.inf

        for a in ACTIONS:
            ns = next_state(state, a)
            val = reward(ns) + gamma * V[ns]
            if val > best_value:
                best_value = val
                best_action = a

        policy[r][c] = best_action

print("Optimal Value Function:")
print(np.round(V, 2))

print("\nOptimal Dispatch Policy:")
print(policy)
