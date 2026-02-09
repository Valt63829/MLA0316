import numpy as np
import random
from collections import defaultdict

# -----------------
# Call Center Setup
# -----------------
N_CSRS = 3  # number of CSRs
AVG_HANDLING_TIMES = [5, 8, 6]  # in minutes per CSR
EPISODES = 5000

# Initialize Q(s,a) and returns
Q = defaultdict(lambda: np.zeros(N_CSRS))
returns = defaultdict(lambda: [[] for _ in range(N_CSRS)])

# Initial policy: random assignment
def get_policy(state):
    if state not in Q:
        return np.random.randint(0, N_CSRS)
    return np.argmax(Q[state])

# Simulate an episode
def generate_episode():
    episode = []
    calls = 10  # number of calls in one episode
    state = tuple([1]*N_CSRS)  # all CSRs available initially
    
    for _ in range(calls):
        # Select action based on epsilon-greedy
        action = get_policy(state)
        # Simulate handling time for assigned CSR
        handling_time = AVG_HANDLING_TIMES[action]
        reward = -handling_time  # negative because we want to minimize time
        episode.append((state, action, reward))
        # For simplicity, all CSRs become available next step
        state = tuple([1]*N_CSRS)
    
    return episode

# Monte Carlo Control
for i in range(EPISODES):
    episode = generate_episode()
    G = 0
    for t in reversed(range(len(episode))):
        state, action, reward = episode[t]
        G += reward
        # First-visit MC: update if first occurrence
        if not any((state == s and action == a) for s, a, r in episode[:t]):
            returns[state][action].append(G)
            Q[state][action] = np.mean(returns[state][action])

# -----------------
# Evaluate Policy
# -----------------
state = tuple([1]*N_CSRS)
total_calls = 10
assigned_actions = []
total_time = 0

for _ in range(total_calls):
    action = np.argmax(Q[state])
    assigned_actions.append(action)
    total_time += AVG_HANDLING_TIMES[action]

print("Optimal CSR Assignment for Calls:", assigned_actions)
print("Total Handling Time:", total_time, "minutes")
print("Average Handling Time per Call:", total_time/total_calls, "minutes")
