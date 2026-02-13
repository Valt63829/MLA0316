import numpy as np
import random

# Agents
num_agents = 3

# Hierarchical tasks
tasks = ["Gather_Wood", "Gather_Metal", "Assemble"]

# Dependencies
dependencies = {
    "Gather_Wood": [],
    "Gather_Metal": [],
    "Assemble": ["Gather_Wood", "Gather_Metal"]
}

# Q-table for each agent (state = completed tasks count, action = task)
Q_tables = [np.zeros((4, len(tasks))) for _ in range(num_agents)]

alpha = 0.1
gamma = 0.9
epsilon = 0.2
episodes = 300

def choose_action(Q, state):
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, len(tasks) - 1)
    return np.argmax(Q[state])

for episode in range(episodes):
    completed = []
    state = 0
    
    while len(completed) < len(tasks):
        for agent in range(num_agents):
            action = choose_action(Q_tables[agent], state)
            task = tasks[action]
            
            reward = -5
            
            if task not in completed:
                if all(dep in completed for dep in dependencies[task]):
                    completed.append(task)
                    reward = 15   # cooperative reward
            
            next_state = len(completed)
            
            # Q update
            Q_tables[agent][state, action] += alpha * (
                reward + gamma * np.max(Q_tables[agent][next_state]) 
                - Q_tables[agent][state, action]
            )
            
            state = next_state
            
            if len(completed) == len(tasks):
                break

print("Training Completed!\n")
for i in range(num_agents):
    print(f"Q-table for Agent {i+1}:\n", Q_tables[i], "\n")
