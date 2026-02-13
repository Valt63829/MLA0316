import numpy as np
import random

# Environment Setup
num_robots = 2
tasks = ["A", "B", "C", "D"]

# Task dependencies
dependencies = {
    "A": [],
    "B": ["A"],
    "C": ["A"],
    "D": ["B", "C"]
}

# Q-table for each robot (states = completed tasks count, actions = tasks)
Q_tables = [np.zeros((5, len(tasks))) for _ in range(num_robots)]

alpha = 0.1
gamma = 0.9
epsilon = 0.2
episodes = 200

def choose_action(Q, state):
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, len(tasks) - 1)
    else:
        return np.argmax(Q[state])

for episode in range(episodes):
    completed = []
    state = 0
    
    while len(completed) < len(tasks):
        for i in range(num_robots):
            action = choose_action(Q_tables[i], state)
            task = tasks[action]
            
            reward = -5  # default penalty
            
            # Check dependency
            if task not in completed:
                if all(dep in completed for dep in dependencies[task]):
                    completed.append(task)
                    reward = 10
            
            next_state = len(completed)
            
            # Q-learning update
            Q_tables[i][state, action] += alpha * (
                reward + gamma * np.max(Q_tables[i][next_state]) 
                - Q_tables[i][state, action]
            )
            
            state = next_state
            
            if len(completed) == len(tasks):
                break

print("Training Completed!")
print("Final Q-table for Robot 1:\n", Q_tables[0])
print("Final Q-table for Robot 2:\n", Q_tables[1])
