import numpy as np
import random

GRID = 5
ACTIONS = [(0,1),(0,-1),(1,0),(-1,0)]

class Agent:
    def __init__(self):
        self.position = (0,0)
        self.has_object = False

def move(agent, action):
    dx, dy = ACTIONS[action]
    x, y = agent.position
    new_pos = (x+dx, y+dy)

    if 0 <= new_pos[0] < GRID and 0 <= new_pos[1] < GRID:
        agent.position = new_pos

def navigate_to(agent, target):
    while agent.position != target:
        action = random.randint(0,3)
        move(agent, action)

def collect(agent, obj_pos):
    if agent.position == obj_pos:
        agent.has_object = True

def deliver(agent, goal):
    if agent.position == goal and agent.has_object:
        return True
    return False

# Simulation
agent = Agent()
object_pos = (2,2)
goal_pos = (4,4)

# MAXQ Task Execution
navigate_to(agent, object_pos)
collect(agent, object_pos)
navigate_to(agent, goal_pos)

success = deliver(agent, goal_pos)

print("Agent final position:", agent.position)
print("Object delivered:", success)
