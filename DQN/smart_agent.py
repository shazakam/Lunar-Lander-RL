import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from dqn_agent import Agent



env = gym.make('LunarLander-v2')


env.seed(0)
agent = Agent(state_size=8, action_size=4, seed=0)
agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

for i in range(3):
    state = env.reset()
    for j in range(200):
        action = agent.act(state)
        env.render(height = 30)
        state, reward, done, _ = env.step(action)
        if done:
            break 
            
env.close()