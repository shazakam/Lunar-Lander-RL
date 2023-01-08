import gymnasium as gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from dqn_agent import Agent
import pandas as pd

def tune_DQN(lr=0.0005,epsilon_dec=0.99,gamma=0.999,batch_size=64,n_episodes=1000):

    env = gym.make('LunarLander-v2', render_mode='rgb_array')
    record_env = gym.wrappers.RecordVideo(env, f"Videos/", episode_trigger=lambda x: x%100 == 0)
    env = record_env

    print('State shape: ', env.observation_space.shape)
    print('Number of actions: ', env.action_space.n)

    agent = Agent(state_size=8, action_size=4, seed=0)
    
    scores = []                        # list containing scores from each episode
    average_score = []
    eps_history = []
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = 1.0       
    # initialize epsilon

    for i_episode in range(1, n_episodes+1):
        
        state = env.reset()
        score = 0
        state, reward, done, trunc, _ = env.step(0)
        for t in range(1000):
            
            action = agent.act(state, eps)
            next_state, reward, done, trunc, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score) # save most recent score
        
        scores.append(score)              # save most recent score
        average_score.append(np.mean(scores_window))
        eps_history.append(eps)
        eps = max(0.01, epsilon_dec*eps) # decrease epsilon
        
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            print('Reward{:.2f}'.format(score))
        if i_episode == n_episodes:
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
    
    
    df = pd.DataFrame(list(zip(average_score,eps_history,len(average_score)*[epsilon_dec],
                               len(average_score)*[gamma],
                               len(average_score)*[lr],
                               len(average_score)*[batch_size])),
                      columns=["Average_Score","Epsilon","Epsilon_Decay","Gamma","Learning_Rate","Batch_Size"],
                      index=[x for x in range(1,len(average_score)+1)])
    
    print(df)
    df.to_excel(r'Hyperparameter_Resuts/\'epsd'+str(epsilon_dec)+'_'+'g'+str(gamma)+'_'+'lr'+str(lr)+'_'+'bs'+str(batch_size)+'.xlsx', index=False) 


gamma_test = [0.9 + 0.02475*x for x in range(5)] #0.9 -> 0.999
lr_test = [1e-4 + 0.000225*x for x in range(5)] #0.0001 -> 0.001
bs_test = [32, 64]
eps_test = [0.9 + 0.02475*x for x in range(5)]
print(gamma_test, '\n', lr_test, '\n', eps_test)
#tests = [epsilon_decay_test,gamma_test,lr_test,bs_test]
tune_DQN()


            
