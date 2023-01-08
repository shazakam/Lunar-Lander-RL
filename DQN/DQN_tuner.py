import gymnasium as gym
import numpy as np
from collections import deque
from dqn_agent import Agent
import pandas as pd

def tune_DQN(lr=5e-4,epsilon_dec=0.995,gamma=0.99,batch_size=64,n_episodes=1000):

    env = gym.make('LunarLander-v2',render_mode="rgb_array")
    record_env = gym.wrappers.RecordVideo(env,f"Videos/",episode_trigger=lambda x: x%100 == 0,)
    env = record_env

    agent = Agent(s_size=8, a_size=4, seed=0,lr=lr,BATCH_SIZE=batch_size,GAMMA=gamma)
    
    scores = []                        # list containing scores from each episode
    average_score = []
    eps_history = []
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = 1.0       
    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        state, _ = env.reset()
        
        score = 0
        for t in range(2000):
            action = agent.act(state, eps)
            next_state, reward, terminated,truncated, _ = env.step(action)
            done = terminated | truncated
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score) 
        
        scores.append(score)              
        average_score.append(np.mean(scores_window))
        eps_history.append(eps)
        eps = max(0.01, epsilon_dec*eps) 
        
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
    
    
    df = pd.DataFrame(list(zip(average_score,eps_history,len(average_score)*[epsilon_dec],
                               len(average_score)*[gamma],
                               len(average_score)*[lr],
                               len(average_score)*[batch_size])),
                      columns=["Average_Score","Epsilon","Epsilon_Decay","Gamma","Learning_Rate","Batch_Size"],
                      index=[x for x in range(1,len(average_score)+1)])
    
    print(df)
    df.to_excel(r'Hyperparameter_Resuts/result.xlsx', index=False) 
    return




#epsilon_decay_test = [0.9+x*0.0198 for x in range(0,6)]
#gamma_test = [0.9+x*0.0198 for x in range(0,6)]
#lr_test = [1e-4+x*0.000225 for x in range(0,6)]
#bs_test = [x for x in range(32,65,8)]
#tests = [epsilon_decay_test,gamma_test,lr_test,bs_test]

"""
for x in epsilon_decay_test:
    tune_DQN(epsilon_dec=x)

for x in gamma_test:
    tune_DQN(gamma=x)
    
for x in bs_test:
    tune_DQN(batch_size=x)

    
for x in lr_test:
    tune_DQN(lr=x)
    
"""


## CALL tune_DQN with specified input parameters to run training of agent to output results to Excel file in Hyperparameter_Results ##
tune_DQN(lr=0.00325,epsilon_dec=0.9792,gamma=0.999,batch_size=40,n_episodes=1000)
    
