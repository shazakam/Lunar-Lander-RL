import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
from model import QNetwork
from memory import ReplayBuffer



BUFFER_SIZE = int(1e5)  # replay buffer size
TAU = 1e-3             
UPDATE_EVERY = 4        

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, s_size, a_size, seed, lr = 5e-4, BATCH_SIZE = 64,GAMMA = 0.99):
        self.GAMMA = GAMMA
        self.lr = lr
        self.BATCH_SIZE = BATCH_SIZE
        self.action_size = a_size
        self.seed = random.seed(seed)
        self.state_size = s_size
        self.current_step = 0
    
        # Q-Network
       
        self.qnetwork_target = QNetwork(s_size, a_size, seed).to(device) 
        self.qnetwork_local = QNetwork(s_size, a_size, seed).to(device) 
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr)

        # Replay memory
        self.memory = ReplayBuffer(a_size, BUFFER_SIZE, self.BATCH_SIZE, seed, device)
       
        
    
    def step(self, s, a, r, s_t, done):
        
        #Adds an experience to tbe Replay Buffer
        self.memory.add(s, a, r, s_t, done)
        
        self.current_step = (self.current_step + 1) % UPDATE_EVERY
        if self.current_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.BATCH_SIZE:
                experiences = self.memory.sample()
                #Learns from sampled experiences in ReplayBuffer
                self.learn(experiences, self.GAMMA)

    def act(self, state, eps=0.):
        #Returns eother a random action or a action following a policy
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        s, a, r, s_t, dones = experiences
        
        Q_targ_next = self.qnetwork_target(s_t).detach().max(1)[0].unsqueeze(1)
        Q_exp = self.qnetwork_local(s).gather(1, a)
        Q_targ = r + (gamma * Q_targ_next * (1 - dones))

        # Calculate Loss between Q_exp and Q_targ
        loss = F.mse_loss(Q_exp, Q_targ)
        # Apply Gradient Descent
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


