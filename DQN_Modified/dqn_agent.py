import numpy as np
import random
from collections import namedtuple, deque
from model import QNetwork
import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.999            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():

    def __init__(self, state_size, action_size, seed):
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        print("state_size: {}, action_size: {}, seed: {}".format(state_size, action_size, self.seed))
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device) #seed
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device) #seed
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        
        self.t_step = 0
    
    def step(self, s, a, r, s_t, done):
        #Add the experience to the memory so we can smaple later
        self.memory.add(s, a, r, s_t, done)
        
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        #Choose action based on epsilon greedy policy as implemented
        if random.random() > eps:
            #Theoretical "best" action
            return np.argmax(action_values.cpu().data.numpy())
        else:
            #low probability random action
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        
        s, a, r, s_t, dones = experiences

        Q_targets_next = self.qnetwork_target(s_t).detach().max(1)[0].unsqueeze(1)
        
        Q_targets = r + (gamma * Q_targets_next * (1 - dones))

        Q_expected = self.qnetwork_local(s).gather(1, a)

        # Compute Huber loss
        loss = F.huber_loss(Q_expected, Q_targets)
        # Minimize the Huber loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        
        for t_par, l_par in zip(target_model.parameters(), local_model.parameters()):
            t_par.data.copy_(tau*l_par.data + (1.0-tau)*t_par.data)


class ReplayBuffer:
    #Memory Replay Buffer class to enable experience replay feature

    def __init__(self, action_size, buffer_size, batch_size, seed):
        #Instantiate object with deque object, batch size hyperparam and experience dic
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["s", "a", "r", "s_t", "done"])
        self.seed = random.seed(seed)
    
    #Add an experience into the memory
    def add(self, s, a, r, s_t, done):
        
        exp = self.experience(s, a, r, s_t, done)
        self.memory.append(exp)
    
    def sample(self):
        #Sample an experience from replay buffer
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.s for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.a for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.r for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.s_t for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        #Current capacity of replay memory
        return len(self.memory)
