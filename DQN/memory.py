from collections import namedtuple, deque
import random
import numpy as np
import torch
import torch.nn.functional as F


class ReplayBuffer:
    
    def __init__(self, a_size, buffer_size, batch_size, seed, device):
        self.action_size = a_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device
    
    def add(self, s, a, r, s_t, done):
        exp = self.experience(s, a, r, s_t, done)
        self.memory.append(exp)
    
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        #Collects all sampled experience values in tensors and puts them in a tuple which is returned
        r = torch.from_numpy(np.vstack([exp.reward for exp in experiences if exp is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([exp.done for exp in experiences if exp is not None]).astype(np.uint8)).float().to(self.device)
        s = torch.from_numpy(np.vstack([exp.state for exp in experiences if exp is not None])).float().to(self.device)
        a = torch.from_numpy(np.vstack([exp.action for exp in experiences if exp is not None])).long().to(self.device)
        s_t = torch.from_numpy(np.vstack([exp.next_state for exp in experiences if exp is not None])).float().to(self.device)
  
        return (s, a, r, s_t, dones)

    def __len__(self):
        return len(self.memory)