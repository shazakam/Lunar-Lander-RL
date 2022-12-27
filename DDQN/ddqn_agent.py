import random
from model import QNetwork
from replay_buffer import ReplayBuffer
import torch
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import numpy as np


class DDQNAgent:
    def __init__(
        self,
        state_size,
        action_size,
        seed,
        buffer_size=10000,
        batch_size=64,
        fc1_units=64,
        fc2_units=64,
        gamma=0.99,
        tau=1e-3,
        lr=5e-4,
        update_every=4,
        # this is used in the training loop but we want to see how a change in this can affect traning so need to be here for optune optimizer
        eps_decay=0.995,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ):
        self.num_exploitative_actions = 0
        self.eps_decay = eps_decay
        self.num_exploratory_actions = 0
        self.loss = 0
        self.device = device
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.update_every = update_every
        self.steps_done = 0
        self.online_network = QNetwork(
            state_size, action_size, self.seed, fc1_units, fc2_units
        ).to(self.device)
        self.target_network = QNetwork(
            state_size, action_size, self.seed, fc1_units, fc2_units
        ).to(self.device)
        self.optimizer = optim.Adam(self.online_network.parameters(), lr=self.lr)
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, self.seed)
        self.batch_size = batch_size
        self.update_counter = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.update_counter = (self.update_counter + 1) % self.update_every
        if self.update_counter == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def act(self, state, eps=0.0):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.online_network.eval()
        with torch.no_grad():
            action_values = self.online_network(state)
        self.online_network.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            self.num_exploratory_actions += 1
            return np.argmax(action_values.cpu().data.numpy())
        else:
            self.num_exploitative_actions += 1
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from online model
        q_online = self.online_network(next_states).detach()
        best_actions = torch.argmax(q_online, dim=1)
        q_target = self.target_network(next_states).detach()
        Q_targets_next = q_target[range(self.batch_size), best_actions]
        Q_targets_next = Q_targets_next.unsqueeze(1)

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.online_network(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        self.loss += loss
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_target_network()

    def update_target_network(self):
        """Update the target network to have the same weights as the online network."""
        for target_param, online_param in zip(
            self.target_network.parameters(), self.online_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * online_param.data + (1.0 - self.tau) * target_param.data
            )
