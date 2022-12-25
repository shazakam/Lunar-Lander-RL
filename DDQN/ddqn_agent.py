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
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ):
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
            state_size, action_size, seed, fc1_units, fc2_units
        ).to(self.device)
        self.target_network = QNetwork(
            state_size, action_size, seed, fc1_units, fc2_units
        ).to(self.device)
        self.optimizer = optim.Adam(self.online_network.parameters(), lr=self.lr)
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed)
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
            return np.argmax(action_values.cpu().data.numpy())
        else:
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


# # Set the hyperparameters
# BUFFER_SIZE = int(1e5)  # replay buffer size
# BATCH_SIZE = 64  # minibatch size
# GAMMA = 0.99  # discount factor
# TAU = 1e-3  # for soft update of target parameters
# LR = 5e-4  # learning rate
# UPDATE_EVERY = 4  # how often to update the network

# # Create the DDQN agent
# agent = DDQNAgent(
#     state_size=8,
#     action_size=4,
#     seed=42,
#     buffer_size=BUFFER_SIZE,
#     batch_size=BATCH_SIZE,
#     lr=LR,
#     update_every=UPDATE_EVERY,
#     eps_start=1.0,
#     eps_end=0.01,
#     eps_decay=0.995,
# )

# # Set the environment
# env = gym.make("LunarLander-v2", render_mode="human")

# # Set the number of episodes and maximum time steps per episode
# n_episodes = 1000
# max_t = 1000

# # Set the epsilon decay rate
# eps_decay_rate = 1.0 / n_episodes

# # Set the scores list and the best score
# scores = []
# best_score = -np.inf

# # Loop over the number of episodes
# for i_episode in range(1, n_episodes + 1):
#     # Reset the environment and the score
#     state, _ = env.reset()
#     score = 0
#     eps = 1.0

#     # Loop over the maximum number of time steps
#     for t in range(max_t):
#         # Select an action using the epsilon-greedy policy
#         action = agent.act(state, eps)

#         # Take the action and observe the next state, reward, and done flag
#         next_state, reward, terminated, truncated, _ = env.step(action)
#         done = terminated | truncated
#         # Store the experience in the replay buffer
#         agent.memory.add(state, action, reward, next_state, done)

#         # Update the online network using the DDQN algorithm
#         agent.update_target_network()

#         # Update the score
#         score += reward

#         # Set the state to the next state
#         state = next_state

#         # If the episode is done, break the loop
#         if done:
#             break

#     # Update the eps
#     # Update the epsilon value
#     eps -= eps_decay_rate
#     eps = max(eps, 0.01)

#     # Append the score to the scores list
#     scores.append(score)

#     # Print the episode and score
#     print(f"Episode {i_episode}\tScore: {score:.2f}")

#     # Update the best score
#     if score > best_score:
#         best_score = score

#     # Check if the mean of the last 100 scores is greater than or equal to 195
#     if np.mean(scores[-100:]) >= 195.0:
#         print(
#             "\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}".format(
#                 i_episode, np.mean(scores[-100:])
#             )
#         )
#         break
