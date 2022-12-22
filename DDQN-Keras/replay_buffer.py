import numpy as np


class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions, discrete=False) -> None:
        self.memory_size = max_size
        self.memory_counter = 0
        self.discrete = discrete
        self.memory_state = np.zeros((self.memory_size, input_shape))
        self.new_memory_state = np.zeros((self.memory_size, input_shape))
        # if continuous action space, actions are not indices but real valued numbers
        self.action_memory = np.zeros(
            (self.memory_size, n_actions),
            dtype=(np.int8 if self.discrete else np.float32),
        )
        self.reward_memory = np.zeros(self.memory_size)
        self.terminal_memory = np.zeros(self.memory_size, dtype=np.float32)

    def store(self, state, action, reward, next_state, done):
        index = self.memory_counter % self.memory_size
        self.memory_state[index] = state
        self.new_memory_state[index] = next_state
        if self.discrete:
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - int(done)
        self.memory_counter += 1

    def sample_buffer(self, batch_size):
        max_memory = min(self.memory_counter, self.memory_size)
        batch = np.random.choice(max_memory, batch_size)
        states = self.memory_state[batch]
        new_states = self.new_memory_state[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, new_states, rewards, terminal
