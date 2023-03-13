import numpy as np


class ReplayBuffer:
    def __init__(self, max_size, input_dim, n_actions=()):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size,) + input_dim, dtype=np.float16)
        self.action_memory = np.zeros((self.mem_size,) + n_actions, dtype=np.int8)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float16)
        self.next_state_memory = np.zeros((self.mem_size,) + input_dim, dtype=np.float16)
        self.done_memory = np.zeros(self.mem_size, dtype=np.bool_)

    def store_data(self, state, action, reward, next_state, done):
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.next_state_memory[index] = next_state
        self.done_memory[index] = 1 - int(done)

        self.mem_cntr += 1

    def sample_data(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        next_states = self.next_state_memory[batch]
        done = self.done_memory[batch]

        return states, actions, rewards, next_states, done
