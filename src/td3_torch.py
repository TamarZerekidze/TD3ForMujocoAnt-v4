import numpy as np
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import os
import gym

sample_env = gym.make('Ant-v4')

MAX_BUF_SIZE = 1000000
OBSERVATION_SHAPE = sample_env.observation_space.shape[0]
ACTION_SHAPE = sample_env.action_space.shape[0]


class ReplayBuffer:
    def __init__(self, max_buffer_size=MAX_BUF_SIZE, observation_shape=OBSERVATION_SHAPE, action_shape=ACTION_SHAPE):
        self.mem_size = max_buffer_size
        self.mem_cnt = 0

        self.states = np.zeros((self.mem_size, observation_shape))
        self.new_states = np.zeros(
            (self.mem_size, observation_shape))
        self.actions = np.zeros((self.mem_size, action_shape))
        self.rewards = np.zeros(self.mem_size)
        self.terminals = np.zeros(self.mem_size,
                                  dtype=np.bool_)

    def add_experience(self, state, action, reward, new_state, done):
        index = self.mem_cnt % self.mem_size

        self.states[index] = state
        self.new_states[index] = new_state
        self.terminals[index] = done
        self.rewards[index] = reward
        self.actions[index] = action

        self.mem_cnt += 1

    def sample_batch(self, batch_size):
        max_mem = min(self.mem_cnt, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        return self.states[batch], self.actions[batch], self.rewards[batch], self.new_states[batch], self.terminals[
            batch]


BETA = 0.001
LAYER1 = 400
LAYER2 = 300
RANDOM_NAME = 'no_name'
CHECK_DIR = 'td3'


class CriticNetwork(nn.Module):
    def __init__(self, beta=BETA, observation_shape=OBSERVATION_SHAPE, layer1_dim=LAYER1, layer2_dim=LAYER2,
                 action_shape=ACTION_SHAPE, name=RANDOM_NAME, checkpoint_dir=CHECK_DIR):
        super(CriticNetwork, self).__init__()
        self.checkpoint_file = os.path.join(checkpoint_dir, name)

        self.layer1 = nn.Linear(observation_shape + action_shape, layer1_dim)
        self.layer2 = nn.Linear(layer1_dim, layer2_dim)
        self.q1 = nn.Linear(layer2_dim, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        q1_action_value = self.layer1(T.cat([state, action], dim=1))
        q1_action_value = F.relu(q1_action_value)
        q1_action_value = self.layer2(q1_action_value)
        q1_action_value = F.relu(q1_action_value)

        q1 = self.q1(q1_action_value)

        return q1

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))


ALPHA = 0.001


class ActorNetwork(nn.Module):
    def __init__(self, alpha=ALPHA, observation_shape=OBSERVATION_SHAPE, layer1_dim=LAYER1, layer2_dim=LAYER2,
                 action_shape=ACTION_SHAPE, name=RANDOM_NAME, checkpoint_dir=CHECK_DIR):
        super(ActorNetwork, self).__init__()
        self.checkpoint_file = os.path.join(checkpoint_dir, name)

        self.layer1 = nn.Linear(observation_shape, layer1_dim)
        self.layer2 = nn.Linear(layer1_dim, layer2_dim)
        self.mu = nn.Linear(layer2_dim, action_shape)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        prob = self.layer1(state)
        prob = F.relu(prob)
        prob = self.layer2(prob)
        prob = F.relu(prob)

        mu = T.tanh(self.mu(prob))

        return mu

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))
