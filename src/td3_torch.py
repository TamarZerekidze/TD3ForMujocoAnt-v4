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


TAU = 0.005
GAMMA = 0.99
UPDATE_INT = 2
XP = 1000
BATCH = 100
NOISE = 0.1
ACT_MAX = sample_env.action_space.high
ACT_MIN = sample_env.action_space.low


class Agent:
    def __init__(self, tau=TAU, max_action=ACT_MAX, min_action=ACT_MIN,
                 gamma=GAMMA, update_interval=UPDATE_INT, xp_time=XP,
                 action_shape=ACTION_SHAPE, batch_size=BATCH, noise=NOISE):
        self.gamma = gamma
        self.tau = tau
        self.noise = noise
        self.buffer = ReplayBuffer()
        self.batch_size = batch_size

        self.actor = ActorNetwork(name='actor')
        self.target_actor = ActorNetwork(name='target_actor')
        self.critic_1 = CriticNetwork(name='critic_1')
        self.critic_2 = CriticNetwork(name='critic_2')
        self.target_critic_1 = CriticNetwork(name='target_critic_1')
        self.target_critic_2 = CriticNetwork(name='target_critic_2')

        self.xp_time = xp_time
        self.action_shape = action_shape
        self.update_interval = update_interval
        self.max_action = max_action
        self.min_action = min_action
        self.learn_step_cnt = 0
        self.time_step = 0

        self.update_network_parameters(tau=1)

    def get_action(self, state):
        if self.time_step >= self.xp_time:
            action = self.actor.forward(T.tensor(state, dtype=T.float).to(self.actor.device)).to(self.actor.device)
        else:
            action = T.tensor(np.random.normal(scale=self.noise, size=(self.action_shape,)))

        if self.xp_time != 0:
            action = action + T.tensor(np.random.normal(scale=self.noise), dtype=T.float).to(self.actor.device)

        action = T.clamp(action, self.min_action[0], self.max_action[0])
        self.time_step += 1

        return action.cpu().detach().numpy()

    def store_xp(self, state, action, reward, new_state, done):
        self.buffer.add_experience(state, action, reward, new_state, done)

    def batch_to_device(self, batch):
        state = T.tensor(batch[0], dtype=T.float).to(self.critic_1.device)
        action = T.tensor(batch[1], dtype=T.float).to(self.critic_1.device)
        reward = T.tensor(batch[2], dtype=T.float).to(self.critic_1.device)
        new_state = T.tensor(batch[3], dtype=T.float).to(self.critic_1.device)
        done = T.tensor(batch[4]).to(self.critic_1.device)

        return state, action, reward, new_state, done

    def update_critics(self, target, q1_val, q2_val):
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        q1_loss = F.mse_loss(target, q1_val)
        q2_loss = F.mse_loss(target, q2_val)
        critic_loss = q1_loss + q2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

    def update_actor(self, state):
        self.actor.optimizer.zero_grad()
        actor_q1_loss = self.critic_1.forward(state, self.actor.forward(state))
        actor_loss = -T.mean(actor_q1_loss)
        actor_loss.backward()
        self.actor.optimizer.step()
        self.update_network_parameters()

    def learn(self):
        if self.buffer.mem_cnt < self.batch_size:
            return
        state, action, reward, new_state, done = self.batch_to_device(self.buffer.sample_batch(self.batch_size))
        target_actions = (self.target_actor.forward(new_state) +
                          T.clamp(T.tensor(np.random.normal(scale=0.2)), -0.5, 0.5))
        target_actions = T.clamp(target_actions, self.min_action[0], self.max_action[0])

        new_q1_val = self.target_critic_1.forward(new_state, target_actions)
        new_q2_val = self.target_critic_2.forward(new_state, target_actions)

        q1_val = self.critic_1.forward(state, action)
        q2_val = self.critic_2.forward(state, action)

        new_q1_val[done] = 0.0
        new_q1_val = new_q1_val.view(-1)
        new_q2_val[done] = 0.0
        new_q2_val = new_q2_val.view(-1)

        min_q_val = T.min(new_q1_val, new_q2_val)

        target = reward + self.gamma * min_q_val
        target = target.view(self.batch_size, 1)

        self.update_critics(target, q1_val, q2_val)
        self.learn_step_cnt += 1

        if self.learn_step_cnt % self.update_interval == 0:
            self.update_actor(state)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = dict(self.actor.named_parameters())
        target_actor_params = dict(self.target_actor.named_parameters())
        critic_1_params = dict(self.critic_1.named_parameters())
        critic_2_params = dict(self.critic_2.named_parameters())
        target_critic_1_params = dict(self.target_critic_1.named_parameters())
        target_critic_2_params = dict(self.target_critic_2.named_parameters())

        for name in critic_1_params:
            critic_1_params[name] = (tau * critic_1_params.get(name).clone() +
                                     (1 - tau) * target_critic_1_params.get(name).clone())
        for name in critic_2_params:
            critic_2_params[name] = (tau * critic_2_params.get(name).clone() +
                                     (1 - tau) * target_critic_2_params.get(name).clone())
        for name in actor_params:
            actor_params[name] = (tau * actor_params.get(name).clone() +
                                  (1 - tau) * target_actor_params.get(name).clone())

        self.target_critic_1.load_state_dict(critic_1_params)
        self.target_critic_2.load_state_dict(critic_2_params)
        self.target_actor.load_state_dict(actor_params)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.target_critic_1.load_checkpoint()
        self.target_critic_2.load_checkpoint()
