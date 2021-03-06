import copy
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from pathlib import Path
import sys
import random
import math

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from flatland_sutd import RvNN, tree_obs_expand, RAdam

# Hyperparameters
from reinforcement_learning.policy import LearningPolicy
from reinforcement_learning.replay_buffer import ReplayBuffer

# https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html

class EpisodeBuffers:
    def __init__(self):
        self.reset()

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def reset(self):
        self.memory = {}

    def get_transitions(self, handle):
        return self.memory.get(handle, [])

    def push_transition(self, handle, transition):
        transitions = self.get_transitions(handle)
        transitions.append(transition)
        self.memory.update({handle: transitions})


class ActorCriticModel(nn.Module):
    def __init__(self, state_size, action_size, device, hidsize1=512, hidsize2=256):
        super(ActorCriticModel, self).__init__()
        self.device = device

        self.common = nn.Sequential(
            nn.Linear(state_size, hidsize2),
            nn.Tanh(),
            nn.Linear(hidsize2, hidsize2),
            nn.Tanh(),
        ).to(self.device)

        self.actor = nn.Sequential(
            nn.Linear(hidsize2, action_size),
            nn.Softmax(dim=-1)
        ).to(self.device)

        self.critic = nn.Sequential(
            nn.Linear(hidsize2, 1)
        ).to(self.device)

    def forward(self, x):
        raise NotImplementedError

    def get_actor_dist(self, state):
        common_state = self.common(state)
        action_probs = self.actor(common_state)

        try:
            dist = Categorical(action_probs)
        except Exception as e:
            print(e)
            print(action_probs)
            print(state)
            print(common_state)

        return dist

    def evaluate(self, states, actions):
        common_states = self.common(states)
        action_probs = self.actor(common_states)

        try:
            dist = Categorical(action_probs)
        except Exception as e:
            print(e)
            print(action_probs)
            print(states)

        action_logprobs = dist.log_prob(actions)

        dist_entropy = dist.entropy()
        state_value = self.critic(common_states)

        return action_logprobs, torch.squeeze(state_value), dist_entropy

    def save(self, filename):
        print("Saving model from checkpoint:", filename)
        torch.save(self.common.state_dict(), filename + ".common")
        torch.save(self.actor.state_dict(), filename + ".actor")
        torch.save(self.critic.state_dict(), filename + ".critic")

    def _load(self, obj, filename):
        if os.path.exists(filename):
            print(' >> ', filename)
            try:
                obj.load_state_dict(torch.load(filename, map_location=self.device))
            except:
                print(" >> failed!")
        return obj

    def load(self, filename):
        print("load model from file", filename)
        self.common = self._load(self.common, filename + ".common")
        self.actor = self._load(self.actor, filename + ".actor")
        self.critic = self._load(self.critic, filename + ".critic")


class PPOPolicy(LearningPolicy):
    def __init__(self, state_size, action_size, use_replay_buffer=False, in_parameters=None):
        print(">> PPOPolicy")
        super(PPOPolicy, self).__init__()
        # parameters
        self.ppo_parameters = in_parameters
        if self.ppo_parameters is not None:
            self.hidsize = self.ppo_parameters.hidden_size
            self.buffer_size = self.ppo_parameters.buffer_size
            self.batch_size = self.ppo_parameters.batch_size
            self.learning_rate = self.ppo_parameters.learning_rate
            self.gamma = self.ppo_parameters.gamma
            # Device
            if self.ppo_parameters.use_gpu and torch.cuda.is_available():
                self.device = torch.device("cuda:0")
                # print("???? Using GPU")
            else:
                self.device = torch.device("cpu")
                # print("???? Using CPU")
        else:
            self.hidsize = 128
            self.learning_rate = 1.0e-3
            self.gamma = 0.99
            self.buffer_size = 32_000
            self.batch_size = 1024
            self.device = torch.device("cpu")

        self.surrogate_eps_clip = 0.2
        self.K_epoch = 40
        self.weight_loss = 0.5
        self.weight_entropy = 0.01

        self.buffer_min_size = 0
        self.use_replay_buffer = use_replay_buffer

        self.current_episode_memory = EpisodeBuffers()
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, self.device)
        self.loss = 0
        self.actor_critic_model = ActorCriticModel(state_size, action_size,self.device,
                                                   hidsize1=self.hidsize,
                                                   hidsize2=self.hidsize)

        self.rvnn_model = RvNN(24, 12, 12, tree_obs_expand, device=self.device)

        self.learning_rate_actor = self.learning_rate
        self.learning_rate_critic = self.learning_rate*5
        self.learning_rate_rvnn = self.learning_rate*3

        self.optimizer = optim.RMSprop(
            [{'params': self.rvnn_model.parameters(), 'lr': self.learning_rate_rvnn},
             {'params': self.actor_critic_model.common.parameters(), 'lr': self.learning_rate_actor},
             {'params': self.actor_critic_model.actor.parameters(), 'lr': self.learning_rate_actor},
             {'params': self.actor_critic_model.critic.parameters(), 'lr': self.learning_rate_critic}]
        )
        self.loss_function = nn.MSELoss()  # nn.SmoothL1Loss()

    def reset(self, env):
        pass

    def rvnn(self, node):
        return self.rvnn_model(node)

    def act(self, handle, state, eps=None):
        # sample a action to take
        torch_state = torch.tensor(state, dtype=torch.float).to(self.device)
        dist = self.actor_critic_model.get_actor_dist(torch_state)

        return torch.argmax(dist.probs).item()

        action = dist.sample()
        return action.item()

    def step(self, handle, state, action, reward, next_state, done):
        # record transitions ([state] -> [action] -> [reward, next_state, done])
        torch_action = torch.tensor(action, dtype=torch.float).to(self.device)
        torch_state = torch.tensor(state, dtype=torch.float).to(self.device)
        # evaluate actor
        dist = self.actor_critic_model.get_actor_dist(torch_state)
        action_logprobs = dist.log_prob(torch_action)
        transition = (state, action, reward, next_state, action_logprobs.item(), done)
        self.current_episode_memory.push_transition(handle, transition)

    def _push_transitions_to_replay_buffer(self,
                                           state_list,
                                           action_list,
                                           reward_list,
                                           state_next_list,
                                           done_list,
                                           prob_a_list):
        for idx in range(len(reward_list)):
            state_i = state_list[idx]
            action_i = action_list[idx]
            reward_i = reward_list[idx]
            state_next_i = state_next_list[idx]
            done_i = done_list[idx]
            prob_action_i = prob_a_list[idx]
            self.memory.add(state_i, action_i, reward_i, state_next_i, done_i, prob_action_i)

    def _convert_transitions_to_torch_tensors(self, transitions_array):
        # build empty lists(arrays)
        state_list, action_list, reward_list, state_next_list, prob_a_list, done_list = [], [], [], [], [], []

        # set discounted_reward to zero
        discounted_reward = 0
        for transition in transitions_array[::-1]:
            state_i, action_i, reward_i, state_next_i, prob_action_i, done_i = transition

            state_list.insert(0, state_i)
            action_list.insert(0, action_i)
            done_list.insert(0, int(done_i))

            discounted_reward = reward_i + self.gamma * discounted_reward # * (1.0-int(done_i))

            reward_list.insert(0, discounted_reward)
            state_next_list.insert(0, state_next_i)
            prob_a_list.insert(0, prob_action_i)

        if self.use_replay_buffer:
            self._push_transitions_to_replay_buffer(state_list, action_list,
                                                    reward_list, state_next_list,
                                                    done_list, prob_a_list)

        # convert data to torch tensors
        states, actions, rewards, states_next, dones, prob_actions = \
            torch.tensor(state_list, dtype=torch.float).to(self.device), \
            torch.tensor(action_list).to(self.device), \
            torch.tensor(reward_list, dtype=torch.float).to(self.device), \
            torch.tensor(state_next_list, dtype=torch.float).to(self.device), \
            torch.tensor(done_list, dtype=torch.float).to(self.device), \
            torch.tensor(prob_a_list).to(self.device)

        return states, actions, rewards, states_next, dones, prob_actions

    def _get_transitions_from_replay_buffer(self, states, actions, rewards, states_next, dones, probs_action):
        if len(self.memory) > self.buffer_min_size and len(self.memory) > self.batch_size:
            states, actions, rewards, states_next, dones, probs_action = self.memory.sample()
            actions = torch.squeeze(actions)
            rewards = torch.squeeze(rewards)
            states_next = torch.squeeze(states_next)
            dones = torch.squeeze(dones)
            probs_action = torch.squeeze(probs_action)
        return states, actions, rewards, states_next, dones, probs_action

    def train_net(self):
        self.optimizer.zero_grad()

        # All agents have to propagate their experiences made during past episode
        for handle in range(len(self.current_episode_memory)):
            # Extract agent's episode history (list of all transitions)
            agent_episode_history = self.current_episode_memory.get_transitions(handle)
            if len(agent_episode_history) > 0:
                # Convert the replay buffer to torch tensors (arrays)
                states, actions, rewards, states_next, dones, probs_action = \
                    self._convert_transitions_to_torch_tensors(agent_episode_history)

                # Optimize policy for K epochs:
                for k_loop in range(int(self.K_epoch)):

                    if self.use_replay_buffer:
                        states, actions, rewards, states_next, dones, probs_action = \
                            self._get_transitions_from_replay_buffer(
                                states, actions, rewards, states_next, dones, probs_action
                            )

                    # Evaluating actions (actor) and values (critic)
                    logprobs, state_values, dist_entropy = self.actor_critic_model.evaluate(states, actions)

                    # Finding the ratios (pi_thetas / pi_thetas_replayed):
                    ratios = torch.exp(logprobs - probs_action.detach())

                    # Finding Surrogate Loos
                    advantages = rewards - state_values.detach()
                    surr1 = ratios * advantages
                    surr2 = torch.clamp(ratios, 1. - self.surrogate_eps_clip, 1. + self.surrogate_eps_clip) * advantages

                    # The loss function is used to estimate the gardient and use the entropy function based
                    # heuristic to penalize the gradient function when the policy becomes deterministic this would let
                    # the gradient becomes very flat and so the gradient is no longer useful.
                    loss = \
                        - torch.min(surr1, surr2) \
                        + self.weight_loss * self.loss_function(state_values, rewards) \
                        - self.weight_entropy * dist_entropy

                    # Make a gradient step
                    self.optimizer.zero_grad()
                    loss.mean().backward()
                    self.optimizer.step()

                    # Transfer the current loss to the agents loss (information) for debug purpose only
                    self.loss = loss.mean().detach().cpu().numpy()

        # Reset all collect transition data
        self.current_episode_memory.reset()

    def end_episode(self, train):
        if train:
            self.train_net()
        else:
            self.current_episode_memory.reset()

    # Checkpointing methods
    def save(self, filename):
        # print("Saving model from checkpoint:", filename)
        self.actor_critic_model.save(filename)
        torch.save(self.optimizer.state_dict(), filename + ".optimizer")
        torch.save(self.rvnn_model.state_dict(), filename + ".rvnn")

    def _load(self, obj, filename):
        if os.path.exists(filename):
            print(' >> ', filename)
            try:
                obj.load_state_dict(torch.load(filename, map_location=self.device))
            except:
                print(" >> failed!")
        else:
            print(" >> file not found!")
        return obj

    def load(self, filename):
        print("load policy from file", filename)
        self.actor_critic_model.load(filename)
        self.rvnn_model = self._load(self.rvnn_model, filename + ".rvnn")

        print("load optimizer from file", filename)
        self.optimizer = self._load(self.optimizer, filename + ".optimizer")

    def clone(self):
        policy = PPOPolicy(self.state_size, self.action_size)
        policy.actor_critic_model = copy.deepcopy(self.actor_critic_model)
        policy.optimizer = copy.deepcopy(self.optimizer)
        return self
