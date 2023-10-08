# predeined models for PPO can be found here
# the repo provides basic MLP models
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, activation):
        super(Actor, self).__init__()
        self.networks = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

    def forward(self, x):
        # get the mean of the action distribution
        return self.networks(x)
    
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim, activation):
        super(Critic, self).__init__()
        self.networks = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # get the value of the state
        return self.networks(x)
    
class ActorCritic(nn.Module):
    def __init__(self, envs, hidden_dim, activation):
        super(ActorCritic, self).__init__()
        state_dim = envs.observation_space.shape[0]
        action_dim = envs.action_space.shape[0]
        print("state_dim: ", state_dim, " action_dim: ", action_dim)
        self.actor = Actor(state_dim, action_dim, hidden_dim, activation)
        self.critic = Critic(state_dim, hidden_dim, activation)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, states):
        return self.critic(states)
    
    def get_action_n_value(self, states, actions=None):
        mean = self.actor(states)
        std = self.log_std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        if actions is None:
            actions = dist.sample()
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return actions, log_prob, entropy, self.critic(states)