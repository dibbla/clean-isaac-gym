# predeined models for PPO can be found here
# the repo provides basic MLP models
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, activation):
        super(Actor, self).__init__()
        self.networks = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_dim)),
            activation(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            activation(),
            layer_init(nn.Linear(hidden_dim, action_dim), std=0.01),
        )

    def forward(self, x):
        # get the mean of the action distribution
        return self.networks(x)
    
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim, activation):
        super(Critic, self).__init__()
        self.networks = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_dim)),
            activation(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            activation(),
            layer_init(nn.Linear(hidden_dim, 1), std=1)
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