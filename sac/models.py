import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN = -5
LOG_STD_MAX = 2

class Actor(nn.Module):
    def __init__(self, envs, hidden_dim, activation):
        super(Actor, self).__init__()

        # this actor supports only 1D obs space
        state_dim = envs.observation_space.shape[0]
        action_dim = envs.action_space.shape[0]

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
        )
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

        # non-learnable parameters for scaling
        self.register_buffer(
            "action_scale", torch.tensor((envs.action_space.high - envs.action_space.low)/2, dtype=torch.float32),
        )
        self.register_buffer(
            "action_bias", torch.tensor((envs.action_space.high + envs.action_space.low)/2, dtype=torch.float32),
        )
    
    def forward(self, x):
        x = self.shared(x)
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mu, log_std
    
    def get_action(self, x):
        mu, log_std = self(x)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        sampled_action = dist.rsample()
        tanh_action = torch.tanh(sampled_action)
        action = tanh_action * self.action_scale + self.action_bias

        # for soft process
        log_prob = dist.log_prob(sampled_action)

        # enforcing action bound
        log_prob -= torch.log(self.action_scale * (1 - tanh_action.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mu = torch.tanh(mu) * self.action_scale + self.action_bias
        return action, log_prob, mu

class QFunction(nn.Module):
    def __init__(self, envs, hidden_dim, activation):
        super(QFunction, self).__init__()

        # this actor supports only 1D obs space
        state_dim = envs.observation_space.shape[0]
        action_dim = envs.action_space.shape[0]

        self.networks = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x, a):
        pair = torch.cat([x, a], dim=-1)
        return self.networks(pair)