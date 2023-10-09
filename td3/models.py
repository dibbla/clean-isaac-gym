import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, envs, hidden_dim, activation):
        super(Actor, self).__init__()

        # this actor supports only 1D obs space
        state_dim = envs.observation_space.shape[0]
        action_dim = envs.action_space.shape[0]

        self.networks = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

        # non-learnable parameters for scaling
        self.register_buffer(
            "action_scale", torch.tensor((envs.action_space.high - envs.action_space.low)/2, dtype=torch.float32),
        )
        self.register_buffer(
            "action_bias", torch.tensor((envs.action_space.high + envs.action_space.low)/2, dtype=torch.float32),
        )
    
    def forward(self, x):
        actions = self.networks(x)
        actions = actions * self.action_scale + self.action_bias
        return actions
    
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