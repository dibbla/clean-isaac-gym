import gym
from gym import spaces
import torch as th
import numpy as np
from _base_replay_buffer import ReplayBuffer
from typing import Union
from typing import NamedTuple

class ReplayBufferSamplesTorch(NamedTuple):
    observations: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    actions: th.Tensor

def check_sample(sample: ReplayBufferSamplesTorch):
    print("Sample checking")
    print(" observations shape: ", sample.observations.shape)
    print(" next_observations shape: ", sample.next_observations.shape)
    print(" dones shape: ", sample.dones.shape)
    print(" rewards shape: ", sample.rewards.shape)
    print(" actions shape: ", sample.actions.shape)

class ReplayBufferTorch(ReplayBuffer):
    """
    Tensor ReplayBuffer for PVP training with Isaac GYm
    """
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1, # in PVP case, we only have one env
        handle_timeout_termination: bool = False,
        force_action_space: spaces.Space = None,
    ):
        print("ReplayBufferTorch initilizing...")
        super(ReplayBuffer, self).__init__(buffer_size, observation_space, action_space, 
                                           device, n_envs=n_envs)
        self.obs_shape = observation_space.shape
        if force_action_space is None:
            self.action_shape = action_space.shape[0] # for goal-controller
        else:
            self.action_shape = force_action_space.shape[0]
        self.buffer_size = max(buffer_size // n_envs, 1)
        self.handle_timeout_termination = handle_timeout_termination

        self.observations = th.zeros((self.buffer_size, self.n_envs) + self.obs_shape, device=self.device)
        self.next_observations = th.zeros((self.buffer_size, self.n_envs) + self.obs_shape, device=self.device)       
        self.actions = th.zeros(self.buffer_size, self.n_envs, self.action_shape, device=self.device)
        self.rewards = th.zeros(self.buffer_size, self.n_envs, 1, device=self.device)
        self.dones = th.zeros(self.buffer_size, self.n_envs, 1, device=self.device, dtype=th.int)
        self.timeouts = th.zeros(self.buffer_size, self.n_envs, 1, device=self.device, dtype=th.int)
        
        print("   observations shape: ", self.observations.shape)
        print("   actions shape: ", self.actions.shape)
        print("   rewards shape: ", self.rewards.shape)
        print("ReplayBufferTorch initilized.")

    def add(self, obs, next_obs, action, reward, done, infos):
        # add to buffer, actions are normalized
        self.observations[self.pos].copy_(obs)
        self.next_observations[self.pos].copy_(next_obs)
        self.actions[self.pos].copy_(action)
        self.rewards[self.pos].copy_(reward.view(-1,1))
        self.dones[self.pos].copy_(done.to(th.int))
        self.timeouts[self.pos].copy_(infos['time_outs'].unsqueeze(1))

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, env = None):
        """
        Sample elements from the replay buffer.

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        results = self._get_samples(batch_inds, env=env)
        return results

    def _get_samples(self, batch_inds: np.ndarray, env=None):
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))
        obs_ = self.observations[batch_inds, env_indices, :]

        next_obs_ = self.next_observations[batch_inds, env_indices, :]

        data =  ReplayBufferSamplesTorch(
            observations=obs_,
            next_observations=next_obs_,
            dones=self.dones[batch_inds, env_indices] & ~ self.timeouts[batch_inds, env_indices],
            rewards=self.rewards[batch_inds, env_indices],
            actions=self.actions[batch_inds, env_indices], 
        )
        return data
    
