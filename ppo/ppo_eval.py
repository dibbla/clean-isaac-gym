# In this file, we can use PPO to collect expert data

import os
import yaml
import gym
import isaacgym
import isaacgymenvs
import torch
import torch.nn as nn
from models import ActorCritic

class ActorCriticEval(ActorCritic):

    def __init__(self, envs, hidden_dim, activation):
        super().__init__(envs, hidden_dim, activation)
    
    def get_eval_action(self, states):
        mean = self.actor(states)
        return mean
    
class RecordEpisodeStatisticsTorch(gym.Wrapper):
    def __init__(self, env, device):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.device = device
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.episode_returns = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.episode_lengths = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.returned_episode_returns = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.returned_episode_lengths = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        return observations

    def step(self, action):
        observations, rewards, dones, infos = super().step(action)
        self.episode_returns += rewards
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= 1 - dones
        self.episode_lengths *= 1 - dones
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        return (
            observations,
            rewards,
            dones,
            infos,
        )

class ExtractObsWrapper(gym.ObservationWrapper):
    def observation(self, obs):
        return obs["obs"]
    
def main(cfg_path, log_path):
    # load config
    cfg = yaml.safe_load(open(cfg_path, "r"))
    device = torch.device(cfg["train"]["device"])

    # create environment
    env_cfg = cfg["train"]["env"]
    envs = isaacgymenvs.make(
        seed=env_cfg["seed"], 
        task=env_cfg["task"], 
        num_envs=env_cfg["num_envs"], 
        sim_device=env_cfg["sim_device"],
        rl_device=env_cfg["rl_device"],
        graphics_device_id=env_cfg["graphics_device_id"],
        headless=env_cfg["headless"],
        multi_gpu=env_cfg["multi_gpu"],
        force_render=env_cfg["force_render"],
    )
    envs = RecordEpisodeStatisticsTorch(ExtractObsWrapper(envs), device)

    # setup network
    network_cfg = cfg["train"]["network"]
    actor_critic = ActorCriticEval(envs, network_cfg["hidden_dim"], nn.ReLU).to(device)

    # load model
    model_path = os.path.join(log_path, cfg["eval"]["task"], "1", cfg["eval"]["model_id"] + ".pth")
    actor_critic.load_state_dict(torch.load(model_path))

    # set up buffer for obs, action, reward
    collect_size = 500
    obs_buffer = torch.zeros((collect_size, envs.num_envs) + envs.observation_space.shape).to(device)
    actions_buffer = torch.zeros((collect_size, envs.num_envs) + envs.action_space.shape).to(device)
    rewards_buffer = torch.zeros((collect_size, envs.num_envs)).to(device)

    # prepare for run
    obs = envs.reset()
    done = False
    actor_critic.eval()

    for step in range(collect_size):
        action = actor_critic.get_eval_action(obs)
        obs, reward, done, info = envs.step(action)
        print(reward)

        # save data
        obs_buffer[step] = obs.clone()
        actions_buffer[step] = action.clone()
        rewards_buffer[step] = reward.clone()

    # flatten the data
    obs_buffer = obs_buffer.view(-1, obs_buffer.shape[-1])
    actions_buffer = actions_buffer.view(-1, actions_buffer.shape[-1])
    rewards_buffer = rewards_buffer.view(-1)

    print(obs_buffer.shape
        , actions_buffer.shape
        , rewards_buffer.shape
    )

    # put to CPU and save
    obs_buffer = obs_buffer.cpu()
    actions_buffer = actions_buffer.cpu()
    rewards_buffer = rewards_buffer.cpu()
    save_path = os.path.join(log_path, f"eval.{cfg['eval']['task']}.{cfg['train']['train_name']}")

    # create a buffer directory and save
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(obs_buffer, os.path.join(save_path, "obs.pth"))
    torch.save(actions_buffer, os.path.join(save_path, "actions.pth"))
    torch.save(rewards_buffer, os.path.join(save_path, "rewards.pth"))


if __name__ == "__main__":
    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    log_path = os.path.join(os.path.dirname(__file__), "runs")
    main(cfg_path, log_path)