import os
import gym
import isaacgym
import isaacgymenvs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import yaml
from models import Actor, QFunction
from replay_buffer import ReplayBufferTorch, check_sample

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

    # setup logging
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=False)
    folder_names = []
    for item in os.listdir(log_path):
        if os.path.isdir(os.path.join(log_path, item)) and item.isdigit():
            folder_names.append(int(item))
    folder_name = str(max(folder_names) + 1) if folder_names else "1"
    writer = SummaryWriter(os.path.join(log_path,folder_name))

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

    # setup models and target model
    network_cfg = cfg["train"]["network"]
    actor = Actor(envs, network_cfg["hidden_dim"], nn.ReLU).to(device)
    qf1 = QFunction(envs, network_cfg["hidden_dim"], nn.ReLU).to(device)
    qf2 = QFunction(envs, network_cfg["hidden_dim"], nn.ReLU).to(device)
    target_actor = Actor(envs, network_cfg["hidden_dim"], nn.ReLU).to(device)
    target_qf1 = QFunction(envs, network_cfg["hidden_dim"], nn.ReLU).to(device)
    target_qf2 = QFunction(envs, network_cfg["hidden_dim"], nn.ReLU).to(device)
    target_actor.load_state_dict(actor.state_dict())
    target_qf1.load_state_dict(qf1.state_dict())
    target_qf2.load_state_dict(qf2.state_dict())

    # setup optimizers
    q_optimizer = torch.optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=cfg["train"]["qf_lr"])
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=cfg["train"]["ac_lr"])

    # setup replay buffer
    buffer_cfg = cfg["train"]["buffer"]
    rb = ReplayBufferTorch(
        buffer_size=buffer_cfg["size"],
        observation_space=envs.observation_space,
        action_space=envs.action_space,
        device=device,
        n_envs=envs.num_envs,
    )

    # training loop
    exploration_noise = cfg["train"]["exploration_noise"]
    action_low, action_high = torch.tensor(envs.action_space.low).to(device), torch.tensor(envs.action_space.high).to(device)
    obs = envs.reset()

    for global_steps in range(cfg["train"]["total_timesteps"]):

        with torch.no_grad():
            actions = actor(obs).to(device)
            actions += torch.normal(0, actor.action_scale * exploration_noise)
            actions = torch.clamp(actions, action_low, action_high)
        
        # apply action to the environment
        next_obs, rewards, dones, infos = envs.step(actions)
        rb.add(obs, next_obs, actions, rewards.unsqueeze(1), dones.unsqueeze(1), infos)
        obs = next_obs

        # log reward
        for idx, d in enumerate(dones):
            if d:
                episodic_return = infos["r"][idx].item()
                print(f"global_step={global_steps}, episodic_return={episodic_return}")
                writer.add_scalar("charts/episodic_return", episodic_return, global_steps)
                writer.add_scalar("charts/episodic_length", infos["l"][idx], global_steps)
                break     


        # train
        data = rb.sample(cfg["train"]["batch_size"])
        with torch.no_grad():
            clipped_noise = (torch.randn_like(data.actions, device=device) * cfg["train"]["policy_noise"]).clamp(
                        -cfg["train"]["noise_clip"], cfg["train"]["noise_clip"]
                    ) * target_actor.action_scale
            next_state_actions = (target_actor(data.next_observations) + clipped_noise).clamp(action_low, action_high)
            qf1_next_target = target_qf1(data.next_observations, next_state_actions)
            qf2_next_target = target_qf2(data.next_observations, next_state_actions)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
            next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * cfg["train"]["gamma"] * (min_qf_next_target).view(-1)
        
        # building q loss
        qf1_action_values = qf1(data.observations, data.actions).view(-1)
        qf2_action_values = qf2(data.observations, data.actions).view(-1)
        qf1_loss = F.mse_loss(qf1_action_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_action_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        # optimize the model
        q_optimizer.zero_grad()
        qf_loss.backward()
        q_optimizer.step()

        # delayed policy update
        if global_steps % cfg["train"]["policy_freq"] == 0:
            actor_loss = -qf1(data.observations, actor(data.observations)).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # update target networks
            for target_param, param in zip(target_actor.parameters(), actor.parameters()):
                target_param.data.copy_(cfg["train"]["tau"] * param.data + (1 - cfg["train"]["tau"]) * target_param.data)
            for target_param, param in zip(target_qf1.parameters(), qf1.parameters()):
                target_param.data.copy_(cfg["train"]["tau"] * param.data + (1 - cfg["train"]["tau"]) * target_param.data)
            for target_param, param in zip(target_qf2.parameters(), qf2.parameters()):
                target_param.data.copy_(cfg["train"]["tau"] * param.data + (1 - cfg["train"]["tau"]) * target_param.data)

if __name__ == "__main__":
    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    log_path = os.path.join(os.path.dirname(__file__), "runs")
    main(cfg_path, log_path)