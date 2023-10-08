import os
import gym
import isaacgym
import isaacgymenvs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import yaml
from models import ActorCritic

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

    # setup network
    network_cfg = cfg["train"]["network"]
    actor_critic = ActorCritic(envs, network_cfg["hidden_dim"], nn.ReLU).to(device)

    # setup optimizer
    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=cfg["train"]["lr"])

    # setup buffers
    obs = torch.zeros((cfg["train"]["num_steps"], envs.num_envs) + envs.observation_space.shape).to(device)
    actions = torch.zeros((cfg["train"]["num_steps"], envs.num_envs) + envs.action_space.shape).to(device)
    log_probs = torch.zeros((cfg["train"]["num_steps"], envs.num_envs)).to(device)
    rewards = torch.zeros((cfg["train"]["num_steps"], envs.num_envs)).to(device)
    dones = torch.zeros((cfg["train"]["num_steps"], envs.num_envs)).to(device)
    values = torch.zeros((cfg["train"]["num_steps"], envs.num_envs)).to(device)
    advantages = torch.zeros_like(rewards).to(device)

    # training loop
    next_obs = envs.reset()
    next_done = torch.zeros(envs.num_envs).to(device)
    
    global_steps = 0 # total interactions with the environment
    num_steps = cfg["train"]["num_steps"]
    total_timesteps = cfg["train"]["total_timesteps"]
    batch_size = cfg["train"]["batch_size"]
    minibatch_size = cfg["train"]["minibatch_size"]
    num_updates = total_timesteps // batch_size
    print(f"num_updates={num_updates}")

    for update in range(num_updates):
        # sampling trajectories for *num_steps* steps
        for step in range(num_steps):
            obs[step] = next_obs
            dones[step] = next_done
            global_steps += envs.num_envs

            with torch.no_grad():
                action, log_prob, _, value = actor_critic.get_action_n_value(obs[step])
            actions[step], log_probs[step] = action, log_prob
            values[step] = value.squeeze(-1)

            # apply action
            next_obs, rewards[step], next_done, info = envs.step(action)
            next_obs = next_obs

            if 0 <= step <= 2:
                for idx, d in enumerate(next_done):
                    if d:
                        episodic_return = info["r"][idx].item()
                        print(f"global_step={global_steps}, episodic_return={episodic_return}")
                        writer.add_scalar("charts/episodic_return", episodic_return, global_steps)
                        writer.add_scalar("charts/episodic_length", info["l"][idx], global_steps)
                        break
        
        # compute advantages with GAE
        with torch.no_grad():
            next_values = actor_critic.get_value(next_obs).squeeze(-1) # value of last step in this round
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            # loop reversely
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done # if next state is non-terminal state
                    nextvalues = next_values
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + cfg["train"]["gamma"] * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + cfg["train"]["gamma"] * cfg["train"]["gae_lambda"] * nextnonterminal * lastgaelam
            returns = advantages + values
        
        # create batch
        b_obs = obs.reshape((-1,) + envs.observation_space.shape)
        b_actions = actions.reshape((-1,) + envs.action_space.shape)
        b_log_probs = log_probs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # update
        clipfracs = []
        for epoch in range(cfg["train"]["update_epochs"]):
            b_indices = torch.randperm(num_steps * envs.num_envs)
            for start in range(0, batch_size, minibatch_size):
                mb = b_indices[start:start + minibatch_size]
                # get actions and value for CURRENT policy
                _, new_log_probs, entropy, mb_values = actor_critic.get_action_n_value(b_obs[mb], b_actions[mb])
                logratio = new_log_probs - b_log_probs[mb]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > cfg["train"]["clip"]).float().mean().item()]

                mb_advantages = b_advantages[mb]

                # policy loss
                policy_loss1 = -ratio * mb_advantages
                policy_loss2 = -torch.clamp(ratio, 1.0 - cfg["train"]["clip"], 1.0 + cfg["train"]["clip"]) * mb_advantages
                policy_loss = torch.max(policy_loss1, policy_loss2).mean()

                # entropy loss
                entropy_loss = entropy.mean()

                # value loss
                new_values = mb_values.view(-1)
                if cfg["train"]["vloss_clip"]:
                    value_loss = (new_values - b_returns[mb]).pow(2)
                    value_clipped = b_values[mb] + torch.clamp(
                        new_values - b_values[mb],
                        -cfg["train"]["vloss_clip_coef"],
                        cfg["train"]["vloss_clip_coef"],
                    )
                    value_loss_clipped = (value_clipped - b_returns[mb]) ** 2
                    v_loss_max = torch.max(value_loss, value_loss_clipped)
                    value_loss = 0.5 * v_loss_max.mean()
                else:
                    value_loss = 0.5 * (new_values - b_returns[mb]).pow(2).mean()

                # total loss
                loss = policy_loss + cfg["train"]["vf_coef"] * value_loss - cfg["train"]["ent_coef"] * entropy_loss

                # update
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(actor_critic.parameters(), cfg["train"]["max_grad_norm"])
                optimizer.step()

                # target KL
                if cfg["train"]["target_kl"]:
                    if approx_kl > cfg["train"]["target_kl_coef"]:
                        break

        writer.add_scalar("train/loss", loss, global_steps)
        writer.add_scalar("train/critic_loss", cfg["train"]["vf_coef"] * value_loss, global_steps)
        writer.add_scalar("train/policy_loss", policy_loss, global_steps)

        # model saving
        if update % cfg["train"]["save_interval"] == 0:
            torch.save(actor_critic.state_dict(), os.path.join(log_path, folder_name, f"model_{update}.pth"))
    
    torch.save(actor_critic.state_dict(), os.path.join(log_path, folder_name, f"model_final.pth"))



if __name__ == "__main__":
    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    log_path = os.path.join(os.path.dirname(__file__), "runs")
    main(cfg_path, log_path)