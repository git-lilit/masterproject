import wandb
import torch
import collections
import numpy as np
import torch.optim as optim
from lib.ReplayBuffer import ReplayBuffer
from lib.sac_utils import (
    sample_sequences_sac,
    hard_update_target_network,
    PolicyNetwork,
    CriticNetwork,
)
from torch.nn import functional as F


class SACAgent:
    def __init__(self, model_params, test_fn, wandb_log, dataset, device):
        self.model_params = model_params
        self.test_fn = test_fn
        self.wandb_log = wandb_log
        self.device = device
        self.dataset = dataset

        self.q_value_network1 = CriticNetwork(**self.model_params, device=self.device)
        self.q_value_network2 = CriticNetwork(**self.model_params, device=self.device)
        self.policy_network = PolicyNetwork(**self.model_params, device=self.device)

        self.q_value_target_network1 = CriticNetwork(**self.model_params, device=self.device)
        self.q_value_target_network2 = CriticNetwork(**self.model_params, device=self.device)

        self.q_value_target_network1.load_state_dict(self.q_value_network1.state_dict())
        self.q_value_target_network1.eval()

        self.q_value_target_network2.load_state_dict(self.q_value_network2.state_dict())
        self.q_value_target_network2.eval()

        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()

    def train(self, params):
        self.memory = ReplayBuffer(buffer_limit=int(params["buffer_size"]))
        self.entropy_target = 0.98 * (-np.log(1 / params["num_actions"]))

        self.q_value1_opt = optim.Adam(self.q_value_network1.parameters(), lr=params["lr"])
        self.q_value2_opt = optim.Adam(self.q_value_network2.parameters(), lr=params["lr"])
        self.policy_opt = optim.Adam(self.policy_network.parameters(), lr=params["lr"])
        self.alpha_opt = optim.Adam([self.log_alpha], lr=params["lr"])

        all_episode_stats = collections.defaultdict(list)
        all_interval_stats = collections.defaultdict(list)
        interval_length = params["print_interval"]

        if params["training_mode"] == "offline":
            X, y = self.dataset
            self.memory.put(X, y)

        for episode in range(params["n_episodes"]):
            alpha_loss, critics_loss, actor_loss = self.train_one_episode(params)

            current_episode_stats = {
                "episode": episode,
                "actor_loss": actor_loss,
                "critics_loss": critics_loss,
                "alpha_loss": alpha_loss,
                "buffer_size": self.memory.size(),
                "max_score": self.memory.max_reward_score(),
            }

            for key, value in current_episode_stats.items():
                all_episode_stats[key].append(value)

            if episode % params["hard_update_freq"] == 0:
                hard_update_target_network(
                    self.q_value_network1, self.q_value_network2, self.q_value_target_network1, self.q_value_target_network2
                )

            if episode % interval_length == 0:
                seq_reward = self.evaluate(params)

                mean_alpha_loss_interval = (
                    sum(all_episode_stats["alpha_loss"][-interval_length:])
                    / interval_length
                )
                mean_actor_loss_interval = (
                    sum(all_episode_stats["actor_loss"][-interval_length:])
                    / interval_length
                )
                mean_critics_loss_interval = (
                    sum(all_episode_stats["critics_loss"][-interval_length:])
                    / interval_length
                )

                current_interval_stats = {
                    "seq_reward": seq_reward.item(),
                    "mean_alpha_loss_interval": mean_alpha_loss_interval,
                    "mean_actor_loss_interval": mean_actor_loss_interval,
                    "mean_critics_loss_interval": mean_critics_loss_interval,
                }

                for key, value in current_interval_stats.items():
                    all_interval_stats[key].append(value)

                self.log_stats(current_episode_stats, current_interval_stats)

        return all_episode_stats, all_interval_stats

    def train_one_episode(self, params):
        """Performs training for one episode, including gradient updates."""
        actor_losses = []
        critics_mean_losses = []
        td_errors = []

        for _ in range(params["gradient_steps"]):
            actor_loss, critics_mean_loss, td_error = self.train_step(params)

            # Append values to lists
            actor_losses.append(actor_loss)
            critics_mean_losses.append(critics_mean_loss)
            td_errors.append(td_error)

        # Compute mean loss and TD error
        mean_actor_loss = sum(actor_losses) / len(actor_losses)
        mean_critics_loss = sum(critics_mean_losses) / len(critics_mean_losses)
        mean_td_error = sum(td_errors) / len(td_errors)

        return mean_actor_loss, mean_critics_loss, mean_td_error


    def train_step(self, params): 
        torch.autograd.set_detect_anomaly(True)

        transitions = self.memory.sample_steps(
            params["batch_size"], params["fraction_best"]
        )
        states, actions, rewards, next_states, done_masks, total_rewards = transitions

        states = states.long().to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device)
        rewards = rewards.to(self.device).unsqueeze(-1)
        next_states = next_states.long().to(self.device)
        done_masks = done_masks.to(self.device).unsqueeze(-1)
        total_rewards = total_rewards.to(self.device)

        # Calculating the Q-Value target
        with torch.no_grad():
            _, next_probs = self.policy_network(next_states)
            next_log_probs = torch.log(next_probs)
            next_q1 = self.q_value_target_network1(next_states)
            next_q2 = self.q_value_target_network2(next_states)
            next_q = torch.min(next_q1, next_q2)
            next_v = (next_probs * (next_q - self.alpha * next_log_probs)).sum(-1).unsqueeze(-1)
            target_q = rewards + params["gamma"] * (1 - done_masks) * next_v

        q1 = self.q_value_network1(states).gather(1, actions.unsqueeze(1))
        q2 = self.q_value_network2(states).gather(1, actions.unsqueeze(1))
        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)

        # Calculating the Policy target
        _, probs = self.policy_network(states)
        log_probs = torch.log(probs)
        with torch.no_grad():
            q1 = self.q_value_network1(states)
            q2 = self.q_value_network2(states)
            q = torch.min(q1, q2)

        policy_loss = (probs * (self.alpha.detach() * log_probs - q)).sum(-1).mean()

        self.q_value1_opt.zero_grad()
        q1_loss.backward()
        self.q_value1_opt.step()

        self.q_value2_opt.zero_grad()
        q2_loss.backward()
        self.q_value2_opt.step()

        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()

        log_probs = (probs * log_probs).sum(-1)
        alpha_loss = -(self.log_alpha * (log_probs.detach() + self.entropy_target)).mean()

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        self.alpha = self.log_alpha.exp()

        return alpha_loss.item(), 0.5 * (q1_loss + q2_loss).item(), policy_loss.item()


    def log_stats(self, episode_stats, interval_stats):
        print(
            f"Episode N: {episode_stats["episode"]:} | Episode TD error: {episode_stats["alpha_loss"]:.3f} | "
            f"Episode actor loss: {episode_stats["actor_loss"]:.3f} | Episode critics loss: {episode_stats['critics_loss']:.3f}"
        )

        print(
            f"Buffer size: {episode_stats["buffer_size"]:} | Buffer Max Score: {episode_stats['max_score']:.3f}"
        )

        print(
            f"Generated Sequence reward mean: {interval_stats["seq_reward"]:.3f} | Mean TD error: {interval_stats["mean_alpha_loss_interval"]:.3f} | "
            f"Mean Actor Loss: {interval_stats["mean_actor_loss_interval"]:.3f} | Mean Critics Loss: {interval_stats["mean_critics_loss_interval"]:.3f}"
        )

        all_stats = {**episode_stats, **interval_stats}

        if self.wandb_log == True:
            wandb.log(all_stats)

    def evaluate(self, params):
        """Logs and evaluates the model's performance at regular intervals."""
        # Sample sequences for evaluation
        state_batch = sample_sequences_sac(
            self.policy_network,
            batch_size=100,
            start_token=params["num_actions"],
            seq_len=params["seq_len"],
            device=self.device,
            greedy=True
        )

        print(state_batch)

        rewards_mean = torch.mean(self.test_fn(state_batch))

        return rewards_mean
