import wandb
import math
import torch
import random
import collections
import numpy as np
import torch.optim as optim
from lib.replay_buffer import ReplayBuffer
from lib.replay_buffer_with_mask import ReplayBufferWithMask
from agents.sac.utils import (
    sample_sequences_sac,
    hard_update_target_network_ensemble,
    PolicyNetwork,
    CriticNetwork,
)
from torch.nn import functional as F
from lib.sequence_model_with_prior import SequenceModelWithPrior


class SACAgentEnsemble:
    def __init__(
        self,
        model_params,
        test_fn,
        wandb_log,
        dataset,
        n_networks,
        integration_type,
        device,
        bootstrapping=False,
        diversification=False,
        with_prior=False,
    ):
        self.model_params = model_params
        self.test_fn = test_fn
        self.wandb_log = wandb_log
        self.device = device
        self.dataset = dataset
        self.ensemble_size = n_networks
        self.integration_type = integration_type
        self.bootstrapping = bootstrapping
        self.diversification = diversification

        if with_prior:
            self.q_value_networks = [
                SequenceModelWithPrior(**self.model_params, device=self.device)
                for _ in range(self.ensemble_size)
            ]
            self.q_value_target_networks = [
                SequenceModelWithPrior(**self.model_params, device=self.device)
                for _ in range(self.ensemble_size)
            ]
        else:
            self.q_value_networks = [
                CriticNetwork(**self.model_params, device=self.device)
                for _ in range(self.ensemble_size)
            ]
            self.q_value_target_networks = [
                CriticNetwork(**self.model_params, device=self.device)
                for _ in range(self.ensemble_size)
            ]
            
            self.policy_networks = [
                PolicyNetwork(**self.model_params, device=self.device)
                for _ in range(self.ensemble_size)
            ]

        for q_net, q_target in zip(self.q_value_networks, self.q_value_target_networks):
            q_target.load_state_dict(q_net.state_dict())
            q_target.eval()

        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()

    def train(self, params):
        if self.bootstrapping:
            self.memory = ReplayBufferWithMask(
                buffer_limit=int(params["buffer_size"]),
                ensemble_size=self.ensemble_size,
                mask_prob=params["bernoulli_p"],
            )
        else:
            self.memory = ReplayBuffer(buffer_limit=int(params["buffer_size"]))
        self.entropy_target = 0.98 * (-np.log(1 / params["num_actions"]))

        self.q_value_optimizers = [
            optim.Adam(q_net.parameters(), lr=params["lr"])
            for q_net in self.q_value_networks
        ]
        self.policy_opts = [
            optim.Adam(policy_net.parameters(), lr=params["lr"])
            for policy_net in self.policy_networks
        ]

        self.alpha_opt = optim.Adam([self.log_alpha], lr=params["lr"])

        all_episode_stats = collections.defaultdict(list)
        all_interval_stats = collections.defaultdict(list)
        interval_length = params["print_interval"]

        if params["training_mode"] == "offline":
            X, y = self.dataset
            self.memory.put(X, y)

        for episode in range(params["n_episodes"]):
            step_stats = self.train_step(params)

            current_episode_stats = {
                "episode": episode,
                "buffer_size": self.memory.size(),
                "max_score": self.memory.max_reward_score(),
                **step_stats,
            }

            for key, value in current_episode_stats.items():
                if not math.isnan(value):
                    all_episode_stats[key].append(value)

            if episode % params["hard_update_freq"] == 0:
                hard_update_target_network_ensemble(
                    self.q_value_networks,
                    self.q_value_target_networks,
                )

            if episode % interval_length == 0:
                seq_reward, originality_score = self.evaluate(params)

                keys = list(step_stats.keys())

                mean_losses = {
                    key: sum(all_episode_stats[key][-interval_length:])
                    / interval_length
                    for key in keys
                }

                current_interval_stats = {
                    "seq_reward": seq_reward.item(),
                    "originality_score": originality_score,
                }

                current_interval_stats.update(
                    {f"mean_{key}": mean_losses[key] for key in keys}
                )

                for key, value in current_interval_stats.items():
                    all_interval_stats[key].append(value)

                self.log_stats(current_episode_stats, current_interval_stats)

        return all_episode_stats, all_interval_stats

    def train_step(self, params):
        torch.autograd.set_detect_anomaly(True)

        transitions = self.memory.sample_steps(
            params["batch_size"], params["fraction_best"]
        )
        if self.bootstrapping:
            states, actions, rewards, next_states, done_masks, total_rewards, masks = (
                transitions
            )
        else:
            states, actions, rewards, next_states, done_masks, total_rewards = (
                transitions
            )

        states = states.long().to(self.device)
        actions = actions.clone().detach().to(dtype=torch.int64, device=self.device)
        rewards = rewards.to(self.device).unsqueeze(-1)
        next_states = next_states.long().to(self.device)
        done_masks = done_masks.to(self.device).unsqueeze(-1)
        total_rewards = total_rewards.to(self.device)

        if self.bootstrapping:
            masks = masks.to(self.device)

        # Select a single Q-value network randomly for both target calculation and update
        q_losses = []

        if self.diversification:
            q_values = torch.stack(
                [q(states) for q in self.q_value_networks]
            )  # Shape: (N, batch_size, num_actions)
            mean_q = q_values.mean(dim=0)  # Shape: (batch_size, num_actions)
            mean_action = torch.gather(
                mean_q, dim=1, index=actions.long().reshape(-1, 1)
            ).squeeze(1)

            mean_action = mean_action.detach()

        for idx in range(self.ensemble_size):
            q_net = self.q_value_networks[idx]
            q_target_net = self.q_value_target_networks[idx]
            optimizer = self.q_value_optimizers[idx]

            # Calculate the Q-Value target using only the selected Q-network
            with torch.no_grad():
                _, next_probs = self.policy_networks[idx](next_states)
                next_log_probs = torch.log(next_probs)
                next_q = q_target_net(next_states)

                next_v = (
                    (next_probs * (next_q - self.alpha * next_log_probs))
                    .sum(-1)
                    .unsqueeze(-1)
                )
                target_q = rewards + params["gamma"] * (1 - done_masks) * next_v

            q_value = q_net(states).gather(1, actions.unsqueeze(1))
            if self.bootstrapping:
                current_mask = masks[:, idx]
                q_loss = ((q_value - target_q) ** 2 * current_mask).mean()
            else:
                q_loss = F.mse_loss(q_value, target_q)

            if self.diversification:
                q_loss += params["eta"] * torch.exp(-params["theta"] * torch.abs(mean_action - q_value)).mean()

            optimizer.zero_grad()
            q_loss.backward()
            optimizer.step()

            q_losses.append(q_loss.item())

        avg_q_loss = sum(q_losses) / len(q_losses)

        # Calculating the Policy target
        for idx in range(self.ensemble_size):
            _, probs = self.policy_networks[idx](states)
            log_probs = torch.log(probs)
            with torch.no_grad():
                q_values_stack = torch.stack([q(states) for q in self.q_value_networks])

                match self.integration_type:
                    case "min":
                        final_q_estimate, _ = torch.min(q_values_stack, dim=0)
                    case "var":
                        mean_qs = torch.mean(q_values_stack, dim=0)
                        std_qs = torch.std(q_values_stack, dim=0)

                        final_q_estimate = mean_qs - params["beta"] * std_qs

            policy_loss = (
                (probs * (self.alpha.detach() * log_probs - final_q_estimate))
                .sum(-1)
                .mean()
            )

            self.policy_opts[idx].zero_grad()
            policy_loss.backward()
            self.policy_opts[idx].step()

        log_probs = (probs * log_probs).sum(-1)
        alpha_loss = -(
            self.log_alpha * (log_probs.detach() + self.entropy_target)
        ).mean()

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        self.alpha = self.log_alpha.exp()

        stats = {
            "alpha_loss": alpha_loss.item(),
            "critics_loss": avg_q_loss,
            "actor_loss": policy_loss.item(),
        }

        return stats

    def log_stats(self, episode_stats, interval_stats):
        print(
            f"Episode N: {episode_stats["episode"]:} | Episode TD error: {episode_stats["alpha_loss"]:.3f} | "
            f"Episode actor loss: {episode_stats["actor_loss"]:.3f} | Episode critics loss: {episode_stats['critics_loss']:.3f}"
        )

        print(
            f"Buffer size: {episode_stats["buffer_size"]:} | Buffer Max Score: {episode_stats['max_score']:.3f}"
        )

        print(
            f"Generated Sequence reward mean: {interval_stats["seq_reward"]:.3f} | Mean TD error: {interval_stats["mean_alpha_loss"]:.3f} | "
            f"Mean Actor Loss: {interval_stats["mean_actor_loss"]:.3f} | Mean Critics Loss: {interval_stats["mean_critics_loss"]:.3f}"
        )

        # all_stats = {**episode_stats, **interval_stats}

        if self.wandb_log == True:
            wandb.log(interval_stats)

    def evaluate(self, params):
        """Logs and evaluates the model's performance at regular intervals."""
        # Sample sequences for evaluation
        state_batch = sample_sequences_sac(
            self.policy_networks,
            batch_size=100,
            start_token=params["num_actions"],
            seq_len=params["seq_len"],
            device=self.device,
            greedy=True,
        )

        rewards_mean = torch.mean(self.test_fn(state_batch))
        originality_score = self.memory.originality_score(state_batch)

        return rewards_mean, originality_score
