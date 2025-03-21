import math
import wandb
import torch
import collections
import numpy as np
import torch.optim as optim
from torch.nn import functional as F
from lib.replay_buffer import ReplayBuffer
from agents.sac.utils import (
    sample_sequences_sac,
    hard_update_target_network,
    PolicyNetwork,
    CriticNetwork,
)


class SACAgent:
    def __init__(self, model_params, test_fn, wandb_log, dataset, device):
        self.model_params = model_params
        self.test_fn = test_fn
        self.wandb_log = wandb_log
        self.device = device
        self.dataset = dataset

        self.policy_network = PolicyNetwork(**self.model_params, device=self.device)
        self.q_value_network1 = CriticNetwork(**self.model_params, device=self.device)
        self.q_value_network2 = CriticNetwork(**self.model_params, device=self.device)

        self.q_value_target_network1 = CriticNetwork(
            **self.model_params, device=self.device
        )
        self.q_value_target_network2 = CriticNetwork(
            **self.model_params, device=self.device
        )

        self.q_value_target_network1.load_state_dict(self.q_value_network1.state_dict())
        self.q_value_target_network1.eval()

        self.q_value_target_network2.load_state_dict(self.q_value_network2.state_dict())
        self.q_value_target_network2.eval()

        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()

    def train(self, params):
        self.memory = ReplayBuffer(buffer_limit=int(params["buffer_size"]))
        self.entropy_target = 0.98 * (-np.log(1 / params["num_actions"]))

        self.q_value1_opt = optim.Adam(
            self.q_value_network1.parameters(), lr=params["lr"]
        )
        self.q_value2_opt = optim.Adam(
            self.q_value_network2.parameters(), lr=params["lr"]
        )

        self.policy_opt = optim.Adam(self.policy_network.parameters(), lr=params["lr"])
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
                hard_update_target_network(
                    self.q_value_network1,
                    self.q_value_network2,
                    self.q_value_target_network1,
                    self.q_value_target_network2,
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
                    "originality_score": originality_score
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
        states, actions, rewards, next_states, done_masks, total_rewards = transitions

        states = states.long().to(self.device)
        actions = actions.clone().detach().to(dtype=torch.int64, device=self.device)
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
            # next_q = (next_q1 + next_q2) / 2

            next_v = (
                (next_probs * (next_q - self.alpha * next_log_probs))
                .sum(-1)
                .unsqueeze(-1)
            )

            target_q = rewards + params["gamma"] * (1 - done_masks) * next_v

        q1 = self.q_value_network1(states).gather(1, actions.unsqueeze(1))
        q2 = self.q_value_network2(states).gather(1, actions.unsqueeze(1))
        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)

        avg_q_loss = (q1_loss + q2_loss) * 0.5

        # Calculating the Policy target
        _, probs = self.policy_network(states)
        log_probs = torch.log(probs)
        with torch.no_grad():
            q1 = self.q_value_network1(states)
            q2 = self.q_value_network2(states)
            q = torch.min(q1, q2)
            # q = (q1 + q2) / 2

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
        alpha_loss = -(
            self.log_alpha * (log_probs.detach() + self.entropy_target)
        ).mean()

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        self.alpha = self.log_alpha.exp()

        # # Q values for all, id and ood actions
        # true_q_values_current = compute_true_q_values(
        #     actor=self.policy_network,
        #     states=states,
        #     actions=actions,
        #     reward_fn=self.test_fn,
        #     gamma=params["gamma"],
        #     max_sequence_length=params["seq_len"],
        # )

        # # Generates the Q values of the next_states + all possible actions
        # true_q_values_next_actions = compute_true_q_values_with_generated_actions(
        #     states=next_states,
        #     num_actions=params["num_actions"],
        #     actor=self.policy_network,
        #     reward_fn=self.test_fn,
        #     device=self.device,
        #     gamma=params["gamma"],
        #     max_sequence_length=params["seq_len"],
        # )

        # in_distribution_mask = self.memory.check_in_distribution_with_generated_actions(
        #     next_states, params["num_actions"], device=self.device
        # )

        # true_q_values_id_actions = true_q_values_next_actions[in_distribution_mask]
        # true_q_values_ood_actions = true_q_values_next_actions[~in_distribution_mask]
        # expected_q = (probs * (q + self.alpha * torch.log(probs))).sum(dim=1).mean()
        # expected_next_q = next_q + self.alpha * torch.log(next_probs)

        stats = {
            "alpha_loss": alpha_loss.item(),
            "critics_loss": avg_q_loss.item(),
            "actor_loss": policy_loss.item(),
            # "true_q_values_current": true_q_values_current.mean().item(),
            # "estimated_q_values_current": expected_q.item(),
            # "true_q_values_id_actions": true_q_values_id_actions.mean().item(),
            # "estimated_q_values_id_actions": expected_next_q[in_distribution_mask]
            # .mean()
            # .item(),
            # "true_q_values_ood_actions": true_q_values_ood_actions.mean().item(),
            # "estimated_q_values_ood_actions": expected_next_q[~in_distribution_mask]
            # .mean()
            # .item(),
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
            self.policy_network,
            batch_size=1,
            start_token=params["num_actions"],
            seq_len=params["seq_len"],
            device=self.device,
            greedy=True,
        )

        originality_score = self.memory.originality_score(state_batch)
        rewards_mean = torch.mean(self.test_fn(state_batch))

        return rewards_mean, originality_score
