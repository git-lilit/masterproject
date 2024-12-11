import wandb
import torch
import collections
import pandas as pd
import torch.optim as optim
from lib.ReplayBuffer import ReplayBuffer
from lib.training_utils import (
    get_epsilon,
    sample_sequences,
    train_step,
    create_fixed_batch,
    soft_update
)
from lib.SequenceModel import SequenceModel


class DQNAgent:
    def __init__(self, model_params, test_fn, wandb_log, dataset, device):
        self.model_params = model_params
        self.test_fn = test_fn
        self.wandb_log = wandb_log
        self.device = device
        self.dataset = dataset

        self.q = SequenceModel(**self.model_params, device=self.device)
        self.q_target = SequenceModel(**self.model_params, device=self.device)
        self.q_target.load_state_dict(self.q.state_dict())

    def train(self, params):
        self.memory = ReplayBuffer(buffer_limit=int(params["buffer_size"]))
        self.optimizer = optim.Adam(self.q.parameters(), lr=params["lr"])

        all_episode_stats = collections.defaultdict(list)
        all_interval_stats = collections.defaultdict(list)
        interval_length = params["print_interval"]

        fixed_state_batch = create_fixed_batch(self.q, params, self.device)

        if params["training_mode"] == "offline":
            X, y = self.dataset
            self.memory.put(X, y)

        for episode in range(params["n_episodes"]):
            if params["training_mode"] == "online":
                self.interact_with_env(episode, params)

            epsilon, loss, td_error = self.train_one_episode(episode, params)

            current_episode_stats = {
                "episode": episode,
                "epsilon": epsilon,
                "loss": loss,
                "td_error": td_error,
                "buffer_size": self.memory.size(),
                "max_score": self.memory.max_reward_score(),
            }

            for key, value in current_episode_stats.items():
                all_episode_stats[key].append(value)

            soft_update(self.q_target, self.q, params["tau"])

            if episode % interval_length == 0:
                seq_reward, max_q = self.evaluate(params, fixed_state_batch)
                len(all_episode_stats["td_error"][-interval_length:])

                mean_td_error_interval = (
                    sum(all_episode_stats["td_error"][-interval_length:])
                    / interval_length
                )
                mean_loss_interval = (
                    sum(all_episode_stats["loss"][-interval_length:]) / interval_length
                )

                current_interval_stats = {
                    "seq_reward": seq_reward.item(),
                    "max_q": max_q,
                    "mean_td_error_interval": mean_td_error_interval,
                    "mean_loss_interval": mean_loss_interval,
                }

                for key, value in current_interval_stats.items():
                    all_interval_stats[key].append(value)

                self.log_stats(current_episode_stats, current_interval_stats)

        return all_episode_stats, all_interval_stats

    def interact_with_env(self, episode, params):
        epsilon = get_epsilon(episode, params)

        state_batch = sample_sequences(
            self.q,
            epsilon=epsilon,
            batch_size=params["batch_size"],
            start_token=params["num_states"],
            seq_len=params["seq_len"],
            num_states=params["num_states"],
            device=self.device,
            deterministic=params["deterministic"]
        )
        total_rewards = self.test_fn(state_batch)
        self.memory.put(state_batch, total_rewards.to(self.device))

    def train_one_episode(self, episode, params):
        """Performs training for one episode, including gradient updates."""
        epsilon = get_epsilon(episode, params)

        losses = []
        td_errors = []

        for _ in range(params["gradient_steps"]):
            loss, td_error = train_step(
                self.q, self.q_target, self.memory, self.optimizer, params, self.device
            )

            # Append values to lists
            losses.append(loss)
            td_errors.append(td_error)

        # Compute mean loss and TD error
        mean_loss = sum(losses) / len(losses)
        mean_td_error = sum(td_errors) / len(td_errors)

        return epsilon, mean_loss, mean_td_error

    def log_stats(self, episode_stats, interval_stats):
        print(
            f"Episode N: {episode_stats["episode"]:} | Epsilon: {episode_stats['epsilon']:.3f} | "
            f"Episode loss: {episode_stats["loss"]:.3f} | Episode TD error: {episode_stats['td_error']:.3f}"
        )

        print(
            f"Buffer size: {episode_stats["buffer_size"]:} | Buffer Max Score: {episode_stats['max_score']:.3f}"
        )

        print(
            f"Generated Sequence reward mean: {interval_stats["seq_reward"]:.3f} | Max Q: {interval_stats['max_q']:.3f} | "
            f"Mean TD error: {interval_stats["mean_td_error_interval"]:.3f} | Mean Loss: {interval_stats["mean_loss_interval"]:.3f}"
        )

        all_stats = {**episode_stats, **interval_stats}

        if self.wandb_log == True:
            wandb.log(all_stats)

    def evaluate(self, params, fixed_state_batch):
        """Logs and evaluates the model's performance at regular intervals."""
        # Sample sequences for evaluation
        batch_size = 1 if params["deterministic"] else params["batch_size"]
        state_batch = sample_sequences(
            self.q,
            epsilon=0,
            batch_size=batch_size,
            start_token=params["num_states"],
            seq_len=params["seq_len"],
            num_states=params["num_states"],
            device=self.device,
            deterministic=params["deterministic"]
        )

        max_q = self.q(fixed_state_batch).max(dim=1)[0].mean().item()
        rewards_mean = torch.mean(self.test_fn(state_batch))

        return rewards_mean, max_q
