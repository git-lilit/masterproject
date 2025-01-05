import wandb
import torch
import collections
import pandas as pd
import torch.optim as optim
from lib.ReplayBuffer import ReplayBuffer
from lib.training_utils import (
    sample_sequences_bc,
)
from lib.SequenceModel import SequenceModel
import torch.nn.functional as F



class BCAgent:
    def __init__(self, model_params, test_fn, wandb_log, dataset, device):
        self.model_params = model_params
        self.test_fn = test_fn
        self.wandb_log = wandb_log
        self.device = device
        self.dataset = dataset

        self.model = SequenceModel(**self.model_params, device=self.device)

    def train(self, params):
        self.memory = ReplayBuffer(buffer_limit=int(params["buffer_size"]))
        self.optimizer = optim.Adam(self.model.parameters(), lr=params["lr"])

        all_episode_stats = collections.defaultdict(list)
        all_interval_stats = collections.defaultdict(list)
        interval_length = params["print_interval"]

        X, y = self.dataset
        self.memory.put(X, y)

        for episode in range(params["n_episodes"]):
            loss = self.train_one_episode(episode, params)

            current_episode_stats = {
                "episode": episode,
                "loss": loss,
                "buffer_size": self.memory.size(),
                "max_score": self.memory.max_reward_score(),
            }

            for key, value in current_episode_stats.items():
                all_episode_stats[key].append(value)

            if episode % interval_length == 0:
                seq_reward = self.evaluate(params)

                mean_loss_interval = (
                    sum(all_episode_stats["loss"][-interval_length:]) / interval_length
                )

                current_interval_stats = {
                    "seq_reward": seq_reward.item(),
                    "mean_loss_interval": mean_loss_interval,
                }

                for key, value in current_interval_stats.items():
                    all_interval_stats[key].append(value)

                self.log_stats(current_episode_stats, current_interval_stats)

        return all_episode_stats, all_interval_stats


    def train_one_episode(self, episode, params):
        """Performs training for one episode, including gradient updates."""
        losses = []

        for _ in range(params["gradient_steps"]):
            loss = self.train_step(
               self.model, self.memory, self.optimizer, params, self.device
            )

            # Append values to lists
            losses.append(loss)

        # Compute mean loss and TD error
        mean_loss = sum(losses) / len(losses)

        return mean_loss

    def log_stats(self, episode_stats, interval_stats):
        print(
            f"Episode N: {episode_stats["episode"]:} | "
            f"Episode loss: {episode_stats["loss"]:.3f} | "
        )

        print(
            f"Buffer size: {episode_stats["buffer_size"]:} | Buffer Max Score: {episode_stats['max_score']:.3f}"
        )

        print(
            f"Generated Sequence reward mean: {interval_stats["seq_reward"]:.3f} | "
            f"Mean Loss: {interval_stats["mean_loss_interval"]:.3f}"
        )

        all_stats = {**episode_stats, **interval_stats}

        if self.wandb_log == True:
            wandb.log(all_stats)

    def evaluate(self, params):
        """Logs and evaluates the model's performance at regular intervals."""
        # Sample sequences for evaluation
        state_batch = sample_sequences_bc(
            self.model,
            batch_size=1,
            start_token=params["num_actions"],
            seq_len=params["seq_len"],
            device=self.device,
        )

        rewards_mean = torch.mean(self.test_fn(state_batch))

        return rewards_mean
    
    def train_step(self, params):
        """
        Perform a single training step for Behavioral Cloning (BC).

        Args:
            model (nn.Module): The neural network to train.
            memory: The replay memory object containing (state, action) pairs.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            params (dict): Configuration dictionary with parameters like batch_size.
            device (str): Device to perform training on ('cuda' or 'cpu').

        Returns:
            float: The loss value for the training step.
        """
        # Sample transitions from memory (only state-action pairs are needed for BC)
        transitions = self.memory.sample_steps(params["batch_size"], params["fraction_best"])
        states, actions, _, _, _, _ = transitions  # Ignore rewards, next_states, and masks for BC

        # Move data to the specified device
        states = states.long().to(self.device)
        actions = actions.to(self.device)

        # Ensure the model is on the correct device
        model = model.to(self.device)

        # Reset optimizer gradients
        self.optimizer.zero_grad()

        # Compute predictions for the current states
        logits = model(states)  # Shape: (batch_size, num_actions)

        # Compute loss using cross-entropy
        loss = F.cross_entropy(logits, actions.long())

        # Perform backpropagation and optimizer step
        loss.backward()
        self.optimizer.step()

        return loss.item()
