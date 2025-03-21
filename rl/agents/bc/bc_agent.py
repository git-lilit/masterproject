import wandb
import torch
import collections
import torch.optim as optim
import torch.nn.functional as F

from lib.replay_buffer import ReplayBuffer
from lib.sequence_model import SequenceModel


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
            loss = self.train_step(params)

            current_episode_stats = {
                "episode": episode,
                "loss": loss,
                "buffer_size": self.memory.size(),
                "max_score": self.memory.max_reward_score(),
            }

            for key, value in current_episode_stats.items():
                all_episode_stats[key].append(value)

            if episode % interval_length == 0:
                seq_reward, originality_score = self.evaluate(params)

                mean_loss_interval = (
                    sum(all_episode_stats["loss"][-interval_length:]) / interval_length
                )

                current_interval_stats = {
                    "seq_reward": seq_reward.item(),
                    "originality_score": originality_score,
                    "mean_loss_interval": mean_loss_interval,
                }

                for key, value in current_interval_stats.items():
                    all_interval_stats[key].append(value)

                self.log_stats(current_episode_stats, current_interval_stats)

        return all_episode_stats, all_interval_stats

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
        state_batch = self.sample_sequences(batch_size=1, params=params)

        max_rewards = torch.mean(self.test_fn(state_batch))
        originality_score_max = self.memory.originality_score(state_batch)

        return max_rewards, originality_score_max

    def train_step(self, params):
        """
        Perform a single training step for Behavioral Cloning (BC).

        Args:
            params (dict): Configuration dictionary with parameters like batch_size.

        Returns:
            float: The loss value for the training step.
        """
        transitions = self.memory.sample_steps(
            params["batch_size"], params["fraction_best"]
        )
        states, actions, _, _, _, _ = (
            transitions
        )

        states = states.long().to(self.device)
        actions = actions.to(self.device)

        model = self.model.to(self.device)

        self.optimizer.zero_grad()
        logits = model(states)
        loss = F.cross_entropy(logits, actions.long())

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def sample_sequences(self, batch_size, params):
        """
        Samples sequences using the given Behavioral Cloning model.

        Args:
            model (nn.Module): The behavioral cloning model (typically a neural network).
            batch_size (int): Number of sequences to sample.
            start_token (int): Starting token for the sequences.
            seq_len (int): Length of each sequence.
            device (str): Device to perform sampling on ('cuda' or 'cpu').

        Returns:
            torch.Tensor: Generated sequences of shape (batch_size, seq_len).
        """
        start_token = params["num_actions"]
        seq_len = params["seq_len"]
        model = self.model.to(self.device)

        state_batch = torch.full(
            (batch_size, 1), start_token, dtype=torch.long, device=self.device
        )

        for _ in range(seq_len):
            with torch.no_grad():
                action_probs = model(
                    state_batch
                )
                actions = torch.argmax(action_probs, dim=1).unsqueeze(1)

            state_batch = torch.cat((state_batch, actions), dim=1)

        generated_sequence = state_batch[:, 1:]
        return generated_sequence
