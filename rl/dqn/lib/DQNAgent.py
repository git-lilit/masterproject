import wandb
import torch
import collections
import torch.optim as optim
from lib.ReplayBuffer import ReplayBuffer
from lib.training_utils import get_epsilon, soft_update
from lib.SequenceModel import SequenceModel
import random
from torch import nn


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

        fixed_state_batch = self.create_fixed_batch(params)

        if params["training_mode"] == "offline":
            X, y = self.dataset
            self.memory.put(X, y)

        for episode in range(params["n_episodes"]):
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


    def train_one_episode(self, episode, params):
        """Performs training for one episode, including gradient updates."""
        epsilon = get_epsilon(episode, params)

        losses = []
        td_errors = []

        for _ in range(params["gradient_steps"]):
            loss, td_error = self.train_step(params)

            # Append values to lists
            losses.append(loss)
            td_errors.append(td_error)

        # Compute mean loss and TD error
        mean_loss = sum(losses) / len(losses)
        mean_td_error = sum(td_errors) / len(td_errors)

        return epsilon, mean_loss, mean_td_error


    def train_step(self, params):
        """
        Perform a single training step for a DQN model on a GPU.
        """

        # Sample transitions from memory
        device = self.device
        transitions = self.memory.sample_steps(
            params["batch_size"], params["fraction_best"]
        )
        states, actions, rewards, next_states, done_masks, total_rewards = transitions

        # Move all data to the specified device
        states = states.long().to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_states = next_states.long().to(device)
        done_masks = done_masks.to(device)
        total_rewards = total_rewards.to(device)

        # Ensure models are on the correct device
        self.q = self.q.to(device)
        self.q_target = self.q_target.to(device)

        # Reset optimizer gradients
        self.optimizer.zero_grad()

        # Compute Q-values for the current states and actions
        q_out = self.q(states)
        q_s_a = torch.gather(q_out, dim=1, index=actions.long().reshape(-1, 1)).squeeze(
            1
        )

        # Compute target Q-values with no gradient calculation
        with torch.no_grad():
            max_q_s_prime = self.q_target(next_states).max(dim=1)[0]
            target = rewards + params["gamma"] * max_q_s_prime * (1 - done_masks)

            # MC
            # discount_exponent = params["seq_len"] - states.shape[1]
            # target = rewards + params["gamma"] ** discount_exponent * total_rewards * (1 - done_masks)

            # Max
            # target = rewards + params["gamma"] * torch.max(max_q_s_prime, total_rewards) * (1 - done_masks)

        # Compute loss and TD error
        loss = nn.functional.mse_loss(q_s_a, target)
        td_error = torch.abs(target.unsqueeze(1) - q_s_a).mean().item()

        if params["loss_type"] == "cql":
            logsumexp_q = torch.logsumexp(q_out, dim=1)
            cql_loss = params["cql_alpha"] * (logsumexp_q.mean() - q_s_a.mean())
            loss += cql_loss

        # Perform backpropagation and optimizer step
        loss.backward()
        self.optimizer.step()

        return loss.item(), td_error


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
        state_batch = self.sample_sequences(epsilon=0, batch_size=1, params=params)

        max_q = self.q(fixed_state_batch).max(dim=1)[0].mean().item()
        rewards_mean = torch.mean(self.test_fn(state_batch))

        return rewards_mean, max_q


    def sample_sequences(self, epsilon, batch_size, params):
        """
        Samples sequences using the given Q-network.

        Args:
            q (nn.Module): The Q-network model.
            epsilon (float): Epsilon value for epsilon-greedy policy.
            batch_size (int): Number of sequences to sample.
            start_token (int): Starting token for the sequences.
            seq_len (int): Length of each sequence.
            num_states (int): Number of possible states (vocabulary size).
            device (str): Device to perform sampling on ('cuda' or 'cpu').

        Returns:
            torch.Tensor: Generated sequences of shape (batch_size, seq_len).
        """
        # Move Q-network to the specified device
        self.q = self.q.to(self.device)
        start_token = params["num_actions"]
        seq_len = params["seq_len"]
        num_states = params["num_actions"]

        # Initialize the state batch with the start token
        state_batch = torch.full(
            (batch_size, 1), start_token, dtype=torch.long, device=self.device
        )

        for _ in range(seq_len):
            if random.random() < epsilon:
                # Random action sampling (exploration)
                actions = torch.randint(
                    0, num_states, (batch_size, 1), dtype=torch.long, device=self.device
                )
            else:
                # Action selection based on deterministic approach
                with torch.no_grad():
                    q_values = self.q(state_batch)  # Compute Q-values
                    actions = torch.argmax(q_values, dim=1).unsqueeze(1)

            # Append actions to the state batch
            state_batch = torch.cat((state_batch, actions), dim=1)

        # Return generated sequences, excluding the start token
        generated_sequence = state_batch[:, 1:]
        return generated_sequence


    def create_fixed_batch(self, params):
        fixed_state_batch = self.sample_sequences(
            epsilon=0,  # Ensure full exploration if required
            batch_size=params["batch_size"],
            params=params
        )

        # Step 1: Generate random cut lengths
        cut_lengths = [
            random.randint(1, params["seq_len"] - 1)
            for _ in range(params["batch_size"])
        ]

        # Step 2: Cut each sequence and store them in a list
        cut_sequences = [
            fixed_state_batch[i, : cut_lengths[i]] for i in range(params["batch_size"])
        ]

        # Step 3: Pad sequences to the maximum cut length
        padded_sequences = torch.nn.utils.rnn.pad_sequence(
            cut_sequences, batch_first=True, padding_value=0
        )

        return padded_sequences


    def interact_with_env(self, episode, params):
        epsilon = get_epsilon(episode, params)

        state_batch = self.sample_sequences(
            epsilon=epsilon, batch_size=params["batch_size"], params=params
        )
        total_rewards = self.test_fn(state_batch)
        self.memory.put(state_batch, total_rewards.to(self.device))