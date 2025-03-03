import wandb
import torch
import collections
import torch.optim as optim
from lib.replay_buffer import ReplayBuffer
from lib.sequence_model import SequenceModel
from agents.dqn.utils import (
    get_epsilon,
    soft_update_k,
    compute_true_q_values,
    compute_true_q_values_with_generated_actions,
)
import random
from torch import nn
from statistics import mean


class DQNAgentEnsemble:
    def __init__(
        self,
        model_params,
        test_fn,
        wandb_log,
        dataset,
        full_dataset,
        n_networks,
        beta,
        integration_type,
        device,
    ):
        self.model_params = model_params
        self.test_fn = test_fn
        self.wandb_log = wandb_log
        self.device = device
        self.dataset = dataset
        self.full_dataset = full_dataset
        self.k = n_networks
        self.beta = beta
        self.integration_type = integration_type

        self.qs = [
            SequenceModel(**self.model_params, device=self.device)
            for i in range(self.k)
        ]
        self.q_targets = [
            SequenceModel(**self.model_params, device=self.device)
            for i in range(self.k)
        ]

        for i in range(self.k):
            q = self.qs[i]
            q_target = self.q_targets[i]
            q_target.load_state_dict(q.state_dict())

    def train(self, params):
        self.memory = ReplayBuffer(buffer_limit=int(params["buffer_size"]))
        self.optimizers = [
            optim.Adam(self.qs[i].parameters(), lr=params["lr"]) for i in range(self.k)
        ]

        all_episode_stats = collections.defaultdict(list)
        all_interval_stats = collections.defaultdict(list)
        interval_length = params["print_interval"]

        # fixed_state_batch = self.create_fixed_batch(params)

        if params["training_mode"] == "offline":
            X, y = self.dataset
            self.memory.put(X, y)

        for episode in range(params["n_episodes"]):
            epsilon = get_epsilon(episode, params)
            if self.integration_type == "rem":
                loss, td_error, q_stats = self.train_step_rem(params)
            else:
                loss, td_error, q_stats = self.train_step(params)

            step_stats = {
                "episode": episode,
                "epsilon": epsilon,
                "loss": loss,
                "td_error": td_error,
                "buffer_size": self.memory.size(),
                "max_score": self.memory.max_reward_score(),
            }

            step_stats.update(q_stats)

            for key, value in step_stats.items():
                all_episode_stats[key].append(value)

            soft_update_k(self.q_targets, self.qs, params["tau"])

            if episode % interval_length == 0:
                seq_reward, max_q = self.evaluate(params, None)

                keys = list(step_stats.keys())

                mean_losses = {
                    key: sum(all_episode_stats[key][-interval_length:])
                    / interval_length
                    for key in keys
                }

                current_interval_stats = {
                    "seq_reward": seq_reward.item(),
                }

                current_interval_stats.update(
                    {f"mean_{key}": mean_losses[key] for key in keys}
                )

                for key, value in current_interval_stats.items():
                    all_interval_stats[key].append(value)

                self.log_stats(step_stats, current_interval_stats)

        return all_episode_stats, all_interval_stats

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

        losses = []
        td_errors = []

        for i in range(self.k):
            q = self.qs[i]
            q_target = self.q_targets[i]
            optimizer = self.optimizers[i]

            # Ensure models are on the correct device
            q = q.to(device)
            q_target = q_target.to(device)

            # Reset optimizer gradients
            optimizer.zero_grad()

            # Compute Q-values for the current states and actions
            q_out = q(states)
            q_s_a = torch.gather(
                q_out, dim=1, index=actions.long().reshape(-1, 1)
            ).squeeze(1)

            # Compute independent target Q-values with no gradient calculation
            with torch.no_grad():
                max_q_s_prime = q_target(next_states).max(dim=1)[0]
                target = rewards + params["gamma"] * max_q_s_prime * (1 - done_masks)

            # Compute loss and TD error
            loss = nn.functional.mse_loss(q_s_a, target)
            td_error = torch.abs(target.unsqueeze(1) - q_s_a).mean()

            # Perform backpropagation and optimizer step
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            td_errors.append(td_error.item())

        return mean(losses), mean(td_errors), {}

    def train_step_rem(self, params):
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

        alphas = self.sample_alphas(self.k, self.device)

        # Compute convex combination of Q-values for the current state-action pair
        q_values = torch.stack(
            [q(states) for q in self.qs]
        )  # Shape: (K, batch_size, action_dim)
        q_s_a = torch.gather(
            q_values,
            dim=2,
            index=actions.long().reshape(1, -1, 1).expand(self.k, -1, 1),
        ).squeeze(2)
        q_alpha_s_a = torch.sum(
            alphas.view(-1, 1) * q_s_a, dim=0
        )  # Shape: (batch_size,)

        with torch.no_grad():
            # Compute convex combination of target Q-values
            q_targets_values = torch.stack(
                [q_target(next_states) for q_target in self.q_targets]
            )  # (K, batch, action_dim)
            max_q_targets = q_targets_values.max(dim=2)[
                0
            ]  # max over actions, shape: (K, batch)
            q_alpha_s_prime = torch.sum(
                alphas.view(-1, 1) * max_q_targets, dim=0
            )  # (batch,)

            # Compute target
            target = rewards + params["gamma"] * q_alpha_s_prime * (1 - done_masks)

        # Compute loss
        loss = nn.functional.mse_loss(q_alpha_s_a, target)
        td_error = torch.abs(target - q_alpha_s_a).mean()

        # Perform backpropagation and optimization step
        for i in range(self.k):
            optimizer = self.optimizers[i]
            optimizer.zero_grad()

        loss.backward()

        for i in range(self.k):
            self.optimizers[i].step()

        return loss.item(), td_error.item(), {}

    def log_stats(self, episode_stats, interval_stats):
        print(
            f"Episode N: {episode_stats["episode"]:} | Epsilon: {episode_stats['epsilon']:.3f} | "
            f"Episode loss: {episode_stats["loss"]:.3f} | Episode TD error: {episode_stats['td_error']:.3f}"
        )

        print(
            f"Buffer size: {episode_stats["buffer_size"]:} | Buffer Max Score: {episode_stats['max_score']:.3f}"
        )

        print(
            f"Generated Sequence reward mean: {interval_stats["seq_reward"]:.3f}| "
            f"Mean TD error: {interval_stats["mean_td_error"]:.3f} | Mean Loss: {interval_stats["mean_loss"]:.3f}"
        )

        all_stats = {**episode_stats, **interval_stats}

        if self.wandb_log == True:
            wandb.log(all_stats)

    def evaluate(self, params, fixed_state_batch):
        """Logs and evaluates the model's performance at regular intervals."""
        # Sample sequences for evaluation
        state_batch = self.sample_sequences(epsilon=0, batch_size=1, params=params)

        rewards_mean = torch.mean(self.test_fn(state_batch))

        return rewards_mean, 0

    def sample_alphas(self, K, device="cpu"):
        """
        Samples a random point alpha in the K-simplex, i.e., alpha >= 0 and sum(alpha)=1.

        Steps:
        1. Draw K values i.i.d. from Uniform(0,1).
        2. Normalize by dividing by the sum.

        Returns:
        alpha: (K,) tensor with non-negative entries summing to 1.
        """
        alpha_unnormalized = torch.rand(K, device=device)  # [K] from Uniform(0,1)
        alpha = alpha_unnormalized / alpha_unnormalized.sum()  # normalize
        return alpha

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
                    q_values_stack = torch.stack([q(state_batch) for q in self.qs])

                    match self.integration_type:
                        case "min":
                            final_q_estimate, _ = torch.min(q_values_stack, dim=0)
                        case "var":
                            mean_qs = torch.mean(q_values_stack, dim=0)
                            std_qs = torch.std(q_values_stack, dim=0)
                            final_q_estimate = mean_qs - self.beta * std_qs
                        case "rem":
                            final_q_estimate = q_values_stack.mean(dim=0)
                        case _:
                            raise ValueError(
                                "Integration type should be one of min, var, rem"
                            )

                    actions = torch.argmax(final_q_estimate, dim=1).unsqueeze(1)

            # Append actions to the state batch
            state_batch = torch.cat((state_batch, actions), dim=1)

        # Return generated sequences, excluding the start token
        generated_sequence = state_batch[:, 1:]
        return generated_sequence

    def create_fixed_batch(self, params):
        fixed_state_batch = self.sample_sequences(
            epsilon=0,  # Ensure full exploration if required
            batch_size=params["batch_size"],
            params=params,
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
