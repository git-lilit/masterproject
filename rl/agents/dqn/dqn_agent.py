import wandb
import torch
import collections
import torch.optim as optim
from lib.replay_buffer import ReplayBuffer
from lib.sequence_model import SequenceModel
from agents.dqn.utils import (
    get_epsilon,
    soft_update,
    compute_true_q_values,
)
import random
from torch.nn.utils import clip_grad_norm_
from torch import nn
from statistics import mean

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")


class DQNAgent:
    def __init__(self, model_params, test_fn, wandb_log, dataset, full_dataset, device):
        self.model_params = model_params
        self.test_fn = test_fn
        self.wandb_log = wandb_log
        self.device = device
        self.dataset = dataset
        self.full_dataset = full_dataset

        self.q = SequenceModel(**self.model_params, device=self.device)
        count_parameters(self.q)

        self.q_target = SequenceModel(**self.model_params, device=self.device)
        self.q_target.load_state_dict(self.q.state_dict())

    def train(self, params):
        self.memory = ReplayBuffer(buffer_limit=int(params["buffer_size"]))
        self.optimizer = optim.Adam(self.q.parameters(), lr=params["lr"])

        all_episode_stats = collections.defaultdict(list)
        all_interval_stats = collections.defaultdict(list)
        interval_length = params["print_interval"]

        if params["training_mode"] == "offline":
            X, y = self.dataset
            self.memory.put(X, y)

        for episode in range(params["n_episodes"]):
            epsilon = get_epsilon(episode, params)
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

            soft_update(self.q_target, self.q, params["tau"])

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
            estimated_q_next = self.q_target(next_states)
            max_q_s_prime = estimated_q_next.max(dim=1)[0]
            target = rewards + params["gamma"] * max_q_s_prime * (1 - done_masks)

            # MC
            # discount_exponent = params["seq_len"] - states.shape[1]
            # target = rewards + params["gamma"] ** discount_exponent * total_rewards * (1 - done_masks)

            # Max
            # target = rewards + params["gamma"] * torch.max(max_q_s_prime, total_rewards) * (1 - done_masks)

        td_error = torch.abs(target.unsqueeze(1) - q_s_a).mean().item()
        loss = nn.functional.mse_loss(q_s_a, target)

        # Conservative Q-Learning (CQL) Loss
        if params["loss_type"] == "cql":
            logsumexp_q = torch.logsumexp(q_out, dim=1)
            cql_loss = params["cql_alpha"] * (logsumexp_q - q_s_a).mean()

            loss = loss + cql_loss
            loss.backward()
            clip_grad_norm_(self.q.parameters(), 1.0)
            self.optimizer.step()
        else:
            # Perform backpropagation and optimizer step
            loss.backward()
            self.optimizer.step()

        # true_q_values_current = compute_true_q_values(
        #     states=states,
        #     actions=actions,
        #     gamma=params["gamma"],
        #     max_sequence_length=params["seq_len"],
        #     full_dataset=self.full_dataset,
        # )

        # in_distribution_mask = self.memory.check_in_distribution_with_generated_actions(
        #     next_states, params["num_actions"], device=self.device
        # )

        # true_q_values_next_actions = compute_true_q_values_with_generated_actions(
        #     full_dataset=self.full_dataset,
        #     states=next_states,
        #     num_actions=params["num_actions"],
        #     device=self.device,
        #     gamma=params["gamma"],
        #     max_sequence_length=params["seq_len"],
        # )

        # q_value_stats = {
        #     "true_q": true_q_values_current.mean(),
        #     "estimated_q": q_s_a.mean(),
        #     "true_q_id": true_q_values_next_actions[in_distribution_mask].mean(),
        #     "estimated_q_id": estimated_q_next[in_distribution_mask].mean(),
        #     "true_q_ood": true_q_values_next_actions[~in_distribution_mask].mean(),
        #     "estimated_q_ood": estimated_q_next[~in_distribution_mask].mean(),
        # }
        q_value_stats = {}

        return loss.item(), td_error, q_value_stats

    def log_stats(self, episode_stats, interval_stats):
        print(
            f"Episode N: {episode_stats["episode"]:} | Epsilon: {episode_stats['epsilon']:.3f} | "
            f"Episode loss: {episode_stats["loss"]:.3f} | Episode TD error: {episode_stats['td_error']:.3f}"
        )

        print(
            f"Buffer size: {episode_stats["buffer_size"]:} | Buffer Max Score: {episode_stats['max_score']:.3f}"
        )

        print(
            f"Generated Sequence reward mean: {interval_stats["seq_reward"]:.3f} | Originality Score: {interval_stats["originality_score"]:.3f} | "
            f"Mean TD error: {interval_stats["mean_td_error"]:.3f} | Mean Loss: {interval_stats["mean_loss"]:.3f}"
        )

        all_stats = {**episode_stats, **interval_stats}

        if self.wandb_log == True:
            wandb.log(all_stats)

    def evaluate(self, params):
        """Logs and evaluates the model's performance at regular intervals."""
        # Sample sequences for evaluation
        state_batch_max = self.sample_sequences(
            epsilon=0, params=params
        )

        max_rewards = torch.mean(self.test_fn(state_batch_max))
        originality_score_max = self.memory.originality_score(state_batch_max)

        return max_rewards, originality_score_max


    def sample_sequences(self, epsilon, params):
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
        batch_size = 1

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
