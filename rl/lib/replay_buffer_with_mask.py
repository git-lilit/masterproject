import torch
import numpy as np
import collections
import torch.nn.functional as F

from lib.dataset_splits import get_saved_dataset, PercentileSplit


class ReplayBufferWithMask:
    def __init__(self, buffer_limit, ensemble_size, mask_prob):
        self.buffer = collections.deque(maxlen=buffer_limit)
        self.ensemble_size = ensemble_size
        self.mask_prob = mask_prob

    def put(self, sequences, rewards):
        sequences_list = sequences.tolist()
        rewards_list = rewards.tolist()

        for sequence, reward in zip(sequences_list, rewards_list):
            mask = torch.bernoulli(torch.full((self.ensemble_size,), self.mask_prob))
            self.buffer.append(
                (
                    torch.tensor(sequence, dtype=torch.float),
                    torch.tensor(reward, dtype=torch.float),
                    mask,
                )
            )

    def sample_episodes_priority(self, n, temperature=0.1):
        rewards = torch.tensor(
            [episode[1].item() for episode in self.buffer], dtype=torch.float32
        )

        if rewards.sum() > 0:
            probabilities = F.softmax(rewards / temperature, dim=0).numpy()
        else:
            probabilities = np.ones(len(self.buffer)) / len(self.buffer)

        sampled_indices = np.random.choice(len(self.buffer), size=n, p=probabilities)
        mini_batch = [self.buffer[idx] for idx in sampled_indices]
        final_sequence_list, total_reward_list, mask_list = zip(*mini_batch)

        return (
            torch.stack(final_sequence_list),
            torch.stack(total_reward_list),
            torch.stack(mask_list),
        )

    def sample_episodes_randomly(self, n):
        sampled_indices = np.random.choice(len(self.buffer), size=n, replace=True)
        mini_batch = [self.buffer[idx] for idx in sampled_indices]
        final_sequence_list, total_reward_list, mask_list = zip(*mini_batch)

        return (
            torch.stack(final_sequence_list),
            torch.stack(total_reward_list),
            torch.stack(mask_list),
        )

    def sample_steps(self, n, fraction_best):
        n_samples_random = int((1 - fraction_best) * n)
        # Sample random and best episodes
        if fraction_best != 1:
            final_sequences, total_rewards, masks = self.sample_episodes_randomly(
                n_samples_random
            )
        if fraction_best > 0 and fraction_best != 1:
            best_final_sequences, best_total_rewards, best_masks = (
                self.sample_episodes_priority(n - n_samples_random)
            )

            # Concatenate the random and best sequences and rewards
            final_sequences = torch.cat([final_sequences, best_final_sequences], dim=0)
            total_rewards = torch.cat(
                [total_rewards, best_total_rewards], dim=0
            ).squeeze(1)
            masks = torch.cat([masks, best_masks], dim=0)
        if fraction_best == 1:
            final_sequences, total_rewards, masks = self.sample_episodes_priority(n)

        seq_len = final_sequences.shape[1]

        # Generate a single cut point for the entire batch
        cut_point = torch.randint(1, seq_len, (1,)).item()

        # States and actions (cut at the same location for every sequence)
        states = final_sequences[:, :cut_point]
        actions = final_sequences[:, cut_point]

        # Next states (cut at the same location for every sequence)
        next_states = final_sequences[:, : cut_point + 1]

        # Convert the boolean comparison to tensor for rewards and done masks
        if cut_point == seq_len - 1:
            rewards = total_rewards
            done_masks = torch.ones((n,))
        else:
            rewards = torch.zeros((n,))
            done_masks = torch.zeros((n,))

        return states, actions, rewards, next_states, done_masks, total_rewards, masks

    def size(self):
        return len(self.buffer)

    def max_reward_score(self):
        if not self.buffer:
            return None

        rewards = torch.tensor([episode[1].item() for episode in self.buffer])
        max_reward = rewards.max().item()

        return max_reward

    def create_prefix_dict(self):
        """
        Creates a prefix dictionary for efficient prefix-based lookups.
        """
        self.prefix_dict = {}
        for sequence, reward, _ in self.buffer:
            sequence = sequence.int().tolist()
            for i in range(1, len(sequence) + 1):
                prefix = tuple(sequence[:i])
                if prefix not in self.prefix_dict:
                    self.prefix_dict[prefix] = True

    def check_in_distribution(self, query_sequences):
        """
        Checks if given sequences are in-distribution based on the prefix dictionary.

        Parameters:
        - query_sequences (torch.Tensor): A tensor of sequences to check.

        Returns:
        - torch.Tensor: A boolean tensor where True indicates in-distribution and False indicates out-of-distribution.
        """
        if not hasattr(self, "prefix_dict"):
            self.create_prefix_dict()

        in_distribution = []
        for sequence in query_sequences:
            sequence = sequence.int().tolist()
            prefix = tuple(sequence)  # Convert tensor to tuple
            in_distribution.append(
                self.prefix_dict.get(prefix, False)
            )  # Check if prefix exists

        return torch.tensor(in_distribution, dtype=torch.bool)

    def check_in_distribution_with_generated_actions(
        self, query_states, num_actions, device
    ):
        """
        Checks if given state-action pairs (resulting in next states) are in-distribution
        based on the prefix dictionary. The actions are generated internally based on num_actions.

        Parameters:
        - query_states (torch.Tensor): A tensor of states to check (shape: [num_states, state_dim]).
        - num_actions (int): The number of possible actions to generate.

        Returns:
        - torch.Tensor: A boolean tensor of shape [num_states, num_actions] where True indicates
        in-distribution and False indicates out-of-distribution for each state-action pair.
        """
        if not hasattr(self, "prefix_dict"):
            self.create_prefix_dict()

        num_states = query_states.size(0)

        # Create action tensor based on num_actions
        actions = torch.arange(num_actions, dtype=torch.int32)  # Shape: [num_actions]

        in_distribution = torch.zeros((num_states, num_actions), dtype=torch.bool)

        # Iterate over all actions
        for i, action in enumerate(actions):
            # Append the action to all states
            action_tensor = torch.full(
                (num_states, 1), action, dtype=torch.int32, device=device
            )  # Shape: [num_states, 1]
            next_states = torch.cat(
                [query_states.int(), action_tensor], dim=1
            )  # Shape: [num_states, state_dim + 1]

            # Check each resulting next state against the prefix dictionary
            for j, next_state in enumerate(next_states):
                prefix = tuple(next_state.tolist())  # Convert tensor to tuple
                in_distribution[j, i] = self.prefix_dict.get(prefix, False)

        return in_distribution
    
    def originality_score(self, query_sequences):
        """
        Computes the originality score for the final sequence in the batch.

        Parameters:
        - query_sequences (torch.Tensor): A tensor of sequences to check.

        Returns:
        - float: The originality score of the final sequence (higher means more original).
        """
        if not hasattr(self, "prefix_dict"):
            self.create_prefix_dict()

        batch_size = query_sequences.shape[0]
        batch_tuples = {tuple(seq.tolist()) for seq in query_sequences}
        number_of_seen = sum(seq in self.prefix_dict for seq in batch_tuples)

        return (batch_size - number_of_seen) / batch_size

