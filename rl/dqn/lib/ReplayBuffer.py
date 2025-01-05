import torch
import numpy as np
import collections
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


class ReplayBuffer:
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, sequences, rewards):
        sequences_list = sequences.tolist()
        rewards_list = rewards.tolist()

        for sequence, reward in zip(sequences_list, rewards_list):
            self.buffer.append(
                (
                    torch.tensor(sequence, dtype=torch.float),
                    torch.tensor(reward, dtype=torch.float),
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
        final_sequence_lst, total_reward_lst = zip(*mini_batch)

        return (
            torch.stack(final_sequence_lst),
            torch.stack(total_reward_lst),
        )

    def sample_episodes_randomly(self, n):
        sampled_indices = np.random.choice(len(self.buffer), size=n, replace=True)
        mini_batch = [self.buffer[idx] for idx in sampled_indices]
        final_sequence_lst, total_reward_lst = zip(*mini_batch)

        return (
            torch.stack(final_sequence_lst),
            torch.stack(total_reward_lst),
        )

    def sample_steps(self, n, fraction_best):
        n_samples_random = int((1 - fraction_best) * n)
        # Sample random and best episodes
        final_sequences, total_rewards = self.sample_episodes_randomly(
            n_samples_random
        )
        if fraction_best > 0:
            best_final_sequences, best_total_rewards = self.sample_episodes_priority(
                n - n_samples_random
            )

            # Concatenate the random and best sequences and rewards
            final_sequences = torch.cat([final_sequences, best_final_sequences], dim=0)
            total_rewards = torch.cat([total_rewards, best_total_rewards], dim=0).squeeze(1)

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

        return states, actions, rewards, next_states, done_masks, total_rewards

    def size(self):
        return len(self.buffer)

    def max_reward_score(self):
        if not self.buffer:
            return None

        rewards = torch.tensor([episode[1].item() for episode in self.buffer])
        max_reward = rewards.max().item()

        return max_reward
