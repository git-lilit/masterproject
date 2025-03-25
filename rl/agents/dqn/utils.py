import torch
import random
import numpy as np
import torch.nn.functional as F


def soft_update(q_target, q, tau):
    for target_param, param in zip(q_target.parameters(), q.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


def soft_update_k(q_targets, qs, tau):
    for i in range(len(q_targets)):
        soft_update(q_targets[i], qs[i], tau)


def get_epsilon(episode, params):
    """
    Calculates the epsilon value for exploration based on the current episode number,
    considering the learning_starts parameter.
    """
    if episode < params["learning_starts"]:
        # Use the initial epsilon before training starts
        return params["exploration_initial_eps"]

    # Adjust the effective range of episodes after training starts
    adjusted_episode = episode - params["learning_starts"]
    adjusted_total_episodes = params["n_episodes"] - params["learning_starts"]

    # Calculate the fraction of progress in the adjusted range
    fraction = min(
        adjusted_episode / (params["exploration_fraction"] * adjusted_total_episodes),
        1.0,
    )

    # Calculate the difference between the final and initial epsilon
    eps_diff = params["exploration_final_eps"] - params["exploration_initial_eps"]

    # Return the epsilon for the current episode
    return params["exploration_initial_eps"] + fraction * eps_diff


def create_prefix_dict(full_dataset):
    """
    Creates a prefix dictionary for efficient prefix-based lookups.
    """
    prefix_dict = {}
    x, y = full_dataset

    for sequence, reward in zip(x, y):
        sequence = sequence.int().tolist()
        for i in range(1, len(sequence) + 1):
            prefix = tuple(sequence[:i])
            if prefix not in prefix_dict:
                prefix_dict[prefix] = reward
            else:
                prefix_dict[prefix] = max(reward, prefix_dict[prefix])


def compute_true_q_values(
    prefix_dict, states, gamma=1.0, max_sequence_length=8, num_actions=4
):
    """
    Computes Q-values for all possible actions for a batch of states using a prefix dictionary for efficient lookups.

    Parameters:
    - prefix_dict: A dictionary mapping state-action prefixes to rewards.
    - states: A batch of states (tensor of shape [B, S], where B is the batch size and S is the state length).
    - gamma: Discount factor for future rewards (default: 1.0).
    - max_sequence_length: The maximum length of sequences (default: 8).
    - num_actions: Number of possible actions (default: 4).

    Returns:
    - q_values: Q-values for all actions for the given batch of states (tensor of shape [B, A]).
    """
    batch_size = states.shape[0]

    # Expand states to shape [B, A, S] by repeating each state for all actions
    expanded_states = states.unsqueeze(1).repeat(1, num_actions, 1)  # [B, A, S]

    # Create an action tensor of shape [B, A, 1] with values from 0 to num_actions-1
    actions = torch.arange(num_actions, dtype=states.dtype, device=states.device)
    actions = actions.unsqueeze(0).expand(batch_size, -1).unsqueeze(-1)  # [B, A, 1]

    # Concatenate states with actions to form sequences of shape [B, A, S+1]
    sequences = torch.cat((expanded_states, actions), dim=-1).cpu().numpy()

    q_values = np.zeros((batch_size, num_actions))  # Placeholder for Q-values

    for i in range(batch_size):
        for j in range(num_actions):
            seq = tuple(sequences[i, j])  # Convert to tuple for dictionary lookup
            best_score = prefix_dict.get(
                seq, 0
            )  # Use prefix dictionary for fast lookup

            q_values[i, j] = best_score

    return torch.tensor(
        q_values, dtype=torch.float32, device=states.device
    )  # Shape [B, A]
