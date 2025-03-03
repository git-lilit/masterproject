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


def compute_true_q_values(
    full_dataset, states, actions, gamma=0.99, max_sequence_length=8
):
    """
    Computes Q-values for a batch of states and actions by finding the best matching sequence from a dataset.

    Parameters:
    - full_dataset: A tuple (X, Y) where X is a numpy array of sequences and Y contains corresponding scores.
    - states: A batch of states (tensor of shape [B, S], where B is the batch size and S is the state length).
    - actions: A batch of actions (tensor of shape [B]).
    - gamma: Discount factor for future rewards (default: 0.99).
    - max_sequence_length: The maximum length of sequences (default: 8).

    Returns:
    - q_values: Q-values for the given batch of states and actions (tensor of shape [B]).
    """
    # Unpack dataset
    X, Y = full_dataset

    # Initialize sequences by appending actions to states
    actions = actions.unsqueeze(1)  # Shape [B, 1]
    sequences = (
        torch.cat((states, actions), dim=1).cpu().numpy()
    )  # Convert to numpy for efficient search

    best_scores = []

    for seq in sequences:
        # Find the best matching sequence from X
        mask = np.all(X[:, : len(seq)] == seq, axis=1)
        if np.any(mask):
            best_score = np.max(Y[mask])
        else:
            best_score = 0  # Default if no match is found

        best_scores.append(best_score)

    rewards = torch.tensor(best_scores, dtype=torch.float32)  # Convert to tensor

    # Compute the Q-values for the initial state-action pairs
    q_values = rewards * (gamma ** (max_sequence_length - states.size(1)))

    return q_values


def compute_true_q_values_with_generated_actions(
    states,
    num_actions,
    full_dataset,
    device,
    gamma=0.99,
    max_sequence_length=8,
):
    """
    Computes the true Q-values for given state-action pairs (resulting in next states)
    generated from a batch of states and actions. The actions are generated internally
    based on num_actions.

    Parameters:
    - query_states (torch.Tensor): A tensor of states to check (shape: [num_states, state_dim]).
    - num_actions (int): The number of possible actions to generate.
    - actor: A function or model that takes a batch of states and outputs the next action deterministically.
    - reward_fn: A function that takes a batch of complete sequences and outputs the final rewards (tensor of shape [B]).
    - device: The device (CPU or GPU) to use for computation.
    - gamma: Discount factor for future rewards (default: 0.99).
    - max_sequence_length: The maximum length of sequences (default: 8).

    Returns:
    - true_q_values: A tensor of true Q-values with shape [num_states, num_actions].
    """
    num_states = states.size(0)

    # Create action tensor based on num_actions
    actions = torch.arange(
        num_actions, dtype=torch.int32, device=device
    )  # Shape: [num_actions]

    # Initialize result tensor for true Q-values
    true_q_values = torch.zeros(
        (num_states, num_actions), dtype=torch.float32, device=device
    )

    # Compute true Q-values for all state-action pairs in batches
    for i, action in enumerate(actions):
        # Append the action to all states
        action_tensor = torch.full(
            (num_states, 1), action, dtype=torch.int32, device=device
        )  # Shape: [num_states, 1]

        # Compute true Q-values for these state-action pairs using actor and reward function
        if states.shape[1] != max_sequence_length:
            true_q_values[:, i] = compute_true_q_values(
                full_dataset,
                states,
                action_tensor.squeeze(),
                gamma,
                max_sequence_length,
            )

    return true_q_values
