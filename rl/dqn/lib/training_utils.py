import torch
import random
from torch import nn
import torch.nn.functional as F


def sample_sequences(
    q, epsilon, batch_size, start_token, seq_len, num_states, device, deterministic=True
):
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
    q = q.to(device)

    # Initialize the state batch with the start token
    state_batch = torch.full(
        (batch_size, 1), start_token, dtype=torch.long, device=device
    )

    for _ in range(seq_len):
        if random.random() < epsilon:
            # Random action sampling (exploration)
            actions = torch.randint(
                0, num_states, (batch_size, 1), dtype=torch.long, device=device
            )
        else:
            # Action selection based on deterministic or softmax approach
            with torch.no_grad():
                q_values = q(state_batch)  # Compute Q-values

                if deterministic:
                    # Greedy action selection (exploitation)
                    actions = torch.argmax(q_values, dim=1).unsqueeze(1)
                else:
                    # Softmax sampling (exploration with probabilities)
                    probabilities = F.softmax(
                        q_values, dim=1
                    )  # Apply softmax to Q-values
                    actions = torch.multinomial(
                        probabilities, 1
                    )  # Sample based on probabilities

        # Append actions to the state batch
        state_batch = torch.cat((state_batch, actions), dim=1)

    # Return generated sequences, excluding the start token
    generated_sequence = state_batch[:, 1:]
    return generated_sequence


def train_step(q, q_target, memory, optimizer, params, device):
    """
    Perform a single training step for a DQN model on a GPU.

    Args:
        q (nn.Module): The main Q-network.
        q_target (nn.Module): The target Q-network.
        memory: The replay memory object.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        config (dict): Configuration dictionary with parameters like batch_size, gamma, etc.
        device (str): Device to perform training on ('cuda' or 'cpu').

    Returns:
        tuple: The loss value and TD error for the training step.
    """
    # Sample transitions from memory
    transitions = memory.sample_steps(params["batch_size"], params["fraction_best"])
    states, actions, rewards, next_states, done_masks, total_rewards = transitions

    # Move all data to the specified device
    states = states.long().to(device)
    actions = actions.to(device)
    rewards = rewards.to(device)
    next_states = next_states.long().to(device)
    done_masks = done_masks.to(device)
    total_rewards = total_rewards.to(device)

    # Ensure models are on the correct device
    q = q.to(device)
    q_target = q_target.to(device)

    # Reset optimizer gradients
    optimizer.zero_grad()

    # Compute Q-values for the current states and actions
    q_out = q(states)
    q_s_a = torch.gather(q_out, dim=1, index=actions.long().reshape(-1, 1)).squeeze(1)

    # Compute target Q-values with no gradient calculation
    with torch.no_grad():
        max_q_s_prime = q_target(next_states).max(dim=1)[0]
        target = rewards + params["gamma"] * max_q_s_prime * (1 - done_masks)

        # # MC
        # discount_exponent = params["seq_len"] - states.shape[1]
        # target = rewards + params["gamma"] ** discount_exponent * total_rewards * (1 - done_masks)

        # Max
        # target = rewards + params["gamma"] * torch.max(max_q_s_prime, total_rewards) * (1 - done_masks)

    # Compute loss and TD error
    loss = nn.functional.mse_loss(q_s_a, target)
    td_error = torch.abs(target.unsqueeze(1) - q_s_a).mean().item()

    # Perform backpropagation and optimizer step
    loss.backward()
    optimizer.step()

    return loss.item(), td_error


def soft_update(q_target, q, tau):
    for target_param, param in zip(q_target.parameters(), q.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


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


def create_fixed_batch(q, params, device):
    fixed_state_batch = sample_sequences(
        q=q,  # Use a random policy (or disable Q-function dependence)
        epsilon=0,  # Ensure full exploration if required
        batch_size=params["batch_size"],
        start_token=params["num_states"],
        seq_len=params["seq_len"],
        num_states=params["num_states"],
        device=device,
    )

    # Step 1: Generate random cut lengths
    cut_lengths = [
        random.randint(1, params["seq_len"] - 1) for _ in range(params["batch_size"])
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
