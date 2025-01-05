import torch
import random
import torch.nn.functional as F



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


def train_step_bc(model, memory, optimizer, params, device):
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
    transitions = memory.sample_steps(params["batch_size"], params["fraction_best"])
    states, actions, _, _, _, _ = transitions  # Ignore rewards, next_states, and masks for BC

    # Move data to the specified device
    states = states.long().to(device)
    actions = actions.to(device)

    # Ensure the model is on the correct device
    model = model.to(device)

    # Reset optimizer gradients
    optimizer.zero_grad()

    # Compute predictions for the current states
    logits = model(states)  # Shape: (batch_size, num_actions)

    # Compute loss using cross-entropy
    loss = F.cross_entropy(logits, actions.long())

    # Perform backpropagation and optimizer step
    loss.backward()
    optimizer.step()

    return loss.item()


def sample_sequences_bc(
    model, batch_size, start_token, seq_len, device
):
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
    # Move the model to the specified device
    model = model.to(device)

    # Initialize the state batch with the start token
    state_batch = torch.full(
        (batch_size, 1), start_token, dtype=torch.long, device=device
    )

    for _ in range(seq_len):
        # Predict the next action using the model (deterministic)
        with torch.no_grad():
            action_probs = model(state_batch)  # Model outputs the action probabilities
            actions = torch.argmax(action_probs, dim=1).unsqueeze(1)

        # Append predicted actions to the state batch
        state_batch = torch.cat((state_batch, actions), dim=1)

    # Return generated sequences, excluding the start token
    generated_sequence = state_batch[:, 1:]
    return generated_sequence