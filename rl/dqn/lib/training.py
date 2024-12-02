import torch
import random
from torch import nn
import torch.nn.functional as F
from lib.constants import *
from torch.nn.utils.rnn import pad_sequence


def sample_sequences(q, epsilon, batch_size, start_token, seq_len, num_states, device):
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
            # Greedy action selection (exploitation)
            with torch.no_grad():
                q_values = q(state_batch)  # Compute Q-values
                actions = torch.argmax(q_values, dim=1).unsqueeze(1)

        # Append actions to the state batch
        state_batch = torch.cat((state_batch, actions), dim=1)

    # Return generated sequences, excluding the start token
    generated_sequence = state_batch[:, 1:]
    return generated_sequence


def train_step(q, q_target, memory, optimizer, config, device="cpu"):
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
    transitions = memory.sample_steps(config["batch_size"], config["fraction_best"])

    states, actions, rewards, next_states, done_masks, total_rewards = transitions

    discount_exponent = seq_len - states.shape[1]

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
    # torch.max(max_q_s_prime, total_rewards)

    # Compute target Q-values with no gradient calculation
    with torch.no_grad():
        max_q_s_prime = q_target(next_states).max(dim=1)[0]

        # target = rewards + config["gamma"] * torch.max(max_q_s_prime, total_rewards) * (1 - done_masks)
        target = rewards + config["gamma"] ** discount_exponent * total_rewards * (1 - done_masks)


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


def get_epsilon(episode_num, config, n_episodes):
    """
    Calculates the epsilon value for exploration based on the current episode number,
    considering the learning_starts parameter.
    """
    if episode_num < config["learning_starts"]:
        # Use the initial epsilon before training starts
        return config["exploration_initial_eps"]

    # Adjust the effective range of episodes after training starts
    adjusted_episode = episode_num - config["learning_starts"]
    adjusted_total_episodes = n_episodes - config["learning_starts"]

    # Calculate the fraction of progress in the adjusted range
    fraction = min(
        adjusted_episode / (config["exploration_fraction"] * adjusted_total_episodes),
        1.0,
    )

    # Calculate the difference between the final and initial epsilon
    eps_diff = config["exploration_final_eps"] - config["exploration_initial_eps"]

    # Return the epsilon for the current episode
    return config["exploration_initial_eps"] + fraction * eps_diff


def train_one_episode(
    q,
    q_target,
    memory,
    optimizer,
    config,
    episode,
    n_episodes,
    train_info,
    test_fn,
    test_fn_params,
    device,
):
    """Performs training for one episode, including sampling and gradient updates."""
    epsilon = get_epsilon(episode, config, n_episodes)

    state_batch = sample_sequences(
        q,
        epsilon=epsilon,
        batch_size=config["batch_size"],
        start_token=test_fn_params["num_states"],
        seq_len=test_fn_params["dim"],
        num_states=test_fn_params["num_states"],
        device=device,
    )
    total_rewards = test_fn(state_batch)
    memory.put(state_batch, total_rewards.to(device))

    if episode > config["learning_starts"]:
        for _ in range(config["replay_ratio"]):
            loss, td_error = train_step(q, q_target, memory, optimizer, config, device)
            train_info["losses"].append(loss)
            train_info["td_errors"].append(td_error)

        soft_update(q_target, q, config["tau"])

    return epsilon


def log_and_evaluate(
    q, memory, train_info, config, episode, test_fn, test_fn_params, fixed_state_batch, device
):
    """Logs and evaluates the model's performance at regular intervals."""
    max_score = memory.max_reward_score()
    # Sample sequences for evaluation
    state_batch = sample_sequences(
        q,
        epsilon=0,
        batch_size=config["batch_size"],
        start_token=test_fn_params["num_states"],
        seq_len=test_fn_params["dim"],
        num_states=test_fn_params["num_states"],
        device=device,
    )

    max_q_s = q(fixed_state_batch).max(dim=1)[0].mean().item()
    total_rewards = test_fn(state_batch)
    mean_batch_score = sum(total_rewards) / config["batch_size"]
    top_5_values, _ = torch.topk(total_rewards, 5)
    mean_top_5 = top_5_values.mean()

    # Store metrics
    train_info["episodes"].append(episode)
    train_info["max_scores"].append(max_score)
    train_info["mean_scores"].append(mean_batch_score)
    train_info["mean_top_fives"].append(mean_top_5)

    # Print metrics
    print(
        f"n_episode :{episode}, n_buffer : {memory.size()}, eps : {train_info['epsilons'][-1]:.1f}%"
    )
    print(
        f"mean score {mean_batch_score:.3f}, max score {max_score:.3f}, mean top 5 {mean_top_5:.3f}"
    )

    # Print mean loss and TD error if applicable
    if memory.size() > config["learning_starts"]:
        mean_loss = sum(train_info["losses"]) / len(train_info["losses"])
        mean_td_error = sum(train_info["td_errors"]) / len(train_info["td_errors"])
        print(
            f"Mean Loss: {mean_loss:.4f}, Mean TD Error: {mean_td_error:.4f}, Max State Q over the fixed batch: {max_q_s:.3f}"
        )
        train_info["mean_losses"].append(mean_loss)
        train_info["mean_td_errors"].append(mean_td_error)
        train_info["max_qs"].append(max_q_s)

    # Clear metrics for the next interval
    train_info["losses"] = []
    train_info["td_errors"] = []

    return mean_batch_score

def create_fixed_batch(config, test_fn_params, q, device):
    fixed_state_batch = sample_sequences(
        q=q,  # Use a random policy (or disable Q-function dependence)
        epsilon=0,  # Ensure full exploration if required
        batch_size=config["batch_size"],
        start_token=test_fn_params["num_states"],
        seq_len=test_fn_params["dim"],
        num_states=test_fn_params["num_states"],
        device=device
    )

    # Step 1: Generate random cut lengths
    cut_lengths = [random.randint(1, seq_len - 1) for _ in range(config["batch_size"])]

    # Step 2: Cut each sequence and store them in a list
    cut_sequences = [fixed_state_batch[i, :cut_lengths[i]] for i in range(config["batch_size"])]

    # Step 3: Pad sequences to the maximum cut length
    padded_sequences = torch.nn.utils.rnn.pad_sequence(cut_sequences, batch_first=True, padding_value=0)

    return padded_sequences

