# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from lib.sequence_model import SequenceModel


class PolicyNetwork(SequenceModel):
    def __init__(
        self, vocab_size, embedding_dim, num_heads, output_dim, seq_len, prior_scale=0, device="cpu"
    ):
        super(PolicyNetwork, self).__init__(
            vocab_size, embedding_dim, num_heads, output_dim, seq_len, device
        )

    def forward(self, x, seq_lens=None):
        logits = super(PolicyNetwork, self).forward(x, seq_lens)
        probs = F.softmax(logits, dim=-1)
        z = (probs == 0.0).float() * 1e-8  # Avoid log(0)
        probs = probs + z  # Out-of-place operation
        return logits, probs


class CriticNetwork(SequenceModel):
    def __init__(
        self, vocab_size, embedding_dim, num_heads, output_dim, seq_len, device="cpu"
    ):
        super(CriticNetwork, self).__init__(
            vocab_size, embedding_dim, num_heads, output_dim, seq_len, device
        )

    def forward(self, x, seq_lens=None):
        return super(CriticNetwork, self).forward(x, seq_lens)


def sample_sequences_sac(
    actor,
    batch_size,
    start_token,
    seq_len,
    device,
    greedy=False,
    temperature=0.25,
    top_k=None,
):
    """
    Samples sequences using the given actor model for SAC with enhanced diversity.

    Args:
        actor (nn.Module): The actor network model that outputs logits.
        batch_size (int): Number of sequences to sample.
        start_token (int): Starting token for the sequences.
        seq_len (int): Length of each sequence.
        device (str): Device to perform sampling on ('cuda' or 'cpu').
        greedy (bool): Whether to use greedy sampling. If False, uses multinomial sampling.
        temperature (float): Temperature for controlling randomness. Defaults to 1.0 (no scaling).
        top_k (int): Limits sampling to top-k tokens. Defaults to None (no restriction).

    Returns:
        torch.Tensor: Generated sequences of shape (batch_size, seq_len).
    """
    actor = actor.to(device)
    state_batch = torch.full(
        (batch_size, 1), start_token, dtype=torch.long, device=device
    )

    for _ in range(seq_len):
        with torch.no_grad():
            logits, probs = actor(state_batch)  # Shape: (batch_size, num_actions)

            # Apply temperature scaling
            scaled_logits = logits / temperature
            probs = F.softmax(scaled_logits, dim=-1)  # Shape: (batch_size, num_actions)

            # Apply top-k filtering if specified
            if top_k is not None:
                top_values, top_indices = torch.topk(probs, top_k, dim=-1)
                probs = torch.zeros_like(probs).scatter_(-1, top_indices, top_values)
                probs /= probs.sum(dim=-1, keepdim=True)  # Renormalize

            # Choose actions
            if greedy:
                actions = torch.argmax(probs, dim=-1, keepdim=True)  # Greedy sampling
            else:
                actions = torch.multinomial(probs, num_samples=1)  # Stochastic sampling

        state_batch = torch.cat((state_batch, actions), dim=1)

    return state_batch[:, 1:]


def hard_update_target_network(
    q_value_network1, q_value_network2, q_value_target_network1, q_value_target_network2
):
    """
    Perform a hard update for the target networks in the ensemble.

    Args:
        q_networks: List of Q-value networks (CriticNetwork).
        q_target_networks: List of target Q-value networks (CriticNetwork).
    """
    q_value_target_network1.load_state_dict(q_value_network1.state_dict())
    q_value_target_network1.eval()
    q_value_target_network2.load_state_dict(q_value_network2.state_dict())
    q_value_target_network2.eval()

def hard_update_target_network_ensemble(q_value_networks, q_value_target_networks):
    for q_net, q_target in zip(q_value_networks, q_value_target_networks):
            q_target.load_state_dict(q_net.state_dict())
            q_target.eval()

# def compute_true_q_values(
#     actor, states, actions, reward_fn, gamma=0.99, max_sequence_length=8
# ):
#     """
#     Computes Q-values for a batch of states and actions using a deterministic actor, assuming fixed-length states.

#     Parameters:
#     - actor: A function or model that takes a batch of states and outputs the next action deterministically.
#     - states: A batch of states (tensor of shape [B, S], where B is the batch size and S is the state length).
#     - actions: A batch of actions (tensor of shape [B]).
#     - reward_fn: A function that takes a batch of complete sequences and outputs the final rewards (tensor of shape [B]).
#     - gamma: Discount factor for future rewards (default: 0.99).
#     - max_sequence_length: The maximum length of sequences (default: 8).

#     Returns:
#     - q_values: Q-values for the given batch of states and actions (tensor of shape [B]).
#     """
#     # Initialize sequences by appending actions to states
#     actions = actions.unsqueeze(1)  # Shape [B, 1]
#     sequences = torch.cat((states, actions), dim=1)  # Shape [B, S+1]

#     # Follow the actor's deterministic policy to complete all sequences
#     while sequences.size(1) < max_sequence_length:
#         # Get next actions for all sequences
#         _, probs = actor(sequences)  # Get action probabilities
#         next_actions = torch.argmax(
#             probs, dim=-1, keepdim=True
#         )  # Deterministic next actions

#         # Append next actions to the sequences
#         sequences = torch.cat((sequences, next_actions), dim=1)

#     # Compute rewards for completed sequences
#     rewards = reward_fn(sequences)  # Shape [B]

#     # Compute the Q-values for the initial state-action pairs
#     q_values = rewards * (gamma ** (sequences.size(1) - states.size(1)))

#     return q_values


def compute_true_q_values_with_generated_actions(
    states,
    num_actions,
    q_function,
    reward_fn,
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
            true_q_values[:, i] = compute_true_q_values_no_actor(
                q_function,
                states,
                action_tensor.squeeze(),
                reward_fn,
                gamma,
                max_sequence_length,
            )

    return true_q_values


def compute_true_q_values_no_actor(
    q_function, states, actions, reward_fn, gamma=0.99, max_sequence_length=8
):
    """
    Computes Q-values for a batch of states and actions using a Q-function, assuming fixed-length states.

    Parameters:
    - q_function: A function or model that takes a batch of states and outputs Q-values for all possible actions.
    - states: A batch of states (tensor of shape [B, S], where B is the batch size and S is the state length).
    - actions: A batch of actions (tensor of shape [B]).
    - reward_fn: A function that takes a batch of complete sequences and outputs the final rewards (tensor of shape [B]).
    - gamma: Discount factor for future rewards (default: 0.99).
    - max_sequence_length: The maximum length of sequences (default: 8).

    Returns:
    - q_values: Q-values for the given batch of states and actions (tensor of shape [B]).
    """
    # Initialize sequences by appending actions to states
    actions = actions.unsqueeze(1)  # Shape [B, 1]
    sequences = torch.cat((states, actions), dim=1)  # Shape [B, S+1]

    # Follow the greedy policy using the Q-function to complete all sequences
    while sequences.size(1) < max_sequence_length:
        # Get Q-values for all possible actions
        q_values = q_function(sequences)  # Shape [B, num_actions]
        next_actions = torch.argmax(q_values, dim=-1, keepdim=True)  # Greedy action selection

        # Append next actions to the sequences
        sequences = torch.cat((sequences, next_actions), dim=1)

    # Compute rewards for completed sequences
    rewards = reward_fn(sequences)  # Shape [B]

    # Compute the Q-values for the initial state-action pairs
    q_values = rewards * (gamma ** (sequences.size(1) - states.size(1)))

    return q_values
