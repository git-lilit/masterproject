# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from lib.SequenceModel import SequenceModel
from torch.distributions.categorical import Categorical


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
            logits, _ = actor(state_batch)  # Shape: (batch_size, num_actions)

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


def hard_update_target_network(q1, q2, q1_target, q2_target):
    q1_target.load_state_dict(q1.state_dict())
    q1_target.eval()
    q2_target.load_state_dict(q2.state_dict())
    q2_target.eval()


class PolicyNetwork(SequenceModel):
    def __init__(
        self, vocab_size, embedding_dim, num_heads, output_dim, seq_len, device="cpu"
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
