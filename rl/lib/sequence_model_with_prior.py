import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.sequence_model import SequenceModel

class SequenceModelWithPrior(nn.Module):
    def __init__(
        self, vocab_size, embedding_dim, num_heads, output_dim, seq_len, prior_scale=10.0, device="cpu"
    ):
        super(SequenceModelWithPrior, self).__init__()
        self.device = device
        self.prior_scale = prior_scale
        
        # Main model
        self.model = SequenceModel(vocab_size, embedding_dim, num_heads, output_dim, seq_len, device)
        
        # Prior model (frozen)
        self.prior = SequenceModel(vocab_size, embedding_dim, num_heads, output_dim, seq_len, device)
        for param in self.prior.parameters():
            param.requires_grad = False  # Freeze prior network

        self.to(device)

    def forward(self, x, seq_lens=None):
        raw_output = self.model(x, seq_lens)
        with torch.no_grad():
            prior_output = self.prior(x, seq_lens)
        return raw_output + self.prior_scale * prior_output