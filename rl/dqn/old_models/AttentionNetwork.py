import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionNetwork(nn.Module):
    def __init__(self, seq_length=128, vocab_size=20, embedding_dim=32, num_heads=4, device='cpu'):
        super(AttentionNetwork, self).__init__()
        
        # Hyperparameters
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.latent_dim = embedding_dim * num_heads  # 128 for 4 heads
        self.device = device
        self.padding_token = self.vocab_size

        # Pre-processing
        self.token_embedding = nn.Embedding(vocab_size + 1, embedding_dim)
        self.positional_embedding = nn.Parameter(torch.randn(seq_length, embedding_dim))
        self.repeat_heads = nn.Linear(embedding_dim, self.latent_dim)

        # Attention Block
        self.layer_norm_1 = nn.LayerNorm(self.latent_dim)
        self.attention = nn.MultiheadAttention(embed_dim=self.latent_dim, num_heads=num_heads, batch_first=True)
        self.look_ahead_mask = self._generate_look_ahead_mask(seq_length)
        self.fc_after_attention = nn.Linear(self.latent_dim, self.latent_dim)
        self.gelu = nn.GELU()

        # Fully Connected Block
        self.layer_norm_fc_1 = nn.LayerNorm(self.latent_dim)
        self.fc_1 = nn.Linear(self.latent_dim, self.latent_dim)
        self.layer_norm_fc_2 = nn.LayerNorm(self.latent_dim)
        self.fc_2 = nn.Linear(self.latent_dim, self.latent_dim)

        # Output Block
        self.layer_norm_out = nn.LayerNorm(self.latent_dim)
        self.output_layer = nn.Linear(self.latent_dim, vocab_size)

        self.to(self.device)

    def _generate_look_ahead_mask(self, size):
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        mask[mask == 1] = float('-inf')
        mask[mask == 0] = 0
        return mask.to(self.device)

    def create_padding_mask(self, x):
        # Padding tokens are assumed to be represented by 
        return (x == self.padding_token).to(self.device)  # (batch_size, seq_length)
    
    def pad_sequence(self, x, target_length):
        """Pads sequences in x to the target length with the padding token."""
        batch_size, seq_length = x.size()
        if seq_length < target_length:
            padding = torch.full((batch_size, target_length - seq_length), self.vocab_size, dtype=torch.long).to(self.device)
            x = torch.cat([x, padding], dim=1)
        return x

    def forward(self, x):
        x = x.to(self.device)
        x = self.pad_sequence(x, self.seq_length)
        batch_size, seq_length = x.size()

        # Pre-processing
        token_embeddings = self.token_embedding(x)  # (batch_size, seq_length, embedding_dim)
        positional_embeddings = self.positional_embedding[:seq_length, :].unsqueeze(0).to(self.device)  # (1, seq_length, embedding_dim)
        embeddings = token_embeddings + positional_embeddings  # (batch_size, seq_length, embedding_dim)
        embeddings = self.repeat_heads(embeddings)  # (batch_size, seq_length, latent_dim)

        # Create padding mask
        padding_mask = self.create_padding_mask(x)  # (batch_size, seq_length)

        # Attention Block
        attn_input = self.layer_norm_1(embeddings)
        attn_output, _ = self.attention(attn_input, attn_input, attn_input, key_padding_mask=padding_mask, attn_mask=self.look_ahead_mask)
        attn_output = self.gelu(attn_output)
        attn_output = self.fc_after_attention(attn_output)
        attn_output = self.gelu(attn_output)
        attn_output = attn_output + embeddings  # Skip connection

        # Fully Connected Block
        fc_input = self.layer_norm_fc_1(attn_output)
        fc_output = self.gelu(self.fc_1(fc_input))
        fc_output = self.layer_norm_fc_2(fc_output)
        fc_output = self.gelu(self.fc_2(fc_output))
        fc_output = fc_output + attn_output  # Skip connection

        # Output Block
        output_input = self.layer_norm_out(fc_output)
        output = self.output_layer(output_input)  # (batch_size, seq_length, vocab_size)

        # Return only the logits for the current token position
        result = output[:, -1, :]
        return result  # Shape: (batch_size, vocab_size)
