import math
import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, feed_forward_size, num_heads, dropout_rate):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, dropout=dropout_rate
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, feed_forward_size),
            nn.ReLU(),
            nn.Linear(feed_forward_size, hidden_size),
        )
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, src_mask=None):
        attn_output, _ = self.attention(x, x, x, key_padding_mask=src_mask)
        x = x + self.dropout(attn_output)
        x = self.layer_norm1(x)

        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.layer_norm2(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create constant 'pe' matrix with values dependent on
        # pos and i
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # Add positional encoding to each token
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix with shape (max_len, hidden_size)
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dimension (B x L x D)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        # Make pe the same size as x, up to seq_len
        x = x + self.pe[:, : x.size(0)]
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(
        self,
        num_blocks,
        hidden_size,
        feed_forward_size,
        num_heads,
        dropout_rate,
        vocab_size,
        max_seq_length,
    ):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size, dropout_rate, max_seq_length)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size, feed_forward_size, num_heads, dropout_rate
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(hidden_size)
        self.output_layer = nn.Linear(hidden_size, 2)

    def forward(self, x, src_mask=None):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        x = self.final_layer_norm(x)
        x = self.output_layer(x.mean(dim=1))

        mean, log_var = x[:, 0], x[:, 1]
        return mean, log_var
