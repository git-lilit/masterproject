import torch
import torch.nn as nn
import torch.nn.functional as F


class SequenceModel(nn.Module):
    def __init__(
        self, vocab_size, embedding_dim, num_heads, output_dim, seq_len, device="cpu"
    ):
        super(SequenceModel, self).__init__()
        self.device = device
        self.start_token_id = vocab_size - 2  # Second-to-last ID is start token
        self.pad_token_id = vocab_size - 1  # Last ID is padding token
        self.hidden_dim = num_heads * embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_embedding = nn.Embedding(seq_len + 1, embedding_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim, num_heads=num_heads, batch_first=True
        )
        self.fc1 = nn.Linear(embedding_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, output_dim)
        self.layernorm1 = nn.LayerNorm(embedding_dim)
        self.layernorm2 = nn.LayerNorm(self.hidden_dim)
        self.gelu = nn.GELU()

        # self._initialize_weights()
        self.to(self.device)

    def forward(self, x, seq_lens=None):
        # Add start token and pad sequences to match the longest in the batch
        x = self.preprocess_input(x, seq_lens)

        # Embedding + Positional Embedding
        token_embeddings = self.embedding(x)
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        position_embeddings = self.positional_embedding(positions)
        embeddings = token_embeddings + position_embeddings

        # Key padding mask (indicates padding positions with True)
        key_padding_mask = x == 0  # Shape: [batch_size, seq_len]

        # Attention mask for causal masking (disallows attending to future tokens)
        seq_len = x.size(1)
        attn_mask = torch.triu(
            torch.ones(seq_len, seq_len), diagonal=1
        )  # Upper triangular mask

        # Ensure types and devices match
        key_padding_mask = key_padding_mask.to(dtype=torch.float32).to(self.device)
        attn_mask = attn_mask.to(dtype=torch.float32).to(self.device)

        # Attention Block
        embeddings = self.layernorm1(embeddings)
        attention_output, _ = self.attention(
            embeddings,
            embeddings,
            embeddings,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
        )
        attention_output = self.gelu(attention_output)

        # Fully Connected Block
        fc_output = self.layernorm2(self.fc1(attention_output))
        fc_output = self.gelu(fc_output)

        # Output Block: Use the output of the start token (first token)
        start_token_output = fc_output[:, 0, :]  # First token for each sequence
        output = self.fc2(start_token_output)
        return output

    def generate_look_ahead_mask(self, size):
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        mask[mask == 1] = float("-inf")
        mask[mask == 0] = 0
        return mask.to(self.device)

    def create_padding_mask(self, x):
        # Padding tokens are assumed to be represented by
        return (x == self.pad_token_id).to(self.device)  # (batch_size, seq_length)

    def preprocess_input(self, x, seq_lens=None):
        """
        Add start token, pad sequences, and create attention masks.
        """
        batch_size, original_seq_len = x.size()
        device = x.device

        # Determine sequence lengths and max length
        if seq_lens is None:
            seq_lens = (
                torch.sum(x != self.pad_token_id, dim=1) + 1
            )  # Include start token
        max_len = seq_lens.max().item()

        # Pad sequences to max_len
        padded_x = torch.full((batch_size, max_len), self.pad_token_id, device=device)
        padded_x[:, : x.size(1)] = x

        return padded_x

    # def _initialize_weights(self):
    #     # Xavier initialization for weights and zero for biases
    #     for module in self.modules():
    #         if isinstance(module, nn.Linear):
    #             nn.init.xavier_uniform_(module.weight)
    #             if module.bias is not None:
    #                 nn.init.zeros_(module.bias)
    #         elif isinstance(module, nn.Embedding):
    #             nn.init.xavier_uniform_(module.weight)
    #         elif isinstance(module, nn.MultiheadAttention):
    #             for param in module.parameters():
    #                 if param.ndimension() > 1:  # Weight parameters
    #                     nn.init.xavier_uniform_(param)
    #                 else:  # Bias parameters
    #                     nn.init.zeros_(param)
