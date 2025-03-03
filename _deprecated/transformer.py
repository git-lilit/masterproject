import os
import math
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerModel(nn.Module):
    def __init__(
        self,
        ntoken: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        ff_size: int,
        num_outputs: int,
    ):
        super().__init__()
        self.model_type = "Transformer"
        self.d_model = d_model
        self.device = initialize_device()
        self.num_outputs = num_outputs

        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout=0)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, d_model)

        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(d_model, ff_size),
            torch.nn.ReLU(),
            torch.nn.Linear(ff_size, num_outputs),
        )

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        for layer in self.feed_forward:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(-initrange, initrange)
                layer.bias.data.zero_()

    def forward(self, src: Tensor) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """

        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)

        # # Permute the dimensions to [batch_size, seq_len, d_model]
        # permuted_tensor = output.permute(1, 0, 2)

        # # Reshape the tensor to [batch_size, seq_len * d_model]
        # reshaped_tensor = permuted_tensor.contiguous().view(batch_size, -1)

        output = self.feed_forward(output.mean(dim=1))

        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return x


def initialize_device():
    os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
