import torch

class RegressionFromTokens(torch.nn.Module):
    def __init__(self, num_tokens, num_outputs, *args, **kwargs):
        super(RegressionFromTokens, self).__init__()
        self.embedding = torch.nn.Embedding(num_tokens, 32)
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, num_outputs),
        )

    def forward(self, x):
        x = self.embedding(x)
        x = torch.flatten(x, start_dim=1)  # produces 8 * 32 = 256
        return self.ff(x)
    

