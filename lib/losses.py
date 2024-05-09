import torch

def gaussian_nll_loss(mean, log_var, target):
    """
    Compute the Gaussian Negative Log-Likelihood loss.

    Args:
    - mean (torch.Tensor): Predicted means from the model.
    - log_var (torch.Tensor): Predicted log variances from the model.
    - target (torch.Tensor): Actual target values.

    Returns:
    - loss (torch.Tensor): The computed Gaussian NLL loss.
    """
    variance = torch.exp(log_var)
    loss = 0.5 * (torch.log(variance) + (target - mean) ** 2 / variance)

    return loss.mean()
