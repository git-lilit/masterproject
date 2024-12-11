import os
import torch
import random
import numpy as np
from scipy.integrate import simpson


def set_seed(n: int) -> None:
    # Taken from d3rlpy __init__
    """Sets random seed value.

    Args:
        n (int): seed value.

    """
    random.seed(n)
    np.random.seed(n)
    torch.manual_seed(n)
    torch.cuda.manual_seed(n)
    # torch.backends.cudnn.deterministic = True # moved to the main as it caused errors here, see https://github.com/facebookresearch/hydra/issues/1180


def num_arguments(model):
    # Separate trainable and non-trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    print(f"Trainable parameters: {trainable_params}")
    print(f"Non-trainable parameters: {non_trainable_params}")


def initialize_device():
    os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_auc(interval_stats):
    rewards = interval_stats['seq_reward']
    x = np.arange(len(rewards)) + 1

    max_area = len(x) - 1

    # Compute the normalized area under the curve using Simpson's rule
    return simpson(rewards, x=x) / max_area
