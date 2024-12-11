import numpy as np
import torch


class TFBindChecker:
    def __init__(self, data_folder: str = "data"):
        """
        Initializes the TFBindChecker, loading data into memory.

        Args:
            data_folder (str): Path to the folder containing the dataset.
        """
        # Load the data from numpy files
        self.x_data = np.load(f"{data_folder}/tf_bind_8-x-0.npy")
        self.y_data = np.load(f"{data_folder}/tf_bind_8-y-0.npy")

        # Convert x_data to a torch tensor for efficient comparison
        self.x_data_tensor = torch.tensor(self.x_data, dtype=torch.long)
        self.y_data_tensor = torch.tensor(self.y_data, dtype=torch.float32)

    def test_fn(self, batch: torch.Tensor):
        """
        Finds the corresponding `y` values for a batch of sequences.

        Args:
            batch (torch.Tensor): A torch tensor of shape (batch_size, 8) representing sequences.

        Returns:
            torch.Tensor: The corresponding `y` values for the sequences in the batch. If a sequence
                        does not exist in the dataset, its score is set to NaN.
        """
        # Perform batch-wise comparison using broadcasting
        batch = batch.cpu()
        matches = (batch.unsqueeze(1) == self.x_data_tensor).all(dim=2)

        # Find indices of matches
        match_found = matches.any(dim=1)
        indices = matches.int().argmax(
            dim=1
        )  # This gives the first matching index per batch sequence

        # Create a result tensor filled with NaN for unmatched sequences
        result = torch.full((batch.size(0),), float("nan"), dtype=torch.float32)

        # Assign matching y values
        result[match_found] = self.y_data_tensor[indices[match_found]].squeeze(1)

        return result
