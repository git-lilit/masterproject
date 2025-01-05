import numpy as np


class RandomSplit:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def split(self, train_size=0.5, seed=None):
        if seed is not None:
            np.random.seed(seed)
        indices = np.arange(len(self.x))
        np.random.shuffle(indices)
        split_idx = int(train_size * len(self.x))
        train_indices, test_indices = indices[:split_idx], indices[split_idx:]
        return (
            self.x[train_indices],
            self.y[train_indices],
            self.x[test_indices],
            self.y[test_indices],
        )


class PercentileSplit:
    def __init__(self, x, y):
        """
        Initialize the PercentileSplit class.

        Parameters:
        x (numpy.ndarray): Feature data with shape (n_samples, n_features).
        y (numpy.ndarray): Target data with shape (n_samples,).
        """
        self.x = x
        self.y = y

    def split(
        self, lower_percentile=15, upper_percentile=85, sample_fraction=0.7, seed=None
    ):
        """
        Filter the middle percentage of data and split into training and testing sets.

        Parameters:
        lower_percentile (float): The lower percentile for filtering.
        upper_percentile (float): The upper percentile for filtering.
        sample_fraction (float): Fraction of filtered data to include in the training set.
        seed (int): Seed for reproducibility.

        Returns:
        x_train, y_train, x_test, y_test: Training and testing splits.
        """
        if seed is not None:
            np.random.seed(seed)

        # Calculate percentile thresholds
        lower_bound = np.percentile(self.y, lower_percentile)
        upper_bound = np.percentile(self.y, upper_percentile)

        # Create mask for middle data
        middle_mask = (self.y >= lower_bound) & (self.y <= upper_bound)

        # Filter x and y using the mask
        x_filtered = self.x[middle_mask.squeeze(1), :]
        y_filtered = self.y[middle_mask].reshape(-1, 1)

        # Determine the size of the training set
        sample_size = int(sample_fraction * len(y_filtered))

        # Sample indices for the training set
        indices = np.random.choice(len(y_filtered), size=sample_size, replace=False)
        train_mask = np.zeros(len(y_filtered), dtype=bool)
        train_mask[indices] = True

        # Split into training and testing sets
        x_train, y_train = x_filtered[train_mask], y_filtered[train_mask]
        x_test, y_test = x_filtered[~train_mask], y_filtered[~train_mask]

        return x_train, y_train, x_test, y_test


class WeightedSplit:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def split(self, train_size=0.8, good_sample_proportion=0.1):
        sorted_indices = np.argsort(self.y)
        split_idx = int(train_size * len(self.y))

        # Low values in training
        low_indices = sorted_indices[:split_idx]
        high_indices = sorted_indices[split_idx:]

        # Select a subset of high values to add to the training set
        num_good_samples = int(good_sample_proportion * len(self.y))
        selected_high_indices = np.random.choice(
            high_indices, num_good_samples, replace=False
        )

        train_indices = np.concatenate([low_indices, selected_high_indices])
        test_indices = np.setdiff1d(np.arange(len(self.y)), train_indices)

        return (
            self.x[train_indices],
            self.y[train_indices],
            self.x[test_indices],
            self.y[test_indices],
        )


def get_saved_dataset():
    file_path_x = "data/tf_bind_8-x-0.npy"
    file_path_y = "data/tf_bind_8-y-0.npy"

    X = np.load(file_path_x)
    print(X.shape)
    y = np.load(file_path_y)

    return X, y

