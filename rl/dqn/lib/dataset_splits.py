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


class SortedSplit:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def split(self, train_size=0.8):
        sorted_indices = np.argsort(self.y)
        split_idx = int(train_size * len(self.y))
        train_indices = sorted_indices[:split_idx]
        test_indices = sorted_indices[split_idx:]
        return (
            self.x[train_indices],
            self.y[train_indices],
            self.x[test_indices],
            self.y[test_indices],
        )


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
        y = np.load(file_path_y)

        return X, y
