import numpy as np
import torch
from torch.utils.data import Dataset


class NumpyDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


def split_randomly(X, y, train_ratio=0.8):
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")

    num_samples = len(X)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    num_train_samples = int(num_samples * train_ratio)

    train_indices = indices[:num_train_samples]
    test_indices = indices[num_train_samples:]

    return train_indices, test_indices


def split_ordered(X, y, train_ratio=0.8):
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")

    sorted_indices = sorted(range(len(y)), key=lambda x: y[x])
    split_idx = int(len(y) * train_ratio)

    train_indices = sorted_indices[:split_idx]
    test_indices = sorted_indices[split_idx:]

    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)

    return train_indices, test_indices


def split_dataset(dataset, split_type="random", train_ratio=0.8):
    X, y = dataset.features, dataset.labels
    if split_type == "random":
        train_indices, rest_indices = split_randomly(X, y, train_ratio)
    elif split_type == "ordered":
        train_indices, rest_indices = split_ordered(X, y, train_ratio)

    train_dataset = NumpyDataset(X[train_indices], y[train_indices])
    rest_dataset = NumpyDataset(X[rest_indices], y[rest_indices])

    return train_dataset, rest_dataset


def get_saved_dataset(type="full"):
    if type == "full":
        file_path_x = "data/tf_bind_8-x-0.npy"
        file_path_y = "data/tf_bind_8-y-0.npy"
    elif type == "train_random":
        file_path_x = "data/train_tf_bind_8-x-0.npy"
        file_path_y = "data/train_tf_bind_8-y-0.npy"
    elif type == "train_hard":
        file_path_x = "data/train_tf_bind_8_hard-x-0.npy"
        file_path_y = "data/train_tf_bind_8_hard-y-0.npy"
    elif type == "test_random":
        file_path_x = "data/test_tf_bind_8-x-0.npy"
        file_path_y = "data/test_tf_bind_8-y-0.npy"
    elif type == "test_hard":
        file_path_x = "data/test_tf_bind_8_hard-x-0.npy"
        file_path_y = "data/test_tf_bind_8_hard-y-0.npy"
    else:
        raise ValueError("Dataset type incorrect.")
    
    X = np.load(file_path_x)
    y = np.load(file_path_y)

    return NumpyDataset(X, y)


def save_dataset(dataset, file_path_x, file_path_y):
    features = dataset.features.numpy()
    labels = dataset.labels.numpy()
    np.save(file_path_x, features)
    np.save(file_path_y, labels)