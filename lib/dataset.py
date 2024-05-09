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


def load_data_from_files(file_path_x, file_path_y):
    X = np.load(file_path_x)
    y = np.load(file_path_y)
    return X, y


def shuffle_and_split_data(X, y, train_ratio=0.8):
    num_samples = len(X)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    num_train_samples = int(num_samples * train_ratio)
    train_indices = indices[:num_train_samples]
    test_indices = indices[num_train_samples:]

    return train_indices, test_indices


def create_datasets(X, y, train_indices, test_indices):
    train_dataset = NumpyDataset(X[train_indices], y[train_indices])
    test_dataset = NumpyDataset(X[test_indices], y[test_indices])
    return train_dataset, test_dataset


def load_saved_dataset(file_path_x, file_path_y):
    X = np.load(file_path_x)
    y = np.load(file_path_y)
    return NumpyDataset(X, y)


def save_dataset(dataset, file_path_x, file_path_y):
    features = dataset.features.numpy()
    labels = dataset.labels.numpy()
    np.save(file_path_x, features)
    np.save(file_path_y, labels)


def load_and_split(train_ratio=0.8, is_test_split=False):
    # This function either does train/test split for initial splitting or train/val split for later splits
    if is_test_split:
        file_path_x = "data/tf_bind_8-x-0.npy"
        file_path_y = "data/tf_bind_8-y-0.npy"
    else:
        file_path_x = "data/train_tf_bind_8-x-0.npy"
        file_path_y = "data/train_tf_bind_8-y-0.npy"

    X, y = load_data_from_files(file_path_x, file_path_y)
    train_indices, rest_indices = shuffle_and_split_data(X, y, train_ratio)
    train_dataset, rest_dataset = create_datasets(X, y, train_indices, rest_indices)

    return train_dataset, rest_dataset


def get_test_data():
    file_path_x = "data/test_tf_bind_8-x-0.npy"
    file_path_y = "data/test_tf_bind_8-y-0.npy"

    return load_saved_dataset(file_path_x, file_path_y)