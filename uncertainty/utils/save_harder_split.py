import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.dataset import get_saved_data, create_datasets, save_dataset, split_by_median

X, y = get_saved_data(dataset="full")
train_indices, test_indices = split_by_median(X, y)
train_dataset, test_dataset = create_datasets(X, y, train_indices, test_indices)

train_file_path_x = "data/train_tf_bind_8_hard-x-0.npy"
train_file_path_y = "data/train_tf_bind_8_hard-y-0.npy"
test_file_path_x = "data/test_tf_bind_8_hard-x-0.npy"
test_file_path_y = "data/test_tf_bind_8_hard-y-0.npy"

save_dataset(train_dataset, train_file_path_x, train_file_path_y)
save_dataset(test_dataset, test_file_path_x, test_file_path_y)
