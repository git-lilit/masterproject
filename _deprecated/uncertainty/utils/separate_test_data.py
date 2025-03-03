import numpy as np
from lib.dataset import load_and_random_split, save_dataset

train_dataset, test_dataset = load_and_random_split(train_ratio=0.8, dataset="full")

train_file_path_x = "data/train_tf_bind_8-x-0.npy"
train_file_path_y = "data/train_tf_bind_8-y-0.npy"
test_file_path_x = "data/test_tf_bind_8-x-0.npy"
test_file_path_y = "data/test_tf_bind_8-y-0.npy"

save_dataset(train_dataset, train_file_path_x, train_file_path_y)
save_dataset(test_dataset, test_file_path_x, test_file_path_y)
