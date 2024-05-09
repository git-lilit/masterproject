import numpy as np
from lib.dataset import *

train_dataset, test_dataset = load_and_split(is_test_split=True)

train_file_path_x = "data/train_tf_bind_8-x-0.npy"
train_file_path_y = "data/train_tf_bind_8-y-0.npy"
test_file_path_x = "data/test_tf_bind_8-x-0.npy"
test_file_path_y = "data/test_tf_bind_8-y-0.npy"

save_dataset(train_dataset, train_file_path_x, train_file_path_y)
save_dataset(test_dataset, test_file_path_x, test_file_path_y)
