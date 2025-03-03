import numpy as np

x = np.load("data/tf_bind_8-x-0.npy")
y = np.load("data/tf_bind_8-y-0.npy")

print(x.shape)
print(y.shape)


# Calculate minimum and maximum values to find the range
min_value = np.min(y)
max_value = np.max(y)

# Calculate mean and standard deviation
mean_value = np.mean(y)
std_dev = np.std(y)

print(mean_value)
print(min_value, max_value)
print(std_dev, max_value - min_value)

import math
RMSE = math.sqrt(0.023863995252431887)
print(RMSE, "RMSE")