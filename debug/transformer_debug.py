from lib.transformer import TransformerModel
from lib.dataset import get_train_data
import os
import torch
import wandb
from tqdm import tqdm
from torch import optim
from torch.nn import GaussianNLLLoss, MSELoss
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lib.training import *
from lib.dataset import get_test_dataset
from lib.regression import RegressionFromTokens
from lib.metrics import calculate_correlations

device = initialize_device()

train_params = {
    "num_epochs": 30,
    "lr": 0.0005,
    "batch_size": 128,
    "model_id": 0,
    "patience": 5,
    "model_type": "transformer"
}

model_params = {
    "ntoken": 4,
    "d_model": 128,
    "nhead": 8,
    "d_hid": 256,
    "nlayers": 4,
    "dropout": 0.4,
    "num_outputs": "homoscedastic",
    "device":  device
}

model = train_model(train_params, model_params, save_model=True, folder_name="transformer_debug")[0]

model_path = os.path.join("transformer_debug", "0.pth")
model = TransformerModel(**model_params)
model.load_state_dict(torch.load(model_path))

test_dataset = get_test_dataset()
targets = test_dataset.labels


x = test_dataset.features.transpose(0, 1)
predictions = model(torch.tensor(x))

print(targets.shape)
print(predictions.shape)
print(calculate_correlations(predictions.detach().numpy().reshape(-1), targets.reshape(-1)))


