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

def load_model(model_id, model_params, base_dir="saved_models/hyperopt"):
   
    return model


train_params = {
    "num_epochs": 30,
    "lr": 0.0005,
    "batch_size": 64,
    "model_id": 0,
    "patience": 5,
    "model_type": "regression"
}

model_params = {
    "num_tokens": 4,
    "num_outputs": 1,
    "num_outputs": "homoscedastic",
    "model_params": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

model = train_model(train_params, model_params, save_model=True, folder_name="regression")[0]

model_path = os.path.join("regression", "0.pth")
model = RegressionFromTokens(**model_params)
model.load_state_dict(torch.load(model_path))

test_dataset = get_test_dataset()
targets = test_dataset.labels

predictions = model(torch.tensor(test_dataset.features))

print(targets.shape)
print(predictions.shape)
print(calculate_correlations(predictions[0].detach().numpy().reshape(-1), targets.reshape(-1)))
