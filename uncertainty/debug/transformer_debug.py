import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from lib.training import initialize_device, train_model
from lib.dataset import get_saved_dataset, split_dataset
from lib.metrics import calculate_correlations
from lib.transformer import TransformerModel
from lib.model_loading import get_best_params
from torch.utils.data import DataLoader

device = initialize_device()
train_params, model_params = get_best_params("study_results/study_results_ff_homo.csv")
train_params["device"] = device
train_params["num_epochs"] = 40

train_dataset = get_saved_dataset("train_hard")
test_dataset = get_saved_dataset("test_hard")

val_dataset, test_dataset = split_dataset(test_dataset, train_ratio=0.5, split_type="random")

train_loader = DataLoader(
    train_dataset, batch_size=train_params["batch_size"], shuffle=True
)
val_loader = DataLoader(
    val_dataset, batch_size=train_params["batch_size"], shuffle=True
)

# model = train_model(train_params,
#                     model_params,
#                     save_model=False,
#                     loaders=[train_loader, val_loader])[0]

targets = test_dataset.labels
x = test_dataset.features.transpose(0, 1)
predictions = model(torch.tensor(x).to(device))

print(targets.shape)
# print(predictions[0].shape)

print(max(test_dataset.labels))
print(max(train_dataset.labels))


# print(max(targets))
# print(max(predictions[0]))
# print(calculate_correlations(predictions.detach().cpu().numpy().reshape(-1), targets.reshape(-1)))

# model_path = os.path.join("transformer_hard_split", "0.pth")
# model = TransformerModel(**model_params)
# model.load_state_dict(torch.load(model_path))