import os
import torch
import wandb
from tqdm import tqdm
from torch import optim
from torch.nn import GaussianNLLLoss, MSELoss
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from lib.dataset import load_and_split
from lib.transformer import TransformerModel


def initialize_device():
    os.environ["CUDA_VISIBLE_DEVICES"] = "5,7"
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_data_loaders(train_params, loaders=None):
    if loaders is None:
        train_dataset, val_dataset = load_and_split(use_full_data=False)
        train_loader = DataLoader(
            train_dataset, batch_size=train_params["batch_size"], shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=train_params["batch_size"], shuffle=True
        )
    else:
        train_loader, val_loader = loaders
    return train_loader, val_loader


def setup_components(model_params, train_params):
    model = TransformerModel(**model_params).to(model_params["device"])
    optimizer = optim.Adam(model.parameters(), lr=train_params["lr"])
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=3, verbose=True
    )
    return model, optimizer, scheduler


def log_model_info(model, params):
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(params)
    print(f"Number of trainable parameters: {num_trainable_params}")
    return num_trainable_params


def train_epoch(model, train_loader, optimizer, loss_fn, error_type, device):
    model.train()
    total_train_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        if error_type == "heteroscedastic":
            mean_pred, log_var_pred = model(inputs)
            var_pred = torch.exp(log_var_pred)
            loss = loss_fn(mean_pred, targets, var_pred)
        else:
            y_pred = model(inputs)
            loss = loss_fn(y_pred, targets)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    return total_train_loss / len(train_loader)


def validate_epoch(model, val_loader, loss_fn, error_type, device):
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            if error_type == "heteroscedastic":
                mean_pred, log_var_pred = model(inputs)
                var_pred = torch.exp(log_var_pred)
                loss = loss_fn(mean_pred, targets, var_pred)
            else:
                y_pred = model(inputs)
                loss = loss_fn(y_pred, targets)
            total_val_loss += loss.item()
    return total_val_loss / len(val_loader)


def train_model(train_params, model_params, save_model, folder_name, loaders=None):
    device = initialize_device()
    model_params["device"] = device
    error_type = model_params["error_type"]

    train_loader, val_loader = prepare_data_loaders(train_params, loaders)
    model, optimizer, scheduler = setup_components(model_params, train_params)

    if error_type == "heteroscedastic":
        loss_fn = GaussianNLLLoss()
    else:
        loss_fn = MSELoss()

    num_trainable_params = log_model_info(model, {**train_params, **model_params})

    wandb.init(
        project="tf-bind-initial",
        config={**model_params, **train_params, "num_params": num_trainable_params},
    )

    best_loss = float("inf")
    early_stopping_counter = 0
    best_model_state = None

    for epoch in tqdm(range(train_params["num_epochs"]), desc="Epochs"):
        avg_train_loss = train_epoch(model, train_loader, optimizer, loss_fn, error_type, device)
        avg_val_loss = validate_epoch(model, val_loader, loss_fn, error_type, device)

        print(f"Epoch {epoch+1}, Training Average Loss: {avg_train_loss}")
        print(f"Epoch {epoch+1}, Validation Average Loss: {avg_val_loss}")
        wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss})

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            early_stopping_counter = 0
            best_model_state = model.state_dict().copy()  # Save the best model state
        else:
            early_stopping_counter += 1
            print(
                f"No improvement in validation loss for {early_stopping_counter} epochs."
            )
            if early_stopping_counter >= train_params["patience"]:
                print("Stopping early due to lack of improvement in validation loss.")
                break

        if save_model:
            model_path = f"./{folder_name}/{train_params['model_id']}.pth"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(best_model_state, model_path)  # Save the best model state

    model.load_state_dict(
        best_model_state
    )  # Load the best model state before returning
    return model, best_loss
