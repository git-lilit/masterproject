import os
import torch
import optuna
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from lib.dataset import load_and_split
from lib.transformer import Transformer
from lib.losses import gaussian_nll_loss
from torch.utils.data import DataLoader
import wandb


def train_model(train_params, model_params, save_model, folder_name):
    train_dataset, val_dataset = load_and_split(is_test_split=False)

    train_loader = DataLoader(
        train_dataset, batch_size=train_params["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=train_params["batch_size"], shuffle=True
    )

    model = Transformer(**model_params)
    print(model)

    # Calculate the number of trainable parameters
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_trainable_params}")

    optimizer = optim.Adam(model.parameters(), lr=train_params["lr"])

    # Early stopping parameters
    patience = train_params["patience"]  # Number of epochs to wait for improvement before stopping
    best_loss = float("inf")
    patience_counter = 0

    wandb.init(
        project="tf-bind-initial",
        config={**model_params, **train_params, "num_params": num_trainable_params},
    )

    for epoch in tqdm(range(train_params["num_epochs"]), desc="Epochs"):
        model.train()
        total_train_loss = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            mean_pred, log_var_pred = model(inputs)
            loss = gaussian_nll_loss(mean_pred, log_var_pred, targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Training Average Loss: {avg_train_loss}")

        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                mean_pred, log_var_pred = model(inputs)
                loss = gaussian_nll_loss(mean_pred, log_var_pred, targets)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1}, Validation Average Loss: {avg_val_loss}")
        wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss})

        # Early stopping logic
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement in validation loss for {patience_counter} epochs.")
            if patience_counter >= patience:
                print("Stopping early due to lack of improvement in validation loss.")
                break

        if save_model:
            model_path = f"./{folder_name}/{train_params["model_id"]}.pth"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)

    return model, best_loss