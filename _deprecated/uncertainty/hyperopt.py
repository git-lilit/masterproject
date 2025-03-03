import torch
import optuna
from lib.training import train_model, initialize_device
from lib.dataset import get_saved_dataset, split_dataset
from lib.metrics import calculate_correlations
from torch.utils.data import DataLoader

dataset = get_saved_dataset(type="train_random")
train_dataset, val_dataset = split_dataset(
    dataset, train_ratio=0.8, split_type="random"
)


def get_dataloaders(train_dataset, val_dataset, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader


def objective(trial):
    model_id = f"model_{trial.number}"
    lr = trial.suggest_float("lr", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

    d_model = trial.suggest_categorical("d_model", [32, 64, 128, 256])
    d_hid = trial.suggest_categorical("d_hid", [32, 64, 128, 256])
    ff_size = trial.suggest_categorical("ff_size", [32, 64, 128, 256])
    nhead = trial.suggest_categorical("nhead", [2, 4, 8])
    nlayers = trial.suggest_int("nlayers", 2, 4)
    dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)

    device = initialize_device()

    train_params = {
        "num_epochs": 40,
        "patience": 5,
        "model_type": "transformer",
        "model_id": model_id,
        "lr": lr,
        "batch_size": batch_size,
    }

    model_params = {
        "ntoken": 4,
        "num_outputs": 2,
        "device": device,
        "d_model": d_model,
        "nhead": nhead,
        "d_hid": d_hid,
        "nlayers": nlayers,
        "dropout": dropout,
        "ff_size": ff_size,
    }

    train_loader, val_loader = get_dataloaders(train_dataset, val_dataset, batch_size)

    model, _ = train_model(
        train_params,
        model_params,
        save_model=True,
        folder_name="saved_models/hyperopt_hetero",
        loaders=(train_loader, val_loader),
    )

    targets = val_dataset.labels.reshape(-1)
    input = torch.tensor(val_dataset.features.transpose(0, 1)).to(device)
    predictions = model(input)

    if model_params["num_outputs"] == 2:
        predictions = predictions[0]

    predictions = predictions.reshape(-1).detach().cpu().numpy()
    correlations = calculate_correlations(predictions, targets)

    return correlations["pearson_corr"]


def main():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    df = study.trials_dataframe()
    df.to_csv("study_results/study_results_ff_hetero.csv", index=False)


if __name__ == "__main__":
    main()
