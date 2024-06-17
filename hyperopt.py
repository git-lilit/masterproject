import torch
import optuna
from lib.training import train_model
from lib.dataset import get_test_dataset
from lib.metrics import calculate_correlations


def objective(trial):
    model_id = f"model_{trial.number}"
    lr = trial.suggest_float("lr", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

    d_model = trial.suggest_categorical("d_model", [32, 64, 128, 256])
    d_hid = trial.suggest_categorical(
        "d_hid", [32, 64, 128, 256]
    )
    ff_size = trial.suggest_categorical(
        "d_hid", [32, 64, 128, 256]
    )
    nhead = trial.suggest_categorical("nhead", [2, 4, 8])
    nlayers = trial.suggest_int("nlayers", 2, 4)
    dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)

    train_params = {
        "num_epochs": 40,
        "lr": lr,
        "batch_size": batch_size,
        "model_id": model_id,
        "patience": 5,
        "model_type": "transformer"
    }

    model_params = {
        "ntoken": 4, 
        "d_model": d_model,
        "nhead": nhead, 
        "d_hid": d_hid,
        "nlayers": nlayers, 
        "dropout": dropout,
        "num_outputs": 1,
        "ff_size": ff_size,
        "device":  torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }

    model, loss = train_model(
        train_params, model_params, save_model=True, folder_name="saved_models/hyperopt3_homo"
    )

    test_dataset = get_test_dataset()
    targets = test_dataset.labels.reshape(-1)
    input = torch.tensor(test_dataset.features.transpose(0, 1)).to(model_params["device"])
    predictions = model(input)
    predictions = predictions.reshape(-1).detach().cpu().numpy()

    correlations = calculate_correlations(predictions, targets)

    return correlations["pearson_corr"]


def main():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    df = study.trials_dataframe()

    df.to_csv("study_results3_homo.csv", index=False)


if __name__ == "__main__":
    main()
