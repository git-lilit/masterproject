import torch
import optuna
from lib.training import train_model


def objective(trial):
    model_id = f"model_{trial.number}"
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

    d_model = trial.suggest_categorical("d_model", [32, 64, 128])
    d_hid = trial.suggest_categorical(
        "d_hid", [32, 64, 128, 256]
    )
    nhead = trial.suggest_categorical("nhead", [1, 2, 4])
    nlayers = trial.suggest_int("nlayers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)

    train_params = {
        "num_epochs": 30,
        "lr": lr,
        "batch_size": batch_size,
        "model_id": model_id,
        "patience": 5,
    }

    model_params = {
        "ntoken": 4, 
        "d_model": d_model,
        "nhead": nhead, 
        "d_hid": d_hid,
        "nlayers": nlayers, 
        "dropout": dropout,
        "error_type": "homoscedastic",
        "device":  torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }

    model, loss = train_model(
        train_params, model_params, save_model=True, folder_name="saved_models/hyperopt3"
    )

    return loss


def main():
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)

    df = study.trials_dataframe()

    df.to_csv("study_results3.csv", index=False)


if __name__ == "__main__":
    main()
