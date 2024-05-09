import optuna
from lib.training import train_model


def objective(trial):
    model_id = f"model_{trial.number}"
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

    hidden_size = trial.suggest_categorical("hidden_size", [24, 32, 64, 128, 256])
    feed_forward_size = trial.suggest_categorical(
        "feed_forward_size", [64, 128, 256, 512]
    )
    num_heads = trial.suggest_categorical("num_heads", [1, 2, 4, 8])
    num_blocks = trial.suggest_int("num_blocks", 1, 4)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.4, step=0.1)

    train_params = {
        "num_epochs": 30,
        "lr": lr,
        "batch_size": batch_size,
        "model_id": model_id,
        "patience": 5,
    }

    model_params = {
        "hidden_size": hidden_size,
        "feed_forward_size": feed_forward_size,
        "num_heads": num_heads,
        "num_blocks": num_blocks,
        "dropout_rate": dropout_rate,
        "max_seq_length": 8,
        "vocab_size": 4,
    }

    model, loss = train_model(
        train_params, model_params, save_model=True, folder_name="saved_models/hyperopt"
    )

    return loss


def main():
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=5)

    df = study.trials_dataframe()

    df.to_csv("study_results.csv", index=False)


if __name__ == "__main__":
    main()
