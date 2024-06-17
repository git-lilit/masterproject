import os
import sys
import torch
import pickle
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.transformer import TransformerModel


def get_top_models_data(num_models, filename):
    df = pd.read_csv(filename)
    df_sorted = df.sort_values(by="value", ascending=True)

    top_k_model_data = df_sorted.head(num_models)
    return top_k_model_data


def load_model(model_id, model_params, base_dir="saved_models/hyperopt"):
    model_path = os.path.join(base_dir, f"model_{model_id}.pth")
    model = TransformerModel(**model_params)
    model.load_state_dict(torch.load(model_path))
    return model


def get_top_models(num_models, filename, base_dir):
    top_k_model_data = get_top_models_data(num_models, filename)
    models = []

    if "hetero" in filename:
        num_outputs = 2
    else: 
        num_outputs = "homoscedastic"

    for _, row in top_k_model_data.iterrows():
        model_id = row["number"]

        model_params = {
            "ntoken": 4,
            "d_model": int(row["params_d_model"]),
            "nhead": int(row["params_nhead"]),
            "d_hid": int(row["params_d_hid"]),
            "nlayers": int(row["params_nlayers"]),
            "dropout": float(row["params_dropout"]),
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            "num_outputs": num_outputs
        }

        model = load_model(model_id, model_params, base_dir)
        models.append(model)

    return models


def get_best_params(filename):
    df = get_top_models_data(1, filename)
    row = df.iloc[0]

    if "hetero" in filename:
        num_outputs = 2
    else: 
        num_outputs = "homoscedastic"

    train_params = {
        "num_epochs": 30,
        "lr": df.iloc[0]["params_lr"],
        "batch_size": int(df.iloc[0]["params_batch_size"]),
        "patience": 5,
    }

    model_params = {
        "ntoken": 4,
        "d_model": int(row["params_d_model"]),
        "nhead": int(row["params_nhead"]),
        "d_hid": int(row["params_d_hid"]),
        "nlayers": int(row["params_nlayers"]),
        "dropout": float(row["params_dropout"]),
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "num_outputs": num_outputs
    }

    return train_params, model_params


def load_predictions(model_name, predictions_folder="../predictions"):
    predictions_path = os.path.join(predictions_folder, f"{model_name}_pred.pkl")

    with open(predictions_path, "rb") as f:
        return pickle.load(f)
