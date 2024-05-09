import os
import torch
import pandas as pd
from lib.transformer import Transformer

RESULTS_FILE = "study_results.csv"


def get_top_models_data(num_models):
    df = pd.read_csv(RESULTS_FILE)
    df_sorted = df.sort_values(by="value", ascending=True)

    top_k_model_data = df_sorted.head(num_models)
    return top_k_model_data


def load_model(model_id, model_params, base_dir="saved_models/hyperopt"):
    model_path = os.path.join(base_dir, f"model_{model_id}.pth")
    model = Transformer(**model_params)
    model.load_state_dict(torch.load(model_path))
    return model


def get_top_models(num_models):
    top_k_model_data = get_top_models_data(num_models)
    models = []

    for _, row in top_k_model_data.iterrows():
        model_id = row["number"]

        model_params = {
            "hidden_size": int(row["params_hidden_size"]),
            "feed_forward_size": int(row["params_feed_forward_size"]),
            "num_heads": int(row["params_num_heads"]),
            "num_blocks": int(row["params_num_blocks"]),
            "dropout_rate": float(row["params_dropout_rate"]),
            "max_seq_length": 8,
            "vocab_size": 4,
        }

        model = load_model(model_id, model_params)
        models.append(model)

    return models


def get_best_params():
    df = get_top_models_data(1)

    train_params = {
        "num_epochs": 30,
        "lr": df.iloc[0]['params_lr'],
        "batch_size": df.iloc[0]['params_batch_size'],
        "patience": 5
    }

    model_params = {
        "hidden_size": df.iloc[0]['params_hidden_size'],
        "feed_forward_size": df.iloc[0]['params_feed_forward_size'],
        "num_heads": df.iloc[0]['params_num_heads'],
        "num_blocks": df.iloc[0]['params_num_blocks'],
        "dropout_rate": df.iloc[0]['params_dropout_rate'],
        "max_seq_length": 8,
        "vocab_size": 4
    }

    return train_params, model_params
