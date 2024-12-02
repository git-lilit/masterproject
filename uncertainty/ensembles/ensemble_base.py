import os
import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader


class EnsembleBase:
    def __init__(self, num_models, device):
        self.num_models = num_models
        self.device = device

    def get_models(self):
        raise NotImplementedError("Subclasses should implement this method")

    def predict_one_model(self, model, test_loader):
        model_mus = []
        model_log_vars = []
        model.to(self.device)

        with torch.no_grad():  # Disable gradient calculation
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs = inputs.transpose(0, 1)
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    mu, log_var = outputs
                    model_mus.append(mu.cpu().numpy())
                    model_log_vars.append(log_var.cpu().numpy())
                else:
                    model_mus.append(outputs.cpu().numpy())

        model_mus = np.concatenate(model_mus)
        model_log_vars = np.concatenate(model_log_vars) if model_log_vars else None

        return model_mus, model_log_vars

    def predict(self, test_dataset, batch_size=256):
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        targets = test_dataset.labels

        all_predictions = []

        for idx, model in enumerate(self.models):
            print(f"Predicting with model {idx}")
            model_mus, model_log_vars = self.predict_one_model(model, test_loader)

            all_predictions.append((model_mus, model_log_vars))

        return all_predictions, targets

    def save_predictions(self, predictions, filename="ensemble_predictions.pkl"):
        file_path = f"predictions/{filename}"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as f:
            pickle.dump(predictions, f)
