import os
import sys
import numpy as np
from sklearn.model_selection import KFold

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.dataset import get_test_dataset, get_train_data, NumpyDataset
from lib.training import train_model
from lib.model_loading import get_best_params
from ensembles.ensemble_base import EnsembleBase
from torch.utils.data import DataLoader


class CVEnsemble(EnsembleBase):
    def __init__(self, num_models, results_filename):
        super().__init__(num_models)
        self.train_params, self.model_params = get_best_params(results_filename)
        self.models = self.get_models()

    def get_models(self):
        models = []
        kf = KFold(n_splits=self.num_models, shuffle=True, random_state=42)
        X, y = get_train_data(use_full_data=False)

        for i, (train_index, val_index) in enumerate(kf.split(X)):
            print(f"Training model {i + 1}/{self.num_models}")
            self.train_params["model_id"] = f"cv_ensemble_model_{i}"

            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            train_dataset = NumpyDataset(X_train, y_train)
            val_dataset = NumpyDataset(X_val, y_val)

            train_loader = DataLoader(
                train_dataset, batch_size=self.train_params["batch_size"], shuffle=True
            )
            val_loader = DataLoader(
                val_dataset, batch_size=self.train_params["batch_size"], shuffle=True
            )

            # Initialize and train the model
            model, loss = train_model(
                self.train_params,
                self.model_params,
                save_model=True,
                folder_name="saved_models/ensembles/cv",
                loaders=[train_loader, val_loader]
            )

            # Store the trained model
            models.append(model)

        return models


if __name__ == "__main__":
    num_models = 5
    test_dataset = get_test_dataset()

    ensemble = CVEnsemble(num_models, results_filename="study_results2.csv")

    predictions = ensemble.predict(test_dataset)
    ensemble.save_predictions(predictions, filename="cv_pred.pkl")
