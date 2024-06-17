import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.dataset import get_test_dataset
from lib.training import train_model
from lib.model_loading import get_best_params
from ensembles.ensemble_base import EnsembleBase


class DeepEnsemble(EnsembleBase):
    def __init__(self, num_models, results_filename):
        super().__init__(num_models)
        self.train_params, self.model_params = get_best_params(results_filename)
        self.models = self.get_models()

    def get_models(self):
        models = []
        for i in range(self.num_models):
            print(f"Training model {i + 1}/{self.num_models}")
            self.train_params["model_id"] = f"deep_ensemble_model_{i}"

            model, loss = train_model(
                self.train_params,
                self.model_params,
                save_model=True,
                folder_name="saved_models/ensembles/deep3_homo",
            )
            models.append(model)

        return models


if __name__ == "__main__":
    num_models = 5
    test_dataset = get_test_dataset()
    ensemble = DeepEnsemble(num_models, results_filename="study_results3_homo.csv")

    predictions = ensemble.predict(test_dataset)
    ensemble.save_predictions(predictions, filename="results3_homo/deep_pred.pkl")
