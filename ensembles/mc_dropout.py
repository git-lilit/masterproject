import sys
import os
import numpy as np
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.model_loading import get_top_models
from lib.dataset import get_test_dataset
from ensembles.ensemble_base import EnsembleBase


class MCDropoutEnsemble(EnsembleBase):
    def __init__(self, num_samples, results_filename, models_base_dir):
        self.num_models = 1
        super().__init__(num_models=self.num_models)
        self.num_samples = num_samples
        self.models = get_top_models(
            num_models=self.num_models,
            filename=results_filename,
            base_dir=models_base_dir,
        )

    def predict(self, test_dataset, batch_size=256):
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        targets = test_dataset.labels
        model = self.models[0]
        model.train()

        all_predictions = []

        for idx in range(self.num_samples):
            print(f"Sampling {idx}")
            model_mus, model_sigmas = self.predict_one_model(model, test_loader)

            all_predictions.append((model_mus, model_sigmas))

        return all_predictions, targets


if __name__ == "__main__":
    test_dataset = get_test_dataset()

    ensemble = MCDropoutEnsemble(
        num_samples=5,
        results_filename="study_results3_homo.csv",
        models_base_dir="saved_models/hyperopt3_homo",
    )

    predictions = ensemble.predict(test_dataset, batch_size=512)
    ensemble.save_predictions(predictions, filename="results3_homo/dropout_pred.pkl")
