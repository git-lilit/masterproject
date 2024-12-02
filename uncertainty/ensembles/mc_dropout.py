import sys
import os
import numpy as np
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.model_loading import get_top_models
from lib.dataset import get_saved_dataset
from lib.training import initialize_device
from ensembles.ensemble_base import EnsembleBase


class MCDropoutEnsemble(EnsembleBase):
    def __init__(self, num_samples, results_filename, models_base_dir, device):
        self.num_models = 1
        super().__init__(num_models=self.num_models, device=device)
        self.num_samples = num_samples
        self.models = get_top_models(
            num_models=self.num_models,
            filename=results_filename,
            base_dir=models_base_dir,
            device=device
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
    model_type = "homo"
    test_dataset = get_saved_dataset(type="test_random")
    device = initialize_device()

    ensemble = MCDropoutEnsemble(
        num_samples=5,
        results_filename=f"study_results/study_results_ff_{model_type}.csv",
        models_base_dir=f"saved_models/hyperopt_{model_type}",
        device=device
    )

    predictions = ensemble.predict(test_dataset, batch_size=512)
    ensemble.save_predictions(predictions, filename=f"results_{model_type}/dropout_pred.pkl")
