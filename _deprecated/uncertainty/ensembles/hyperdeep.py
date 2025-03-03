import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.model_loading import get_top_models
from lib.dataset import get_saved_dataset
from lib.training import initialize_device
from ensembles.ensemble_base import EnsembleBase


class HyperdeepEnsemble(EnsembleBase):
    def __init__(self, num_models, results_filename, models_base_dir, device):
        super().__init__(num_models, device)
        self.models = get_top_models(
            num_models, results_filename, models_base_dir, device
        )


if __name__ == "__main__":
    num_models = 5
    test_dataset = get_saved_dataset(type="test_random")
    device = initialize_device()
    model_type = "homo"

    ensemble = HyperdeepEnsemble(
        num_models,
        results_filename=f"study_results/study_results_ff_{model_type}.csv",
        models_base_dir=f"saved_models/hyperopt_{model_type}",
        device=device
    )

    predictions = ensemble.predict(test_dataset)
    ensemble.save_predictions(predictions, filename=f"results_{model_type}/hyperdeep_pred.pkl")
