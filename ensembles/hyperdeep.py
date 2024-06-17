import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.model_loading import get_top_models
from lib.dataset import get_test_dataset
from ensembles.ensemble_base import EnsembleBase


class HyperdeepEnsemble(EnsembleBase):
    def __init__(self, num_models, results_filename, models_base_dir):
        super().__init__(num_models)
        self.models = get_top_models(
            num_models, results_filename, models_base_dir
        )


if __name__ == "__main__":
    num_models = 5
    test_dataset = get_test_dataset()

    ensemble = HyperdeepEnsemble(
        num_models,
        results_filename="study_results3_homo.csv",
        models_base_dir="saved_models/hyperopt3_homo",
    )

    predictions = ensemble.predict(test_dataset)
    ensemble.save_predictions(predictions, filename="results3_homo/hyperdeep_pred.pkl")
