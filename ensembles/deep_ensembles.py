import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.dataset import get_test_data
from lib.training import train_model
from lib.prediction import predict_with_ensemble
from lib.model_loading import get_best_params


def get_models(num_models, train_params, model_params):
    models = []

    for i in range(num_models):
        print(f"Training model {i+1}/{num_models}")
        train_params["model_id"] = f"deep_ensemble_model_{i}"

        model, loss = train_model(
            train_params,
            model_params,
            save_model=True,
            folder_name="saved_models/ensembles/deep",
        )
        models.append(model)

    return models


if __name__ == "__main__":
    num_models = 5
    train_params, model_params = get_best_params()

    models = get_models(num_models, train_params, model_params)

    X_test, y_test = get_test_data()
    ensemble_predictions = predict_with_ensemble(models, X_test)

    np.save('ensemble_predictions.npy',  np.array(ensemble_predictions))
