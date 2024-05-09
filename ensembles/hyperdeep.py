import sys
import os
import numpy as np
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.model_loading import get_top_models
from lib.dataset import get_test_data
from lib.prediction import predict_with_ensemble

if __name__ == "__main__":
    num_models = 5
    models = get_top_models(num_models)

    test_dataset = get_test_data()
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=True)

    ensemble_predictions = predict_with_ensemble(models, test_loader)

    np.save("predictions/hyperdeep.npy", ensemble_predictions)
