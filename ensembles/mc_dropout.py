import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.model_loading import get_top_models


if __name__ == "__main__":
    num_models = 1
    model = get_top_models(num_models)[0]
    model.train()

