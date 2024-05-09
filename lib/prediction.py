import numpy as np

def predict_with_ensemble(models, test_loader):
    all_predictions = []
    
    for idx, model in enumerate(models):
        print(f"Predicting with model N{idx}")
        model_predictions = []
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            outputs = model(inputs)
            model_predictions.append(outputs.detach())
        all_predictions.append(model_predictions)
    
    return all_predictions
