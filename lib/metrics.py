import torch
import numpy as np
from lib.losses import gaussian_nll_loss
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt 


def compute_mean_nll(ensemble_predictions, true_values):
    """
    Compute the Negative Log Likelihood (NLL) for an ensemble of Gaussian predictions using PyTorch.

    Parameters:
    ensemble_predictions (list of tuples): Each tuple contains two np.arrays: (means, log_vars) for each model in the ensemble.
    true_values (torch.Tensor): True values.

    Returns:
    torch.Tensor: The averaged NLL value for the ensemble.
    """
    nlls = []
    for means, log_vars in ensemble_predictions:
        nll = gaussian_nll_loss(
            torch.tensor(means), torch.tensor(log_vars), true_values
        )
        nlls.append(nll)

    avg_nll = torch.mean(torch.tensor(nlls))
    return avg_nll


def compute_combined_parameters(ensemble_predictions):
    """
    Compute the combined parameters for an ensemble given predictions.

    Parameters:
    ensemble_predictions (list of tuples): Each tuple contains two np.arrays: (means, log_variances) for each model in the ensemble.

    Returns:
    tuple: Two np.arrays, one for combined means and one for combined standard deviations for the ensemble.
    """
    # Extract means and log variances from model predictions
    means_list = [pred[0] for pred in ensemble_predictions]
    log_variances_list = [pred[1] for pred in ensemble_predictions]

    # Convert to numpy arrays for easier manipulation
    means = np.array(means_list)
    log_variances = np.array(log_variances_list)

    # Convert log variances to variances
    variances = np.exp(log_variances)

    # Compute the mean of the means
    mean_of_means = np.mean(means, axis=0)

    # Compute the combined variance
    variance_component = np.mean(variances, axis=0)
    bias_component = np.mean((means - mean_of_means) ** 2, axis=0)
    combined_log_variances = variance_component + bias_component

    # Convert combined variances back to log variances
    combined_log_variances = np.log(combined_log_variances)

    return torch.tensor(mean_of_means), torch.tensor(combined_log_variances)


def calculate_picp_with_intervals(means, stds, true_values, confidence_level=0.95):
    """
    Calculate the Prediction Interval Coverage Probability (PICP) given mean and standard deviation predictions.

    Parameters:
    means (list): A list of mean predictions.
    stds (list): A list of standard deviation predictions.
    true_values (list): A list of true values.
    confidence_level (float): The confidence level for the prediction intervals (default is 0.95).

    Returns:
    float: The PICP value.
    """
    import scipy.stats as stats

    # Calculate the z-score for the given confidence level
    z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)

    n = len(true_values)
    if n == 0:
        return 0.0

    coverage_count = 0
    for mu, sigma, y in zip(means, stds, true_values):
        L_i = mu - z_score * sigma
        U_i = mu + z_score * sigma
        if L_i <= y <= U_i:
            coverage_count += 1

    picp = coverage_count / n
    return picp


def calculate_correlations(x, y):
    """
    Calculate the Pearson and Spearman correlations between two sets of values.

    Parameters:
    x (list or np.array): First set of values (e.g., uncertainties).
    y (list or np.array): Second set of values (e.g., errors).

    Returns:
    dict: A dictionary with Pearson and Spearman correlation coefficients and p-values.
    """
    # Calculate Pearson correlation
    pearson_corr, pearson_p_value = pearsonr(x, y)

    # Calculate Spearman rank correlation
    spearman_corr, spearman_p_value = spearmanr(x, y)

    # Return results as a dictionary
    return {
        "pearson_corr": pearson_corr,
        "pearson_p_value": pearson_p_value,
        "spearman_corr": spearman_corr,
        "spearman_p_value": spearman_p_value,
    }


def calculate_all_metrics(ensemble_predictions, true_values):
    mean_of_means, combined_log_variances = compute_combined_parameters(
        ensemble_predictions
    )

    stds = np.sqrt(np.exp(combined_log_variances))
    errors = np.abs(true_values.reshape(-1) - mean_of_means)

    results = {
        "mean_nll": compute_mean_nll(ensemble_predictions, true_values),
        "combined_nll": gaussian_nll_loss(
            mean_of_means, combined_log_variances, true_values
        ),
        "picp": calculate_picp_with_intervals(mean_of_means, stds, true_values),
        "correlations": calculate_correlations(combined_log_variances, errors),
    }

    return results


def calibration_plot(predictions, true_values, num_bins=10):
    """
    Generate a calibration plot.

    Parameters:
    predictions (np.array): Predicted values (e.g., means).
    true_values (np.array): True values.
    num_bins (int): Number of bins to use for the plot.
    """
    # Ensure predictions and true_values are numpy arrays
    predictions = np.array(predictions)
    true_values = np.array(true_values)

    # Sort the predictions and true values based on predictions
    sorted_indices = np.argsort(predictions)
    sorted_predictions = predictions[sorted_indices]
    sorted_true_values = true_values[sorted_indices]

    # Create bins and calculate mean predictions and true values for each bin
    bins = np.array_split(np.arange(len(sorted_predictions)), num_bins)
    bin_means = [np.mean(sorted_predictions[bin]) for bin in bins]
    bin_true_means = [np.mean(sorted_true_values[bin]) for bin in bins]

    # Plot the calibration plot
    plt.plot(bin_means, bin_true_means, marker="o", linestyle="-", label="Empirical")
    plt.plot(
        [min(bin_means), max(bin_means)],
        [min(bin_means), max(bin_means)],
        linestyle="--",
        label="Ideal",
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Calibration Plot")
    plt.legend()
    plt.grid(True)
    plt.show()
