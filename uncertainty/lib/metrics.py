import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from torch.nn import GaussianNLLLoss, MSELoss


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
    loss_fn = GaussianNLLLoss()

    for means, log_vars in ensemble_predictions:
        variances = torch.exp(torch.tensor(log_vars))

        nll = loss_fn(torch.tensor(means), true_values, variances)
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
    means = np.array([pred[0] for pred in ensemble_predictions])
    mean_of_means = np.mean(means, axis=0)

    bias_component = np.mean((means - mean_of_means) ** 2, axis=0)
    combined_variances = bias_component

    variance_of_the_first_model = ensemble_predictions[0][1]
    if variance_of_the_first_model is not None:
        log_variances = np.array([pred[1] for pred in ensemble_predictions])
        variances = np.exp(log_variances)
        variance_component = np.mean(variances, axis=0)
        combined_variances += variance_component

    combined_log_variances = np.log(combined_variances)

    return torch.tensor(mean_of_means).reshape(-1), torch.tensor(
        combined_log_variances
    ).reshape(-1)


def calculate_picp_and_piw(means, stds, true_values, confidence_level=0.95):
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
    piw_values = []

    for mu, sigma, y in zip(means, stds, true_values):
        L_i = mu - z_score * sigma
        U_i = mu + z_score * sigma
        piw_values.append(U_i - L_i)
        if L_i <= y <= U_i:
            coverage_count += 1

    picp = coverage_count / n
    piw = np.mean(piw_values)

    return picp, piw


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
        "spearman_corr": spearman_corr,
    }


def calculate_all_metrics(ensemble_predictions, true_values):
    mean_of_means, combined_log_variances = compute_combined_parameters(
        ensemble_predictions
    )

    variances = np.exp(combined_log_variances)
    stds = np.sqrt(variances)
    errors = np.abs(true_values.reshape(-1) - mean_of_means)
    loss_fn = GaussianNLLLoss()

    results = {
        "combined_nll": loss_fn(mean_of_means, true_values, variances),
        "intervals": calculate_picp_and_piw(mean_of_means, stds, true_values),
        "correlations_uncertainty": calculate_correlations(combined_log_variances, errors),
        "correlations_predictions": calculate_correlations(mean_of_means, true_values.reshape(-1)),
    }

    variance_of_the_first_model = ensemble_predictions[0][1]
    if variance_of_the_first_model is not None:
        results["mean_nll"] = compute_mean_nll(ensemble_predictions, true_values)

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


def visualize_intervals(true_values, means, variances, num_points=None):
    """
    Visualizes prediction intervals.

    Parameters:
    true_values (np.array): Array of true values.
    means (np.array): Array of mean predictions.
    variances (np.array): Array of log variances of the predictions.
    num_points (int): Number of points to plot. If None, plot all points.

    The plot will show:
    - Mean predictions with a solid line.
    - 1 standard deviation intervals with a semi-transparent shading.
    - 2 standard deviation intervals with a lighter semi-transparent shading.
    - Perfect line (y=x) for reference.
    """
    # Convert inputs to numpy arrays if they are not
    true_values = np.array(true_values)
    means = np.array(means)
    
    if num_points is not None and num_points < len(true_values):
        indices = np.random.choice(len(true_values), num_points, replace=False)
        true_values = true_values[indices]
        means = means[indices]
        variances = variances[indices]
    
    std_devs = np.sqrt(variances)
    
    # Calculate the intervals
    one_std_up = means + std_devs
    one_std_down = means - std_devs
    two_std_up = means + 2 * std_devs
    two_std_down = means - 2 * std_devs

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot the perfect line
    plt.plot(true_values, true_values, 'r--', label='Perfect Line (y=x)')

    # Plot the mean predictions and intervals
    for i in range(len(true_values)):
        plt.plot([true_values[i], true_values[i]], [two_std_down[i], two_std_up[i]], color='blue', alpha=0.1)  # 2 Std Dev
        plt.plot([true_values[i], true_values[i]], [one_std_down[i], one_std_up[i]], color='blue', alpha=0.2)  # 1 Std Dev
        plt.plot(true_values[i], means[i], 'o', color='blue')  # Mean Prediction

    # Add labels and legend
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Prediction Intervals')
    plt.legend()

    # Show the plot
    plt.show()
