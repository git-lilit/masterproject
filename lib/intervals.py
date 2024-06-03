def calculate_picp(prediction_intervals, true_values):
    """
    Calculate the Prediction Interval Coverage Probability (PICP).
    
    Parameters:
    prediction_intervals (list of tuples): A list of tuples, each containing the lower and upper bounds of the prediction intervals.
    true_values (list or np.array): A list of true values.

    Returns:
    float: The PICP value.
    """
    n = len(true_values)
    if n == 0:
        return 0.0

    coverage_count = 0
    for (L_i, U_i), y_i in zip(prediction_intervals, true_values):
        if L_i <= y_i <= U_i:
            coverage_count += 1
    
    picp = coverage_count / n
    return picp

def calculate_piw(prediction_intervals):
    """
    Calculate the average Prediction Interval Width (PIW).
    
    Parameters:
    prediction_intervals (list of tuples): A list of tuples, each containing the lower and upper bounds of the prediction intervals.

    Returns:
    float: The average PIW value.
    """
    n = len(prediction_intervals)
    if n == 0:
        return 0.0

    total_width = sum(U_i - L_i for L_i, U_i in prediction_intervals)
    piw = total_width / n
    return piw

def calculate_cwc(prediction_intervals, true_values, picp_desired, alpha=1.0):
    """
    Calculate the Coverage Width-Based Criterion (CWC).
    
    Parameters:
    prediction_intervals (list of tuples): A list of tuples, each containing the lower and upper bounds of the prediction intervals.
    true_values (list or np.array): A list of true values.
    picp_desired (float): The desired PICP value.
    alpha (float): The penalty factor for not meeting the desired PICP.

    Returns:
    float: The CWC value.
    """
    picp = calculate_picp(prediction_intervals, true_values)
    piw = calculate_piw(prediction_intervals)
    
    penalty = max(0, picp_desired - picp)
    cwc = piw * (1 + alpha * penalty)
    return cwc

# Example usage
prediction_intervals = [(2, 5), (3, 6), (4, 8), (1, 4), (2, 6)]
true_values = [4, 5, 7, 2, 6]
picp_desired = 0.95
alpha = 1.0

picp = calculate_picp(prediction_intervals, true_values)
piw = calculate_piw(prediction_intervals)
cwc = calculate_cwc(prediction_intervals, true_values, picp_desired, alpha)

print(f"PICP: {picp}")
print(f"PIW: {piw}")
print(f"CWC: {cwc}")
