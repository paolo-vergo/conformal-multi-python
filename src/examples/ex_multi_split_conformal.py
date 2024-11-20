import numpy as np
from scipy.stats import multivariate_normal
from src.confidence_regions.msplit_test import conformal_multidim_msplit_test
from src.confidence_regions.multi_split_conformal import conformal_multidim_msplit
from src.helpers.prediction_models import *
from src.plots.plot_msplit_conformal import plot_multidim_msplit
from src.plots.plot_split_conformal import plot_multidim_split_custom


# Function to set up data for the experiment
def generate_data(n, n0, p, mu, variances, response_dim):
    """
    Generate training and test datasets with specified parameters.
    Now handles a response variable of arbitrary dimension.

    Parameters:
        n (int): Number of training samples.
        n0 (int): Number of test samples.
        p (int): Dimensionality of features.
        mu (array): Mean vector for multivariate normal distribution.
        variances (array): Variances for each feature (diagonal of the covariance matrix).
        response_dim (int): The dimensionality of the response variable y.

    Returns:
        tuple: (x, y, x0) where x is the training data, y is the response variable, and x0 is the test data.
    """
    covariance_matrix = np.diag(variances)
    # Generate feature samples from a multivariate normal distribution
    x = multivariate_normal.rvs(mean=mu, cov=covariance_matrix, size=n)

    # Non-linear response generation (response dimension can differ from input)
    y = np.array([
        2 * u[0] + np.sin(np.pi * u[1]) + np.random.normal(0, 0.1, response_dim)
        for u in x
    ])

    # Generate test points from the same distribution
    x0 = multivariate_normal.rvs(mean=mu, cov=covariance_matrix, size=n0)

    return x, y, x0

# Function to select a prediction model
def get_prediction_model(model_name="xgboost"):
    """
    Select the appropriate prediction model based on the given name.

    Parameters:
        model_name (str): Name of the model to select. Default is "xgboost".

    Returns:
        dict: A dictionary containing the training and prediction functions for the chosen model.
    """
    if model_name == "xgboost":
        return multivariate_xgboost()
    elif model_name == "linear":
        return multivariate_linear_regressor()
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")

# Main experiment setup
def run_conformal_msplit_experiment(n=50, n0=3, p=2, mu=None, variances=None, model_name="xgboost", B=50, alpha=0.1, response_dim=1):
    """
    Run a conformal multi-dimensional multi-split experiment.

    Parameters:
        n (int): Number of training samples.
        n0 (int): Number of test samples.
        p (int): Dimensionality of features.
        mu (array): Mean vector for multivariate normal distribution.
        variances (array): Variances for each feature (diagonal of the covariance matrix).
        model_name (str): Name of the prediction model to use (default is "xgboost").
        B (int): Number of replications for the multi-split conformal method.
        alpha (float): Significance level for confidence intervals.
        response_dim (int): Dimensionality of the response variable y (can be greater than 1).

    Returns:
        dict: The result from the conformal multi-split function containing lower and upper bounds.
    """
    # Default mean and variances if not provided
    if mu is None:
        mu = np.linspace(0, 1, p)
    if variances is None:
        variances = np.array([5, 20])

    # Generate data
    x, y, x0 = generate_data(n, n0, p, mu, variances, response_dim)

    # Select the prediction model
    model = get_prediction_model(model_name)

    # Run the conformal multi-split procedure
    result = conformal_multidim_msplit(
        x=x, y=y, x0=x0,
        train_fun=model["train_fun"], predict_fun=model["predict_fun"],
        alpha=alpha, split=None, seed=False, randomized=False,
        seed_beta=False, verbose=False, training_size=None,
        s_type="alpha-max", B=B, lambda_=0, score="l2"
    )

    return result

# Run the experiment with response dimension greater than 1
response_dim = 3  # Example: 3-dimensional response
final_multi = run_conformal_msplit_experiment(response_dim=response_dim)

# Print results
print("Lower bounds:\n", final_multi['lo'])
print("Upper bounds:\n", final_multi['up'])

# Plot the results
plot_multidim_split_custom(final_multi)
