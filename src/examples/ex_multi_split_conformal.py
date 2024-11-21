from src.confidence_regions.multi_split_conformal import conformal_multidim_msplit
from src.helpers.generate_random_data import generate_random_data
from src.plots.plot_split_conformal import plot_multi_split
from src.prediction_models.models import *
from src.prediction_models.models import get_prediction_model


# Main experiment setup
def run_conformal_msplit_experiment(n=50, n0=3, p=2, mu=None, variances=None, model_name="xgboost", B=50, alpha=0.1,
                                    response_dim=1):
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
    x, y, x0 = generate_random_data(n, n0, p, mu, variances, response_dim)

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
plot_multi_split(final_multi)
