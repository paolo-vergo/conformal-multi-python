import numpy as np
from scipy.stats import multivariate_normal


def generate_random_data(n=50, n0=3, p=4, mu=None, variances=None, response_dim=2):
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
