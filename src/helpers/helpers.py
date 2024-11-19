from itertools import product
from scipy.stats import uniform
from scipy.spatial.distance import mahalanobis

import numpy as np

def check_input_validation(num_grid_pts_dim: int, grid_factor: float) -> None:
    """Helper function to check the validity of inputs."""
    if not (1 < num_grid_pts_dim < 1000) or not isinstance(num_grid_pts_dim, int):
        raise ValueError("num_grid_pts_dim must be an integer between 1 and 1000")
    if not (grid_factor > 0):
        raise ValueError("grid_factor must be positive")


def calculate_scores(residuals: np.ndarray, score: str) -> np.ndarray:
    """Helper function to calculate the chosen scoring method."""
    if score == 'l2':
        return np.sum(residuals ** 2, axis=1)
    elif score == 'mahalanobis':
        inv_cov = np.linalg.inv(np.cov(residuals.T))
        return np.array([mahalanobis(r, np.mean(residuals, axis=0), inv_cov) for r in residuals])
    elif score == 'max':
        return np.max(np.abs(residuals), axis=1)
    elif score == 'scaled.max':
        scaled_residuals = residuals / np.var(residuals, axis=0)
        return np.max(np.abs(scaled_residuals), axis=1)
    else:
        raise ValueError(f"Unsupported score method: {score}")


def compute_residuals(
    xx: np.ndarray, yy: np.ndarray, train_fun, predict_fun, iteration: int, model_out
) -> np.ndarray:
    """
    Train the model and compute residuals for given inputs.
    """
    model_out = train_fun(xx, yy) if iteration == 0 else train_fun(xx, yy, model_out)
    predictions = predict_fun(model_out, xx)
    return yy - predictions


def apply_mad_scaling(
    xx: np.ndarray, residuals: np.ndarray, mad_train_fun, mad_predict_fun, iteration: int
) -> np.ndarray:
    """
    Apply MAD scaling to residuals.
    """

    if mad_train_fun and mad_predict_fun:
        out_mad = mad_train_fun(xx, np.abs(residuals)) if iteration == 0 else mad_train_fun(xx, np.abs(residuals))
        residuals /= mad_predict_fun(out_mad, xx)

    return residuals


def generate_test_points_grid(y: np.ndarray, grid_factor: float, num_grid_pts_dim: int) -> np.ndarray:
    """
    Generate a grid of possible target values based on the range of y.
    """
    ymax = np.max(np.abs(y), axis=0)
    y_marg = [np.linspace(-grid_factor * val, grid_factor * val, num_grid_pts_dim) for val in ymax]
    return np.array(list(product(*y_marg)))

def log(message, verbose, prefix=""):
    """Log messages based on verbosity level."""
    if verbose:
        if isinstance(verbose, str):
            prefix = f"{verbose}: "
        print(f"{prefix}{message}")


def split_data_indices(n, training_size, seed=None):
    """Split data indices into training and calibration sets."""
    if seed is not None:
        np.random.seed(seed)
    training_size_n = int(np.ceil(n * training_size))
    training_indices = np.random.choice(n, training_size_n, replace=False)
    calibration_indices = np.setdiff1d(np.arange(n), training_indices)
    return training_indices, calibration_indices


def generate_tau(randomized, seed_tau=None):
    """Generate tau for randomized conformal prediction."""
    if not randomized:
        return 1
    if seed_tau is not None:
        np.random.seed(seed_tau)
    return uniform.rvs(loc=0, scale=1)


def compute_mad_adjustments(
    x_train, residuals_train, x_calibration, x0, mad_train_fun, mad_predict_fun
):
    """Compute MAD adjustments for residuals and predictions."""
    mad_model = mad_train_fun(x_train, residuals_train)
    adjusted_residuals = mad_predict_fun(mad_model, x_calibration)
    mad_adjustment_x0 = mad_predict_fun(mad_model, x0)
    return adjusted_residuals, mad_adjustment_x0


def compute_prediction_bands(k_s, scaling_factors, x0_size, use_mad=False, mad_adjustments=None):
    """Compute prediction bands for new data points."""
    if use_mad:
        return mad_adjustments * k_s
    return k_s * np.tile(scaling_factors, (x0_size, 1))
