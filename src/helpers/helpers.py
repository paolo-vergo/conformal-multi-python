from itertools import product

import numpy as np
from scipy.spatial.distance import mahalanobis

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