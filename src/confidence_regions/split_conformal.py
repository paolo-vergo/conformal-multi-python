import numpy as np

from src.helpers.helpers import calculate_scores, compute_prediction_bands, log, compute_mad_adjustments, generate_tau, \
    split_data_indices
from src.helpers.s_regression import compute_s_regression

"""
    Split conformal prediction intervals with multivariate response.

    Parameters:
    - x: np.ndarray, feature variables (n x p).
    - y: np.ndarray, multivariate responses (n x q).
    - x0: np.ndarray, new points to evaluate (n0 x p).
    - train_fun: function, model training function.
    - predict_fun: function, model prediction function.
    - alpha: float, miscoverage level (default: 0.1).
    - split: np.ndarray or None, indices defining the training split.
    - seed: int or None, seed for reproducibility.
    - randomized: bool, whether to use randomized approach.
    - seed_tau: int or None, seed for randomized tau generation.
    - verbose: bool or str, verbosity level.
    - training_size: float, proportion of training data.
    - score: str, non-conformity measure ("max", "l2", "mahalanobis").
    - s_type: str, modulation function type ("identity", "st-dev", "alpha-max").
    - mad_train_fun: function or None, model training on residuals.
    - mad_predict_fun: function or None, prediction on residuals.

    Returns:
    - dict with keys: "x0", "pred", "k_s", "s_type", "s", "alpha", "randomized",
      "tau", "average_width", "lo", "up".
    """


def conformal_multidim_split(
        x, y, x0, train_fun, predict_fun, alpha=0.1, split=None, seed=None,
        randomized=False, seed_tau=None, verbose=False, training_size=0.5,
        score="l2", s_type="st-dev", mad_train_fun=None, mad_predict_fun=None
):
    n, p = x.shape  # Number of data points and features
    _, q = y.shape  # Number of response variables
    n0 = x0.shape[0]  # Number of new points to evaluate

    # Determine if MAD (modulation on residuals) is being used
    use_mad = mad_train_fun is not None and mad_predict_fun is not None
    if use_mad:
        score = "identity"

    # Step 1: Split data into training and calibration sets
    if split is None:
        training_indices, calibration_indices = split_data_indices(n, training_size, seed)
    else:
        training_indices = split
        calibration_indices = np.setdiff1d(np.arange(n), training_indices)

    # Step 2: Generate tau for randomized conformal prediction
    tau = generate_tau(randomized, seed_tau)

    # Step 3: Train the model and generate predictions
    log("Training the model on the training set...", verbose)
    model = train_fun(x[training_indices], y[training_indices])
    predictions_full = predict_fun(model, x)
    predictions_x0 = predict_fun(model, x0)

    # Step 4: Compute residuals and scaling factors
    log("Calculating residuals and scaling factors...", verbose)
    residuals = y - predictions_full
    scaling_factors = compute_s_regression(residuals[training_indices], s_type, alpha, tau)

    # Rescale residuals for the calibration set
    calibration_residuals = residuals[calibration_indices] / scaling_factors

    # If MAD is used, adjust residuals and predictions accordingly
    mad_adjustment_x0 = None
    if use_mad:
        log("Adjusting residuals using MAD functions...", verbose)
        adjusted_residuals, mad_adjustment_x0 = compute_mad_adjustments(
            x[training_indices], residuals[training_indices],
            x[calibration_indices], x0, mad_train_fun, mad_predict_fun
        )
        calibration_residuals /= adjusted_residuals

    # Step 5: Calculate conformity scores and determine prediction intervals
    log("Computing conformity scores and prediction intervals...", verbose)
    conformity_scores = calculate_scores(calibration_residuals, score)
    l = len(calibration_indices)  # Size of the calibration set
    k_s = np.sort(conformity_scores)[int(np.ceil(l + tau - (l + 1) * alpha)) - 1]

    average_width = np.mean(2 * k_s * scaling_factors)

    # Calculate prediction bands
    prediction_bands = compute_prediction_bands(
        k_s, scaling_factors, n0, use_mad=use_mad, mad_adjustments=mad_adjustment_x0
    )
    lower_bound = predictions_x0 - prediction_bands
    upper_bound = predictions_x0 + prediction_bands

    # Return results as a dictionary
    return {
        "x0": x0,
        "pred": predictions_x0,
        "k_s": k_s,
        "s_type": s_type,
        "s": scaling_factors,
        "alpha": alpha,
        "randomized": randomized,
        "tau": tau,
        "average_width": average_width,
        "lo": lower_bound,
        "up": upper_bound
    }
