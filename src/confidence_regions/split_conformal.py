import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.stats import uniform

from src.helpers.s_regression import compute_s_regression


def conformal_multidim_split(x, y, x0, train_fun, predict_fun, alpha=0.1,
                             split=None, seed=None, randomized=False, seed_tau=None,
                             verbose=False, training_size=0.5, score="l2", s_type="st-dev",
                             mad_train_fun=None, mad_predict_fun=None):
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

    n, p = x.shape
    q = y.shape[1]
    n0 = x0.shape[0]
    flag = False  # Default: no MAD functions

    # Check for MAD functions
    if mad_train_fun is not None and mad_predict_fun is not None:
        score = "identity"
        flag = True

    txt = ""
    if verbose and isinstance(verbose, str):
        txt = verbose
        verbose = True

    # Define training and calibration splits
    if split is None:
        m = int(np.ceil(n * training_size)) - 1 if np.ceil(n * training_size) == n else int(np.ceil(n * training_size))
        l = n - m

        if seed is not None:
            np.random.seed(seed)

        training = np.random.choice(n, m, replace=False)
    else:
        training = split

    calibration = np.setdiff1d(np.arange(n), training)

    # Generate tau
    tau = 1
    if randomized:
        if seed_tau is not None:
            np.random.seed(seed_tau)
        tau = uniform.rvs(loc=0, scale=1)

    # Train and compute residuals
    if verbose:
        print(f"{txt}Computing models on first part...")

    model = train_fun(x[training], y[training])
    fit = predict_fun(model, x)
    pred = predict_fun(model, x0)

    if verbose:
        print(f"{txt}Computing residuals and quantiles on second part...")

    residuals = y - fit
    s = compute_s_regression(residuals[training], s_type, alpha, tau)
    resc = residuals[calibration] / s

    if flag:
        mad_model = mad_train_fun(x[training], residuals[training])
        resc = resc / mad_predict_fun(mad_model, x[calibration])
        mad_x0 = mad_predict_fun(mad_model, x0)

    # Compute scores
    if score == "max":
        rho = np.max(np.abs(resc), axis=1)
    elif score == "l2":
        rho = np.sum(resc**2, axis=1)
    elif score == "mahalanobis":
        rho = np.array([mahalanobis(row, np.mean(resc, axis=0), np.cov(resc, rowvar=False)) for row in resc])
    else:
        raise ValueError(f"Unknown score: {score}")

    k_s = np.sort(rho)[int(np.ceil(l + tau - (l + 1) * alpha)) - 1]
    avg_width = np.mean(2 * k_s * s)

    if flag:
        band = mad_x0 * k_s
    else:
        band = k_s * np.tile(s, (n0, 1))

    lo = pred - band
    up = pred + band

    return {
        "x0": x0, "pred": pred, "k_s": k_s, "s_type": s_type, "s": s, "alpha": alpha,
        "randomized": randomized, "tau": tau, "average_width": avg_width, "lo": lo, "up": up
    }