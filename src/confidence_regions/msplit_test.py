import numpy as np
from multiprocessing import Pool

from src.confidence_regions.split_conformal import conformal_multidim_split

def conformal_multidim_msplit_test(
    x, y, x0, train_fun, predict_fun, alpha=0.1, split=None, seed=None,
    randomized=False, seed_beta=None, verbose=False, training_size=None,
    score="max", s_type="st-dev", B=100, lambda_=0, tau=None):
    """
    Compute prediction intervals using Multi-Split conformal inference with multivariate response.

    Parameters:
        x (numpy.ndarray): Feature matrix (n x p).
        y (numpy.ndarray): Multivariate response matrix (n x q).
        x0 (numpy.ndarray): New points for evaluation (n0 x p).
        train_fun (callable): Function to train the model.
        predict_fun (callable): Function to predict responses for new features.
        alpha (float): Miscoverage level (default=0.1).
        split (list or None): Indices for data-split (default=None, random split).
        seed (int or None): Seed for random split (default=None).
        randomized (bool): Whether to use a randomized approach (default=False).
        seed_beta (int or None): Seed for randomized version (default=None).
        verbose (bool): Verbosity (default=False).
        training_size (list or None): Split proportions for training/calibration (default=0.5 for all splits).
        score (str): Score type (default="max").
        s_type (str): Modulation function type (default="st-dev").
        B (int): Number of repetitions (default=100).
        lambda_ (float): Smoothing parameter (default=0).
        tau (float or None): Smoothing parameter (default=1 - (B+1)/(2*B)).

    Returns:
        dict: Prediction intervals containing 'lo', 'up', and 'x0'.
    """
    if training_size is None or len(training_size) != B:
        training_size = [0.5] * B

    tau = tau or (1 - (B + 1) / (2 * B))

    if seed is not None:
        np.random.seed(seed)

    n0, q = x0.shape[0], y.shape[1]
    full = n0 * q

    # Generate predictions for all splits
    lo_up = np.zeros((B, 2 * full))
    for bbb in range(B):
        if verbose:
            print(f"Processing split {bbb + 1}/{B}...")

        # Train and predict
        out = conformal_multidim_split(
            x, y, x0, train_fun, predict_fun,
            alpha * (1 - tau) + (alpha * lambda_) / B,
            split, seed + bbb if seed else None, randomized,
            seed_beta, verbose, training_size[bbb], score, s_type
        )

        # Flatten and store results
        lo_up[bbb, :full] = out['lo'].ravel()
        lo_up[bbb, full:] = out['up'].ravel()

    # Combine results to compute intervals
    Y = np.vstack([lo_up[:, :full], lo_up[:, full:]])
    tr = tau * B + 0.001

    # Compute final intervals
    lo, up = _compute_intervals(Y, B, tr, n0, q)

    return {'lo': lo, 'up': up, 'x0': x0}


def _compute_intervals(Y, B, tr, n0, q):
    """
    Helper function to compute final prediction intervals.

    Parameters:
        Y (numpy.ndarray): Combined lower and upper bounds from all splits.
        B (int): Number of splits.
        tr (float): Truncation threshold.
        n0 (int): Number of evaluation points.
        q (int): Number of response dimensions.

    Returns:
        tuple: Lower and upper bounds reshaped into (n0, q).
    """
    full = n0 * q
    intervals = np.array([interval_build(Y[:, kk], B, tr) for kk in range(full)])
    lo = intervals[:, 0].reshape(n0, q)
    up = intervals[:, 1].reshape(n0, q)
    return lo, up


def interval_build(yyy, B, tr):
    """
    Builds the interval for each x0 by combining multiple splits.

    Parameters:
        yyy (numpy.ndarray): Array of B lower and B upper bounds.
        B (int): Number of repetitions.
        tr (float): Truncation threshold.

    Returns:
        tuple: Lower and upper bounds of the interval.
    """
    h = np.concatenate([np.ones(B), np.zeros(B)])
    sorted_indices = np.lexsort((-h, yyy))  # Sort by yyy first, then by h
    ys, hs = yyy[sorted_indices], h[sorted_indices]

    count = 0
    lo, up = 0, 0

    for j in range(2 * B):
        if hs[j] == 1:
            count += 1
            if tr < count <= tr + 1:
                lo = ys[j]
        else:
            if tr < count <= tr + 1:
                up = ys[j]
            count -= 1

    return lo, up
