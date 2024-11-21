import numpy as np

from src.prediction_regions.split_conformal import conformal_multidim_split

"""
    Compute multidimensional conformal prediction intervals using the MultiSplit algorithm.

    Parameters:
    - x: Training input data.
    - y: Training target data.
    - x0: Test data to compute prediction intervals for.
    - train_fun: Function to train a model on the training data.
    - predict_fun: Function to make predictions with the trained model.
    - alpha: Significance level for the prediction intervals.
    - split: Splitting strategy for the training data (optional).
    - seed: Random seed for reproducibility (optional).
    - randomized: Whether to randomize the split.
    - seed_beta: Seed for randomization in beta (optional).
    - verbose: Whether to print debug information.
    - training_size: List of training sizes for each replication.
    - score: Scoring function to evaluate the model.
    - s_type: Type of score to use.
    - B: Number of replications for the MultiSplit algorithm.
    - lambda_: Scaling factor for the alpha value.
    - tau: Truncation threshold for the MultiSplit intervals (optional).

    Returns:
    - A dictionary containing the lower ('lo') and upper ('up') bounds of the prediction intervals.
    """


def conformal_multidim_msplit(x, y, x0, train_fun, predict_fun, alpha=0.1,
                              split=None, seed=None, randomized=False, seed_beta=False,
                              verbose=False, training_size=None, score="max",
                              s_type="st-dev", B=100, lambda_=0, tau=None):
    if tau is None:
        tau = 1 - (B + 1) / (2 * B)
    if training_size is None or len(training_size) != B:
        training_size = [0.5] * B
    if seed is not None:
        np.random.seed(seed)

    n0, q = x0.shape[0], y.shape[1]
    full = q * n0
    alpha_adjusted = alpha * (1 - tau) + (alpha * lambda_) / B

    # Initialize lists to store results for each bootstrap iteration
    lo_list = []
    up_list = []
    pred_list = []

    # Pre-allocate array for storing flattened lower and upper bounds
    # lo_up will have shape (B, 2 * n0 * q), where:
    #   - B: number of bootstrap iterations
    #   - n0: number of test points
    #   - q: number of response variables
    lo_up = np.zeros((2 * B, n0 * q))  # Shape: (B, 2 * n0 * q)

    # Loop through bootstrap iterations
    for b in range(B):
        # Call conformal_multidim_split function
        out = conformal_multidim_split(
            x, y, x0,
            train_fun, predict_fun,
            alpha_adjusted,
            split, seed + b,
            randomized, seed_beta,
            verbose, training_size[b],
            score, s_type
        )

        # Extract lower, upper bounds, and predictions (shape: (n0, q))
        lo = out['lo']  # Shape: (n0, q)
        up = out['up']  # Shape: (n0, q)
        pred = out['pred']  # Shape: (n0, q)

        # Flatten and store in the pre-allocated lo_up array
        # Flatten lo and up and store them into the respective columns of lo_up
        lo_up[b, :] = lo.flatten()  # Store flattened lower bounds
        lo_up[b + B - 1, :] = up.flatten()  # Store flattened upper bounds

    # Final result matrices
    final_lo_up = lo_up.flatten()  # Shape: (B * 2 * n0 * q,)
    # Combine bounds into a single array
    tr = tau * B + 0.001

    lower, upper = [], []
    # Build final intervals
    for k in range(full):
        lower_bound, upper_bound = interval_build(lo_up[:, k], B, tr)
        lower.append(lower_bound)
        upper.append(upper_bound)

    # Convert lists to numpy arrays for reshaping
    lower = np.array(lower)
    upper = np.array(upper)

    # Reshape to (n0, q)
    lo = lower.reshape(n0, q)
    up = upper.reshape(n0, q)

    model = train_fun(x, y)
    predictions_full = predict_fun(model, x)
    predictions_x0 = predict_fun(model, x0)

    return {'lo': lo, 'up': up, 'x0': x0, 'pred': predictions_x0}


def interval_build(yyy, B, tr):
    # Step 1: Create h array which alternates between 1 and 0 for B times
    h = np.tile([1] * B + [0] * B, 1)

    # Step 2: Order yyy according to the modified h array (2 - h)
    o = np.argsort(yyy * (2 - h))  # Sorting based on the 2 - h scheme

    ys = np.array(yyy)[o]  # Apply the sorting order to yyy
    hs = h[o]  # Apply the same sorting order to h

    count = 0
    leftend = 0
    lo = up = 0

    # Step 3: Loop over 2 * B elements
    for j in range(2 * B):
        if hs[j] == 1:
            count += 1
            if count > tr and (count - 1) <= tr:
                leftend = ys[j]
        else:
            if count > tr and (count - 1) <= tr:
                rightend = ys[j]
                lo = leftend
                up = rightend
            count -= 1

    # Step 4: Return the lower and upper bounds as a tuple
    return (lo, up)
