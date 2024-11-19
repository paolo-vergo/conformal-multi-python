import numpy as np

from src.helpers.helpers import check_input_validation, calculate_scores, compute_residuals, apply_mad_scaling, \
    generate_test_points_grid

"""
Full Conformal Prediction Intervals for Multivariate Response.

Compute prediction intervals using full conformal inference for multivariate responses.

Parameters
----------
features : np.ndarray
    Training feature matrix of shape (n_samples, n_features).
responses : np.ndarray
    Training response matrix of shape (n_samples, n_responses).
new_features : np.ndarray
    New feature matrix of shape (n_predictions, n_features) for predictions.
train_function : function
    Function to train the model and return an estimator of E(Y|X). 
    Accepts features, responses, and optionally a model state.
predict_function : function
    Function to predict responses for new feature values.
    Accepts the trained model and new feature values.
alpha : float, optional
    Miscoverage level for prediction intervals. Default is 0.1.
mad_train_function : function, optional
    Function to estimate expected absolute residuals. Default is None.
mad_predict_function : function, optional
    Function to predict mean absolute residuals for new feature values. Default is None.
score_method : str, optional
    Method to compute the nonconformity score. Options: 'l2', 'mahalanobis', 'max', 'scaled.max'.
    Default is 'l2'.
grid_points_per_dimension : int, optional
    Number of grid points per response dimension for conformal intervals. Default is 100.
grid_expansion_factor : float, optional
    Factor to scale the range of grid values around observed responses. Default is 1.25.
verbose : bool, optional
    If True, prints intermediate progress. Default is False.

Returns
-------
dict
    Contains:
    - 'valid_points': A list of dataframes for each prediction point with valid response grid points and p-values.
    - 'predictions': Predicted responses as a dataframe with shape (n_predictions, n_responses).

Notes
-----
- Assumes multivariate responses are exchangeable.
- Restricted to bivariate responses due to computational overhead.
"""

def conformal_multidim_full(
    x: np.ndarray, y: np.ndarray, x0: np.ndarray,
    train_fun, predict_fun, alpha: float = 0.1,
    mad_train_fun=None, mad_predict_fun=None,
    score: str = 'l2', num_grid_pts_dim: int = 100,
    grid_factor: float = 1.25, verbose: bool = False
) -> dict:
    """
    Main function that calculates valid conformal prediction sets for multidimensional data.
    """
    # Convert inputs and reshape x0
    x, y, x0 = map(np.array, (x, y, x0))
    x0 = x0.reshape(-1, x.shape[1])
    n, p = x.shape
    n0, q = x0.shape[0], y.shape[1]

    # Validate input parameters
    check_input_validation(num_grid_pts_dim, grid_factor)

    # Train the main model and predict for query points
    if verbose: print("Initial training on full data set...")
    main_model_out = train_fun(x, y)
    pred = predict_fun(main_model_out, x0).reshape(n0, q)

    # Generate grid points for target values
    yvals = generate_test_points_grid(y, grid_factor, num_grid_pts_dim)

    # Compute valid points for each query point
    valid_points = [
        process_query_point(
            k, x, y, x0[k], yvals, train_fun, predict_fun,
            mad_train_fun, mad_predict_fun, score, alpha, verbose, main_model_out
        )
        for k in range(n0)
    ]

    return {"valid_points": valid_points, "pred": pred}




def process_query_point(
    k: int, x: np.ndarray, y: np.ndarray, x0_k: np.ndarray,
    yvals: np.ndarray, train_fun, predict_fun,
    mad_train_fun, mad_predict_fun, score: str, alpha: float, verbose: bool, main_model_out
) -> np.ndarray:
    """
    Process a single query point and compute valid target values.
    """
    if verbose: print(f"Processing point {k + 1}...")

    xx = np.vstack([x, x0_k])
    pvals = np.zeros(yvals.shape[0])

    # Iterate over each candidate y-value combination
    for j, yval in enumerate(yvals):
        yy = np.vstack([y, yval])
        residuals = compute_residuals(xx, yy, train_fun, predict_fun, j, main_model_out)

        # Apply MAD scaling if provided
        if mad_train_fun and mad_predict_fun:
            residuals = apply_mad_scaling(
                xx, residuals, mad_train_fun, mad_predict_fun, j
            )

        # Calculate scores and compute p-values
        ncm = calculate_scores(residuals, score)
        pvals[j] = np.sum(ncm >= ncm[-1]) / (xx.shape[0] + 1)

    # Identify valid points based on p-values
    valid_indices = np.where(pvals > alpha)[0]
    return np.hstack([yvals[valid_indices], pvals[valid_indices, None]])
