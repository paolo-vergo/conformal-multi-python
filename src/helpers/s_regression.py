import numpy as np


def compute_s_regression(mat_residual, type, alpha, tau):
    check_s_regression(mat_residual, type)
    q = mat_residual.shape[1]

    # Naive cases: just one observation and type in ["st-dev", "alpha-max"]
    if isinstance(mat_residual, np.ndarray) and mat_residual.ndim == 1 and type == "st-dev":
        raise ValueError("st-dev cannot be computed when the number of observations is equal to 1.")

    if isinstance(mat_residual, np.ndarray) and mat_residual.ndim == 1 and type == "alpha-max":
        return np.abs(mat_residual)

    # Non-naive cases
    if type == "identity":
        return np.ones(q)

    if type == "st-dev":
        return np.std(mat_residual, axis=0)

    if type == "alpha-max":
        check_num_01(tau)

        abs_mat_residual = np.abs(mat_residual)

        # Check on alpha
        if np.ceil(abs_mat_residual.shape[0] + tau - (abs_mat_residual.shape[0] + 1) * alpha) >= abs_mat_residual.shape[
            0]:
            return np.max(abs_mat_residual, axis=0)

        if np.ceil(abs_mat_residual.shape[0] + tau - (abs_mat_residual.shape[0] + 1) * alpha) <= 0:
            return np.ones(q)

        # S ALPHA-MAX
        sequence_sup = np.max(abs_mat_residual, axis=1)
        gamma = np.sort(sequence_sup)[
            int(np.ceil(abs_mat_residual.shape[0] + tau - (abs_mat_residual.shape[0] + 1) * alpha)) - 1]
        position_functions_in_H = np.where(sequence_sup <= gamma)[0]
        return np.max(abs_mat_residual[position_functions_in_H, :], axis=0)


def extremes(l, tau, alpha, rho, k_s):
    if np.ceil(l + tau - (l + 1) * alpha) == 1:
        v = 0
    else:
        v = np.sum(np.sort(rho)[0:int(np.ceil(l + tau - (l + 1) * alpha)) - 1] == k_s)

    if np.ceil(l + tau - (l + 1) * alpha) == l:
        r = 0
    else:
        r = np.sum(np.sort(rho)[int(np.ceil(l + tau - (l + 1) * alpha)):] == k_s)

    return tau > (alpha * (l + 1) - np.floor(alpha * (l + 1) - tau) + r) / (r + v + 2)


def check_s_regression(mat_residual, type):
    if not isinstance(mat_residual, (np.ndarray, np.matrix)) and not isinstance(mat_residual, list):
        raise ValueError("mat_residual must be either a matrix, a dataframe, or an atomic vector (naive case).")

    possible_s_functions = ["identity", "st-dev", "alpha-max"]
    if type not in possible_s_functions:
        raise ValueError(
            f"The 'type' argument is not correct. Please select one of the following: {', '.join(possible_s_functions)}.")


def check_num_01(tau):
    if not (0 <= tau <= 1):
        raise ValueError("tau must be a number between 0 and 1.")
