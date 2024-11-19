import numpy as np


def mean_multi():
    # Training function
    def train_fun(x, y, out=None):
        # Compute the column-wise mean of y
        m = y.mean(axis=0)
        return {"m": m}

    # Prediction function
    def predict_fun(out, newx):
        # Retrieve the mean vector
        temp = out["m"]
        n0 = newx.shape[0]
        # Repeat the mean vector for each row in newx
        sol = np.tile(temp, (n0, 1))
        return sol

    return {"train.fun": train_fun, "predict.fun": predict_fun}
