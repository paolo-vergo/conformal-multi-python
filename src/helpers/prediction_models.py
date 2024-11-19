from sklearn.linear_model import LinearRegression
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

    return {"train_fun": train_fun, "predict_fun": predict_fun}


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def lm_multi():
    # Training function
    def train_fun(x, y, out=None):
        q = y.shape[1]
        p = x.shape[1]

        # Combining x and the first column of y into a DataFrame (not strictly necessary in Python)
        df1 = pd.DataFrame(np.hstack([x, y[:, [0]]]))

        # Compute coefficients for each column of y
        coeff = np.array([
            LinearRegression(fit_intercept=True).fit(x, y[:, i]).coef_
            for i in range(q)
        ]).T

        intercepts = np.array([
            LinearRegression(fit_intercept=True).fit(x, y[:, i]).intercept_
            for i in range(q)
        ])

        coeff = np.vstack([intercepts, coeff])  # Combine intercepts with coefficients

        return {"coeff": coeff}

    # Prediction function
    def predict_fun(out, newx):
        c = out["coeff"]
        n0 = newx.shape[0]
        newxx = np.hstack([np.ones((n0, 1)), newx])
        sol = np.dot(newxx, c)
        return sol

    return {"train_fun": train_fun, "predict_fun": predict_fun}


