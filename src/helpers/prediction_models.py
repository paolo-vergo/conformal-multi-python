import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors


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


def multivariate_linear_regressor():
    def train_fun(x, y, out=None):
        # Train a separate linear regression model for each response dimension
        models = [LinearRegression().fit(x, y[:, i]) for i in range(y.shape[1])]
        return {"models": models}

    def predict_fun(out, newx):
        # Predict for each response dimension
        models = out["models"]
        predictions = np.column_stack([model.predict(newx) for model in models])
        return predictions

    return {"train_fun": train_fun, "predict_fun": predict_fun}

from sklearn.ensemble import RandomForestRegressor

def multivariate_random_forest():
    def train_fun(x, y, out=None):
        # Train a random forest regressor for each response dimension
        models = [RandomForestRegressor(n_estimators=100, random_state=42).fit(x, y[:, i]) for i in range(y.shape[1])]
        return {"models": models}

    def predict_fun(out, newx):
        # Predict using each random forest model
        models = out["models"]
        predictions = np.column_stack([model.predict(newx) for model in models])
        return predictions

    return {"train_fun": train_fun, "predict_fun": predict_fun}



def multivariate_nearest_neighbor():
    def train_fun(x, y, out=None):
        # Store the training data
        return {"x_train": x, "y_train": y}

    def predict_fun(out, newx):
        # Find the nearest training neighbor for each test point
        x_train = out["x_train"]
        y_train = out["y_train"]
        nn = NearestNeighbors(n_neighbors=1).fit(x_train)
        distances, indices = nn.kneighbors(newx)
        # Return corresponding y values for nearest neighbors
        predictions = y_train[indices.flatten()]
        return predictions

    return {"train_fun": train_fun, "predict_fun": predict_fun}


import xgboost as xgb
import numpy as np

def multivariate_xgboost():
    def train_fun(x, y, out=None):
        # Train an XGBoost model for each response dimension
        models = []
        for i in range(y.shape[1]):
            dtrain = xgb.DMatrix(x, label=y[:, i])
            model = xgb.train(params={"objective": "reg:squarederror"},
                              dtrain=dtrain,
                              num_boost_round=100)
            models.append(model)
        return {"models": models}

    def predict_fun(out, newx):
        # Predict using each XGBoost model
        models = out["models"]
        predictions = []
        for model in models:
            dtest = xgb.DMatrix(newx)
            predictions.append(model.predict(dtest))
        # Stack predictions into a matrix
        predictions = np.column_stack(predictions)
        return predictions

    return {"train_fun": train_fun, "predict_fun": predict_fun}

