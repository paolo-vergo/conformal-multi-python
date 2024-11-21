import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
import xgboost as xgb


def mean_multi():
    """
    Mean model for multivariate data. Predicts the column-wise mean of the training data.
    """
    def train_fun(x, y, out=None):
        return {"m": y.mean(axis=0)}

    def predict_fun(out, newx):
        mean_vector = out["m"]
        return np.tile(mean_vector, (newx.shape[0], 1))

    return {"train_fun": train_fun, "predict_fun": predict_fun}


def multivariate_linear_regressor():
    """
    Linear regression model for multivariate data. Trains a separate regressor for each target dimension.
    """
    def train_fun(x, y, out=None):
        models = [LinearRegression().fit(x, y[:, i]) for i in range(y.shape[1])]
        return {"models": models}

    def predict_fun(out, newx):
        models = out["models"]
        return np.column_stack([model.predict(newx) for model in models])

    return {"train_fun": train_fun, "predict_fun": predict_fun}


def multivariate_random_forest():
    """
    Random forest regressor for multivariate data. Trains a separate random forest for each target dimension.
    """
    def train_fun(x, y, out=None):
        models = [RandomForestRegressor(n_estimators=100, random_state=42).fit(x, y[:, i]) for i in range(y.shape[1])]
        return {"models": models}

    def predict_fun(out, newx):
        models = out["models"]
        return np.column_stack([model.predict(newx) for model in models])

    return {"train_fun": train_fun, "predict_fun": predict_fun}


def multivariate_nearest_neighbor():
    """
    Nearest neighbor predictor for multivariate data. Uses the closest training point for predictions.
    """
    def train_fun(x, y, out=None):
        return {"x_train": x, "y_train": y}

    def predict_fun(out, newx):
        x_train, y_train = out["x_train"], out["y_train"]
        nn = NearestNeighbors(n_neighbors=1).fit(x_train)
        _, indices = nn.kneighbors(newx)
        return y_train[indices.flatten()]

    return {"train_fun": train_fun, "predict_fun": predict_fun}


def multivariate_xgboost():
    """
    XGBoost regressor for multivariate data. Trains a separate XGBoost model for each target dimension.
    """
    def train_fun(x, y, out=None):
        models = []
        for i in range(y.shape[1]):
            dtrain = xgb.DMatrix(x, label=y[:, i])
            model = xgb.train(params={"objective": "reg:squarederror"}, dtrain=dtrain, num_boost_round=100)
            models.append(model)
        return {"models": models}

    def predict_fun(out, newx):
        models = out["models"]
        dtest = xgb.DMatrix(newx)
        return np.column_stack([model.predict(dtest) for model in models])

    return {"train_fun": train_fun, "predict_fun": predict_fun}


def get_prediction_model(model_name="xgboost"):
    """
    Retrieve the appropriate prediction model based on the given name.

    Parameters:
        model_name (str): Name of the model to use. Options are:
                          "mean", "linear", "random_forest", "nearest_neighbor", "xgboost".

    Returns:
        dict: A dictionary containing 'train_fun' and 'predict_fun' for the chosen model.
    """
    models = {
        "mean": mean_multi,
        "linear": multivariate_linear_regressor,
        "random_forest": multivariate_random_forest,
        "nearest_neighbor": multivariate_nearest_neighbor,
        "xgboost": multivariate_xgboost,
    }

    if model_name not in models:
        raise ValueError(f"Model '{model_name}' is not supported. Choose from {list(models.keys())}.")

    return models[model_name]()
