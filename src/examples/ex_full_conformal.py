import numpy as np

from src.confidence_regions.full_conformal import conformal_multidim_full
from src.helpers.generate_random_data import generate_random_data
from src.plots.plot_full_conformal import plot_multi_full
from src.prediction_models.models import get_prediction_model

n = 20  # Number of training samples
n0 = 2  # Number of prediction samples
p = 2  # Number of features
q = 2  # Dimension of the response
mu = np.linspace(0, 1, p)
variances = np.array([5, 20])
x, y, x0 = generate_random_data(n, n0, p, mu, variances, q)

my_grid = np.linspace(0, 1, num=2)

fun = get_prediction_model("mean")

final_full = conformal_multidim_full(
    x, y, x0,
    fun["train_fun"],
    fun["predict_fun"],
    score="l2",
    num_grid_pts_dim=50,
    grid_factor=1.25,
    verbose=False
)

plot_multi_full(final_full)
