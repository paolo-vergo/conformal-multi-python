import numpy as np
from scipy.stats import multivariate_normal

from src.confidence_regions.split_conformal import conformal_multidim_split
from src.plots.plot_split_conformal import plot_multidim_split
from src.helpers.prediction_models import mean_multi

# Parameters
n = 40  # Number of training samples
n0 = 2  # Number of prediction samples
p = 2  # Number of features
mu = np.zeros(p)  # Mean vector for multivariate normal

# Generate training features (x) from multivariate normal distribution
x = multivariate_normal.rvs(mean=mu, size=n)

# Define grid
my_grid = np.linspace(0, 1, num=2)

# Generate responses (y) based on a nonlinear function
y = np.array([
    [u[0] + u[1] * np.cos(6 * np.pi * g) for g in my_grid] for u in x
])

x0 = multivariate_normal.rvs(mean=mu, size=n0)

# Select the function
fun = mean_multi()

# Split conformal

final_point = conformal_multidim_split(
    x=x, y=y, x0=x0,
    train_fun=fun["train.fun"], predict_fun=fun["predict.fun"],
    alpha=0.1, split=None, seed=None, randomized=False, seed_tau=None,
    verbose=False, training_size=0.5, score="l2", s_type="st-dev"
)

ppp2 = plot_multidim_split(final_point)