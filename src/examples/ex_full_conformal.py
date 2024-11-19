import numpy as np
from scipy.stats import multivariate_normal

from src.helpers.prediction_models import mean_multi, lm_multi
from src.plots.plot_full_conformal import plot_multidim_full_scatter
from src.confidence_regions.full_conformal import conformal_multidim_full

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

# Generate prediction features (x0) from multivariate normal distribution
x0 = multivariate_normal.rvs(mean=mu, size=n0)

# Select function (mean_multi as default, can be switched to lm_multi)
fun = mean_multi()
#fun = lm_multi()

final_full = conformal_multidim_full(
    x, y, x0,
    fun["train_fun"],
    fun["predict_fun"],
    score="l2",
    num_grid_pts_dim =50,
    grid_factor=1.25,
    verbose=False
)

plot_multidim_full_scatter(final_full)
#plot_multidim_full_3d(final_full)
