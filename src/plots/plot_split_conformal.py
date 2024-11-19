import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_multidim_split_custom(split, same_scale=False, color="blue", ci_color="red", figsize=(10, 8)):
    """
    Plot Confidence Regions obtained with Split Conformal.
    If input data has 2 dimensions, create separate 3D plots for each response dimension.

    Parameters:
        split (dict): The output of a split multivariate conformal prediction function.
                      Must contain:
                      - 'x0': Independent variables (n0 x p matrix).
                      - 'lo': Lower bounds for confidence regions (n0 x q matrix).
                      - 'up': Upper bounds for confidence regions (n0 x q matrix).
                      - 'pred': Predictions (n0 x q matrix).
        same_scale (bool): If True, forces the same scale for all y-axes. Default is False.
        color (str): Color for predicted points. Default is "blue".
        ci_color (str): Color for confidence interval shading. Default is "red".
        figsize (tuple): Figure size for the plot. Default is (10, 8).

    Returns:
        matplotlib.figure.Figure: The figure containing the plot(s).
    """

    # Extract data
    x0 = np.array(split["x0"])
    lo = np.array(split["lo"])
    up = np.array(split["up"])
    pred = np.array(split["pred"])

    if not (x0.shape[0] == lo.shape[0] == up.shape[0] == pred.shape[0]):
        raise ValueError("All input matrices must have the same number of rows (n0).")

    # Check for 2D input data (p=2)
    p = x0.shape[1]
    q = lo.shape[1]  # Number of predictions (response dimensions)

    # If the input data has 2 dimensions (p=2), create separate 3D plots for each response
    if p == 2:
        fig, axes = plt.subplots(1, q, figsize=(figsize[0] * q, figsize[1]), subplot_kw={'projection': '3d'})

        if q == 1:
            axes = [axes]  # Ensure axes is iterable even if there's only one plot

        for jj in range(q):
            ax = axes[jj]

            # Prepare data for 3D plot for the jj-th response
            x1 = x0[:, 0]  # First independent variable (x-axis)
            x2 = x0[:, 1]  # Second independent variable (y-axis)
            y_pred = pred[:, jj]  # Predictions for the jj-th response
            y_min = lo[:, jj]  # Lower bound for confidence interval for the jj-th response
            y_max = up[:, jj]  # Upper bound for confidence interval for the jj-th response

            # Scatter plot for predicted points
            ax.scatter(x1, x2, y_pred, color=color, label="Predicted Points")

            # Create meshgrid for x1 and x2
            x1_grid, x2_grid = np.meshgrid(np.linspace(np.min(x1), np.max(x1), 100),
                                           np.linspace(np.min(x2), np.max(x2), 100))

            # Interpolate y_min and y_max over the grid (assuming linear interpolation)
            y_min_grid = np.interp(x1_grid.flatten(), x1, y_min)  # Interpolate based on x1
            y_max_grid = np.interp(x1_grid.flatten(), x1, y_max)  # Interpolate based on x1

            # Reshape back to match the grid dimensions
            y_min_grid = y_min_grid.reshape(x1_grid.shape)
            y_max_grid = y_max_grid.reshape(x2_grid.shape)

            # Create surface plot for the confidence region
            ax.plot_surface(x1_grid, x2_grid, y_min_grid, color=ci_color, alpha=0.3, label="Confidence Region (Min)")
            ax.plot_surface(x1_grid, x2_grid, y_max_grid, color=ci_color, alpha=0.3, label="Confidence Region (Max)")

            # Set labels and title
            ax.set_xlabel("x1")
            ax.set_ylabel("x2")
            ax.set_zlabel("Prediction")
            ax.set_title(f"Confidence Region for y{jj + 1} (p=2)", fontsize=14)

            # Add legend
            ax.legend(loc="upper right")

        plt.tight_layout()

    else:
        # If p > 2, default to 2D subplots (as shown previously)
        y_min_global, y_max_global = (np.min(lo), np.max(up)) if same_scale else (None, None)
        padding = 0.05  # Extra padding for y-axis limits

        fig, axes = plt.subplots(p, q, figsize=figsize, squeeze=False)
        fig.suptitle("Confidence Regions for Split Conformal Predictions", fontsize=16)

        # Generate subplots
        for ii in range(p):
            for jj in range(q):
                ax = axes[ii, jj]
                df = pd.DataFrame({
                    'xd': x0[:, ii],
                    'y_pred': pred[:, jj],
                    'y_min': lo[:, jj],
                    'y_max': up[:, jj]
                })

                sns.scatterplot(
                    data=df, x='xd', y='y_pred', ax=ax,
                    label="Predicted Points", color=color, s=20
                )

                ax.fill_between(
                    df['xd'], df['y_min'], df['y_max'],
                    color=ci_color, alpha=0.2, label="Confidence Interval"
                )

                ax.set_xlabel(f"x{ii + 1}")
                ax.set_ylabel(f"y{jj + 1}")
                ax.set_title(f"y{jj + 1} vs. x{ii + 1}", fontsize=10)

                if same_scale:
                    ax.set_ylim(y_min_global - padding, y_max_global + padding)
                else:
                    y_min, y_max = df[['y_min', 'y_pred']].min().min(), df[['y_max', 'y_pred']].max().max()
                    ax.set_ylim(y_min - padding, y_max + padding)

                if ii == 0 and jj == q - 1:
                    ax.legend(loc="upper right", fontsize=8)
                else:
                    ax.legend().remove()

        plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.show()
    return fig
