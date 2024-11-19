import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_multidim_split(split, same_scale=False):
    """
    Plot Confidence Regions obtained with Split Conformal.

    Parameters:
        split (dict): The output of a split multivariate conformal prediction function.
                      It should contain 'x0', 'lo', 'up', and 'pred'.
                      - x0: Data for independent variables (n0 x p matrix)
                      - lo: Lower bounds for the confidence region (n0 x q matrix)
                      - up: Upper bounds for the confidence region (n0 x q matrix)
                      - pred: Predictions (n0 x q matrix)
        same_scale (bool, optional): Should the same scale be forced for all the y-axes? Default is False.

    Returns:
        list: A list of matplotlib figures, each representing a confidence region for the test observations.
    """

    # Extract the data
    x0 = np.array(split["x0"])
    lo = np.array(split["lo"])
    up = np.array(split["up"])
    pred = np.array(split["pred"])

    # Find bounds for the plots
    if same_scale:
        y_up = np.max(up) + 0.01 * np.std(up)
        y_lo = np.min(lo) - 0.01 * np.std(lo)
    else:
        y_up, y_lo = None, None

    # Define dimensions
    p = x0.shape[1]  # Number of predictors
    q = lo.shape[1]  # Number of predictions

    # Initialize list to store plots
    plots = []

    # Generate the plots
    for ii in range(p):
        for jj in range(q):
            # Prepare the dataframe for plotting
            df = pd.DataFrame({
                'xd': x0[:, ii],
                'yg': pred[:, jj],
                'y_min': lo[:, jj],
                'y_max': up[:, jj]
            })

            # Create the plot
            fig, ax = plt.subplots(figsize=(8, 6))

            # Heatmap for confidence regions using a scatter plot
            sns.scatterplot(data=df, x='xd', y='yg', ax=ax, label="Prediction Point", color="blue")

            # Plot confidence interval as a vertical line
            ax.errorbar(df['xd'], df['yg'], yerr=[df['yg'] - df['y_min'], df['y_max'] - df['yg']], fmt='o',
                        color='red', label="Confidence Interval", elinewidth=2)

            # Customize plot labels
            ax.set_xlabel(f"x {ii + 1}", fontsize=12)
            ax.set_ylabel(f"y {jj + 1}", fontsize=12)

            # Set y-limits if the same scale is applied
            if same_scale:
                ax.set_ylim(y_lo, y_up)

            ax.set_title(f"Confidence Interval for y{jj + 1} and x{ii + 1}", fontsize=14)
            ax.legend()

            # Append the plot to the list
            plots.append(fig)

    # Display all the plots
    plt.show()

    return plots
