import matplotlib.pyplot as plt
import pandas as pd


def plot_multi_full(full, figsize=(8, 6)):
    """
    Plot Confidence Regions using scatter plots obtained from Full Conformal Prediction.

    Parameters:
        full (dict): The output of the multivariate full conformal prediction function.
                     It should contain:
                     - `valid_points` (list): A list of dataframes with confidence region data,
                                              each having three unnamed columns corresponding to
                                              "Var1", "Var2", and "pval".
                     - `pred` (numpy.ndarray): A 2D array where each row represents a prediction point.
        figsize (tuple, optional): Size of each plot. Default is (8, 6).

    Returns:
        list: A list of matplotlib figures, each representing the confidence region for
              a test observation.
    """
    valid_points = full["valid_points"]
    predictions = full["pred"]

    num_test_samples = len(valid_points)
    plots = []

    for k in range(num_test_samples):
        df = pd.DataFrame(valid_points[k])
        df.columns = ["Var1", "Var2", "pval"]

        pred_point = predictions[k]

        fig, ax = plt.subplots(figsize=figsize)

        # Scatter plot of points
        sc = ax.scatter(df["Var1"], df["Var2"], c=df["pval"], cmap="RdPu", s=100, alpha=0.7, label="Confidence Region")

        # Prediction point
        ax.scatter(pred_point[0], pred_point[1], color='blue', marker='*', s=200, label="Prediction Point")

        ax.set_title(f"Test Observation {k + 1}", fontsize=14)
        ax.set_xlabel("y1", fontsize=12)
        ax.set_ylabel("y2", fontsize=12)
        ax.legend()

        # Add color bar
        plt.colorbar(sc, ax=ax, label='-pval')

        plots.append(fig)

        # Show the plot explicitly after each one is created
        plt.show()

    return plots


def plot_multi_full_contour(full, figsize=(8, 6)):
    """
    Plot Confidence Regions using contour plots obtained from Full Conformal Prediction.

    Parameters:
        full (dict): The output of the multivariate full conformal prediction function.
                     It should contain:
                     - `valid_points` (list): A list of dataframes with confidence region data,
                                              each having three unnamed columns corresponding to
                                              "Var1", "Var2", and "pval".
                     - `pred` (numpy.ndarray): A 2D array where each row represents a prediction point.
        figsize (tuple, optional): Size of each plot. Default is (8, 6).

    Returns:
        list: A list of matplotlib figures, each representing the confidence region for
              a test observation.
    """
    valid_points = full["valid_points"]
    predictions = full["pred"]

    num_test_samples = len(valid_points)
    plots = []

    for k in range(num_test_samples):
        df = pd.DataFrame(valid_points[k])
        df.columns = ["Var1", "Var2", "pval"]

        pred_point = predictions[k]

        fig, ax = plt.subplots(figsize=figsize)

        # Create a grid of points for contour plot
        x = df["Var1"]
        y = df["Var2"]
        z = df["pval"]

        # Contour plot
        ax.tricontourf(x, y, z, cmap="RdPu", levels=10, alpha=0.7)

        # Add the prediction point
        ax.scatter(pred_point[0], pred_point[1], color='blue', marker='*', s=200, label="Prediction Point")

        ax.set_title(f"Test Observation {k + 1}", fontsize=14)
        ax.set_xlabel("y1", fontsize=12)
        ax.set_ylabel("y2", fontsize=12)
        ax.legend()

        plots.append(fig)

        plt.show()

    return plots
