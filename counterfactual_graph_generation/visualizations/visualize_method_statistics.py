
import os

import click
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

naming_dict = {
    "main_method" : "Classifier Guided CF",
    "closest_latent_train": "Decoded Mean of k-NN",
    "closest_graph_train": "Graph of NN from Training",
    "random_sampled_CFs_list": "Random Sampling from Prior"
}

fidelity_measures = ["Latent Euclidean Distance (Fidelity)", "Latent Cossimilary (Fidelity)", "GED (Fidelity)"]
y_axis_fr = 'Cummulative flip-rate'
y_axis_cc = 'Cummulative confidence'
y_axis_names = [y_axis_fr, y_axis_cc]

color_map = {
    'main_method_counterfactual_statistics.csv': '#3C5488FF',
    'random_sampled_CFs_list_counterfactual_statistics.csv': '#00A087FF',
    'closest_latent_train_counterfactual_statistics.csv': '#E64B35FF',
    'closest_graph_train_counterfactual_statistics.csv': '#4DBBD5FF',
}


def plot_line(df, x_col, y_col, method, label=None, ax=None):
    """
    Plots a line from a DataFrame.

    Parameters:
    - df: pandas DataFrame containing the data.
    - x_col: str, the name of the column for the x-axis.
    - y_col: str, the name of the column for the y-axis.
    - label: str, label for the line (used in the legend).
    - ax: matplotlib Axes object, optional. If None, a new plot is created.

    Returns:
    - ax: the Axes object with the plot.
    """
    if ax is None:
        ax = plt.gca()  # Get current Axes instance

    ax.plot(df[x_col], df[y_col], label=label, c=color_map[method])
    return ax

def skip_path(path):
    return path.split('_')[-4] == 'statistics/aggregated'

@click.command()
@click.option('--prefix', default="test_nci1", help='Dataset configuration prefix <split>_<dataset-name>')
@click.option('--svg', default=False, help='Whether the plot should be made as .svg')
@click.option('--normalize', default=False, help='Normalization procedure')
@click.option('--exclude', default=10, help='Number of points to exclude from the top of the dataframe when plotting')
def main(prefix, svg, normalize, exclude):
    dir_path = f'./data/predictions/{prefix}_method_statistics/'
    paths = [os.path.join(dirpath,f) for (dirpath, dirnames, filenames) in os.walk(dir_path) for f in filenames] # Gets paths of all method files
    # Create figure specs:
    fig, axs = plt.subplots(3, len(fidelity_measures), figsize=(18, 14))

    gs = fig.add_gridspec(3, len(fidelity_measures), height_ratios=[1, 1, 0.3]) # Make the last row narrow
    for i in range(3):
        for j, _ in enumerate(fidelity_measures):
            axs[i, j].set_position(gs[i, j].get_position(fig))  # Re-position axes using the gridspec
            axs[i, j].set_subplotspec(gs[i, j])
            axs[i, j].tick_params(axis='both', which='major', labelsize=18)

    min_max_df = {}
    for k, path in enumerate(paths): # Iterates over all methods adding them iteratively
        min_max_df[path] = []
        df = pd.read_csv(path)
        if skip_path(path):
            continue
        method = path.split("/")[-1]
        # Extract label
        words = method.split('_')[:-2]
        label = '_'.join(words)
        for j, fidelity_measure in enumerate(fidelity_measures): # Iterates over all fidelity measures
            n = len(df)
            if "Latent Cossimilary (Fidelity)" == fidelity_measure:
                sorted_df = df.sort_values(fidelity_measure, ascending=False)
                sorted_df[y_axis_fr] = [(sorted_df['Target Class'][:i] == sorted_df['Predicted Counterfactual Class'][:i]).sum() for i in range(len(sorted_df))]
                sorted_df[y_axis_cc] = [(sorted_df['Absolute Difference (Validity)'][:i]).sum() for i in range(len(sorted_df))]
                sorted_df['Negative Cossimilary (Fidelity)'] = -1*sorted_df[fidelity_measure]
                fidelity_measure = 'Negative Cossimilary (Fidelity)'
            else:
                sorted_df = df.sort_values(fidelity_measure, ascending=True)
                sorted_df[y_axis_fr] = [(sorted_df['Target Class'][:i] == sorted_df['Predicted Counterfactual Class'][:i]).sum() for i in range(len(sorted_df))]
                sorted_df[y_axis_cc] = [(sorted_df['Absolute Difference (Validity)'][:i]).sum() for i in range(len(sorted_df))]
            # Normalize df
            if normalize:
                sorted_df[y_axis_fr] = sorted_df[y_axis_fr] / np.array(range(1, n+1))
                sorted_df[y_axis_cc] = sorted_df[y_axis_cc] / np.array(range(1, n+1))
            else:
                sorted_df[y_axis_fr] = sorted_df[y_axis_fr] / n
                sorted_df[y_axis_cc] = sorted_df[y_axis_cc] / n
            # Burn in dataframe:
            sorted_df[y_axis_fr] = sorted_df[y_axis_fr].iloc[exclude:]
            sorted_df[y_axis_cc] = sorted_df[y_axis_cc].iloc[exclude:]
            # Plot
            plot_line(sorted_df, fidelity_measure, y_axis_fr, label=naming_dict[label], ax=axs[0, j], method=method)
            plot_line(sorted_df, fidelity_measure, y_axis_cc, ax=axs[1, j], method=method)
            # Log min-max values
            drop_nan_df = sorted_df.dropna(subset=[fidelity_measure])
            if len(drop_nan_df) != n:
                print(f'Dropped NAN values for {fidelity_measure}')
            first_and_last_df = drop_nan_df.iloc[[0, -1]]
            min_max_df[path].append(first_and_last_df)


    for j, y_name in enumerate(y_axis_names):
        y_upper_lim = 0
        y_lower_lim = 0
        for i, fidelity_measure in enumerate(fidelity_measures):
            x_lower_lim, x_upper_lim = axs[j, i].get_xlim()
            y_lower_lim_, y_upper_lim_ = axs[j, i].get_ylim()
            y_upper_lim = max(y_upper_lim, y_upper_lim_)
            y_lower_lim = min(y_lower_lim, y_lower_lim_)
            for k, path in enumerate(paths):
                if skip_path(path):
                    continue
                method = path.split("/")[-1]
                fidelity_measure = 'Negative Cossimilary (Fidelity)' if fidelity_measure == "Latent Cossimilary (Fidelity)" else fidelity_measure
                df = min_max_df[path][i]#.sort_values(fidelity_measure, ascending=True)
                y_min, y_max = df[y_name].iloc[0:2]
                x_min, x_max = df[fidelity_measure].iloc[0:2]
                # Gets deine extensions
                x_max_extension = np.linspace(x_max, x_upper_lim, 50)
                x_min_extension = np.linspace(x_lower_lim, x_min, 50)
                # Plot curve
                axs[j, i].plot(x_max_extension, np.full_like(x_max_extension, y_max), linestyle='--', c=color_map[method])
                axs[j, i].plot(x_min_extension, np.full_like(x_min_extension, y_min), linestyle='--', c=color_map[method])

    # Plot edits
    for i, fidelity_measure in enumerate(fidelity_measures):
        # Set y-limits:
        axs[0, i].set_ylim(bottom=-0.05, top=1.05)
        axs[0, i].grid(axis='y', which='both', linestyle='--', linewidth=0.7, color='lightgray')
        axs[1, i].set_ylim(bottom=y_lower_lim-0.03, top=y_upper_lim + 0.05)
        axs[1, i].grid(axis='y', which='both', linestyle='--', linewidth=0.7, color='lightgray')
        #axs[1, i].grid(True, which='both', linestyle='--', linewidth=0.7, color='lightgray')
        axs[2, i].set_xlabel(fidelity_measure, fontsize=20)
    axs[0, 0].set_ylabel('Flip-Ratio - Validity', fontsize=20)
    axs[1, 0].set_ylabel('Mean Absolute Difference - Validity', fontsize=20)

    # Add histograms
    y_lower_lim, y_upper_lim = 0, 0
    for i, fidelity_measure in enumerate(fidelity_measures):
        for k, path in enumerate(paths):
            if skip_path(path):
                continue
            method = path.split("/")[-1]
            df = pd.read_csv(path)
            if "Latent Cossimilary (Fidelity)" == fidelity_measure:
                axs[-1,i].hist(-1*df[fidelity_measure], bins=10, color=color_map[method], alpha=0.3)
            else:
                axs[-1,i].hist(df[fidelity_measure], bins=10, color=color_map[method], alpha=0.3)
        y_lower_lim_, y_upper_lim_ = axs[-1, i].get_ylim()
        y_lower_lim, y_upper_lim = min(y_lower_lim, y_lower_lim_), max(y_upper_lim, y_upper_lim_)

    # Set y-axis scale on histograms to be the same:
    for i, fidelity_measure in enumerate(fidelity_measures):
        axs[-1, i].set_ylim(bottom=y_lower_lim-0.03, top=y_upper_lim + 0.05)
        axs[-1, i].set_ylim(bottom=y_lower_lim-0.03, top=y_upper_lim + 0.05)
        axs[-1, i].set_ylim(bottom=y_lower_lim-0.03, top=y_upper_lim + 0.05)

    # Add legend
    axs[0,-1].legend(loc="upper right", fontsize=18)
    #fig.suptitle(f'{prefix} (this title should not be included in paper))', fontsize=26)
    if svg:
        plt.savefig(f'./data/predictions/fidelity_validity_plots/{prefix}_fidelity-validity-plot.svg')
    else:
        plt.savefig(f'./data/predictions/fidelity_validity_plots/{prefix}_fidelity-validity-plot')
    return

if __name__ == '__main__':
    main()
