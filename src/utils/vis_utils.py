import matplotlib.pyplot as plt
import numpy as np

from utils import general_utils


def plot_scaled_distances(rhos, output_path, mask):
    rhos[:, ~mask] = np.nan
    general_utils.plot_results_in_subplots(output_path=output_path,
                                           data=rhos,
                                           figshape=[1, len(rhos)], show_bar=[True] * len(rhos))


def discrete_matshow(data, output_path):
    cmap = plt.get_cmap('RdBu', np.nanmax(data) - np.nanmin(data) + 1)
    mat = plt.matshow(data, cmap=cmap, vmin=np.nanmin(data) - 0.5,
                      vmax=np.nanmax(data) + 0.5)
    plt.colorbar(mat, ticks=np.arange(np.nanmin(data), np.nanmax(data) + 1))
    plt.savefig(output_path)
    plt.close()
