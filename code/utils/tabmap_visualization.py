import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def cmap_map(function, cmap):
    """ Applies function (which should operate on vectors of shape 3: [r, g, b]), on colormap cmap.
    This routine will break any discontinuous points in a colormap.
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # Firt get the list of points where the segments start or end
    for key in ('red', 'green', 'blue'):
        step_dict[key] = list(map(lambda x: x[0], cdict[key]))
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step : np.array(cmap(step)[0:3])
    old_LUT = np.array(list(map(reduced_cmap, step_list)))
    new_LUT = np.array(list(map(function, old_LUT)))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(['red','green','blue']):
        this_cdict = {}
        for j, step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j, i]
            elif new_LUT[j,i] != old_LUT[j, i]:
                this_cdict[step] = new_LUT[j, i]
        colorvector = list(map(lambda x: x + (x[1], ), this_cdict.items()))
        colorvector.sort()
        cdict[key] = colorvector
    
    return matplotlib.colors.LinearSegmentedColormap('colormap', cdict, 1024)


light_jet = cmap_map(lambda x: x/1.1 + 0.2, matplotlib.cm.Spectral_r)


def plot_tabmap(tabmap, figsize=(3, 3), cmap=None, ax=None, point_scale=1.0):
    """
    Plots a tabular map using a scatter plot.

    Parameters:
    tabmap (np.ndarray): tabmap to plot.
    figsize (tuple): Figure size, default (3, 3).
    cmap (matplotlib.colors.Colormap): Colormap for the scatter plot, defaults to 'viridis' if None.
    ax (matplotlib.axes.Axes, optional): Axis object to plot on. Creates a new figure if None.
    point_scale (float): Scaling factor for the size of the points in the scatter plot.
    
    Returns:
    tuple: Figure and axis of the plot.
    """
    if cmap is None:
        cmap = light_jet
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    n_row, n_col = tabmap.shape
    center_offset = (n_row // 2) if n_row % 2 == 0 else (n_row - 1) // 2

    coords = np.array([(r - center_offset, c - center_offset) for r in range(n_row) for c in range(n_col)])
    values = tabmap.flatten()
    point_size = (200 * figsize[0] / n_row) * point_scale

    scatter = ax.scatter(coords[:, 1], -coords[:, 0], c=values, s=point_size, cmap=cmap or plt.get_cmap('viridis'), edgecolors="white")
    ax.set_axis_off()
    ax.set_aspect('equal')
    
    plt.colorbar(scatter, ax=ax, orientation='vertical')
    plt.tight_layout()

    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
    for _, spine in ax.spines.items():
        spine.set_visible(True)
    
    return fig, ax