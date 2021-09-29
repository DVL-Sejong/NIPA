import matplotlib.ticker as mticker
import matplotlib.pyplot as plt


def plot_multiple_graph(data, names=None, xlabel=None, ylabel=None, fig_path=None, n_cols=2, xticks=20, figsize=(15, 60)):
    regions = data[0].index.to_list()
    dates = data[0].columns.tolist()

    n_rows = len(regions) // n_cols
    if len(regions) % n_cols != 0:
        n_rows += 1

    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize, constrained_layout=True)
    locator = mticker.MultipleLocator(xticks)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    for i, region in enumerate(regions):
        for j, elem in enumerate(data):
            value = elem.loc[region].to_list()
            axs[i // n_cols, i % n_cols].plot(dates, value, colors[j], label=names[j])

        axs[i // n_cols, i % n_cols].legend(loc='upper left')

    for i, ax in enumerate(axs.flat):
        if i >= len(regions): break

        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.xaxis.set_major_locator(locator)
        ax.title.set_text(regions[i])

    if fig_path is not None:
        fig.savefig(fig_path)
        print(f'saving plotted graph to {fig_path}')

    plt.show()
    return fig, axs
