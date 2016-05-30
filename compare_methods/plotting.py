import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import seaborn as sns
sns.set_style("whitegrid", {'legend.frameon': True})
from scipy.stats import gaussian_kde

def plot_mape(mapes_df, errors_df, name = None, colors = None, markers = None):
    pred_years = mapes_df.columns.values
    model_names = list(mapes_df.index)
    if colors is None:
        colors = cm.rainbow(np.linspace(0, 1, len(model_names)))
    if markers is None:
        markers = np.repeat("o", mapes_df.shape[0])

    order = np.argsort(mapes_df.values[:,-1])
    #offsets = np.linspace(-.18, .18, len(model_names))
    offsets = np.zeros(len(model_names))
    for i in range(len(model_names)):
        dark_color = np.copy(colors[i])
        dark_color[0:3] = dark_color[0:3] / 2.0
        s = 80
        markersize = 10
        if markers[i] == "*":
            s = 150
            markersize = 13
        plt.errorbar(pred_years + offsets[np.where(order == i)], mapes_df.values[i,:], yerr = 2 * errors_df.values[i,:],
                     color = colors[i], label = model_names[i], marker = markers[i],
                     markerfacecolor=dark_color, markeredgecolor="black", markersize = markersize,
                     zorder=1, lw=3)
        plt.scatter(pred_years + offsets[np.where(order == i)], mapes_df.values[i, :], color=dark_color,
                    marker=markers[i], s=s, zorder=2, edgecolor=dark_color)

    plt.margins(x = 0.05)

    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, labels, loc=0, prop={'size': 20}, fancybox=True, framealpha=1.0)
    plt.xlabel("Year", fontsize=20)
    plt.ylabel("Mean % Error", fontsize=20)
    #ymin, ymax = plt.gca().get_ylim()
    #plt.ylim(0, max(ymax, 0.7))
    reformat_axes()
    if name != None:
        plt.savefig("plots/" + name + ".pdf")
    else:
        plt.show()
    plt.close()

def plot_mape_per_count(model, X, y, year, base_feature, name = None):
    base_values = X[[base_feature]].values[:,0]
    min_base_value = max(int(np.min(base_values)), 1)
    max_base_value = int(np.max(base_values))
    mape_for_value = {}
    num_obs_for_value = {}
    base_range = range(min_base_value, max_base_value + 1)
    preds = model.predict(X, year)
    for i in base_range:
        inds = (base_values == i)
        if np.any(inds):
            mape_for_value[i] = np.mean(np.abs(preds[inds] - y[inds]) / y[inds])
            num_obs_for_value[i] = inds.sum()
    s = [4720 * num_obs_for_value[k] / (1.0 * X.shape[0]) for k in mape_for_value.keys()]
    plt.scatter(np.array(mape_for_value.keys()),
                [mape_for_value[i] for i in mape_for_value.keys()],
                 s = s)

    if "citation" in base_feature.lower():
        xlab = "# Citations in 2005"
        xscale = "log"
    elif "hindex" in base_feature.lower():
        xlab = "H-Index in 2005"
        xscale = "linear"
        plt.xlim(left = 0)
    elif "age" in base_feature.lower():
        xlab = "Age in 2005"
        xscale = "linear"
        plt.xlim(left=0)

    plt.xlabel(xlab, fontsize=20)
    plt.ylabel("% Error", fontsize=20)
    #plt.ylim(bottom=0)
    plt.xscale(xscale)
    reformat_axes()
    if name != None:
        plt.savefig("plots/" + name + ".pdf")
        plt.close()
    else:
        plt.show()
        plt.close()

def plot_ape_scatter(model, X, y, year, base_feature, name=None, heat_map=False):
    base_values = X[[base_feature]].values[:,0]
    preds = model.predict(X, year)
    non_zero_inds = np.where(y != 0)
    all_apes = (preds - y) / y
    all_apes = all_apes[non_zero_inds]

    if "citation" in base_feature.lower():
        xlab = "Citations in 2005"
        xscale = "log"
    elif "hindex" in base_feature.lower():
        xlab = "H-Index in 2005"
        xscale = "linear"
    elif "author_age" in base_feature.lower():
        xlab = "Length of Author Career in 2005"
        xscale = "linear"
    elif "paper_age" in base_feature.lower():
        xlab = "Paper Age in 2005"
        xscale = "linear"
    else:
        raise Exception("Invalid base feature.")

    base_values = base_values[non_zero_inds]
    if heat_map:
        xy_all = np.vstack([base_values, all_apes])
        unique_bases = np.unique(base_values)
        for base in unique_bases:
            y = xy_all[:, xy_all[0,:] == base][1]
            z = gaussian_kde(y)(y)
            idx = z.argsort()
            y, z = y[idx], z[idx]
            plt.scatter(np.repeat(base, len(y)), y, c=cm.jet(z / np.max(z)), s=20, edgecolor='')
    else:
        plt.scatter(base_values, all_apes, s=20, edgecolor='')
    plt.xlabel(xlab, fontsize=20)
    plt.ylabel("% Error after 10 Years", fontsize=20)
    plt.xscale(xscale)
    plt.xlim(left=0)
    plt.ylim(bottom=-1.5)
    reformat_axes()
    if name != None:
        if heat_map:
            fig = plt.gcf()
            fig.set_size_inches(12, 4)
            reformat_axes()
            plt.savefig("plots/heat-" + name + ".pdf")
        else:
            plt.savefig("plots/" + name + ".pdf")
        plt.close()
    else:
        plt.show()
        plt.close()

def plot_r_squared(rsq_df, name = None, colors = None, markers = None,
                 xlabel="Past Adjusted $R^2$"):
    pred_years = rsq_df.columns.values
    if colors is None:
        colors = cm.rainbow(np.linspace(0, 1, rsq_df.shape[0]))
    if markers is None:
        markers = np.repeat("o", rsq_df.shape[0])
    model_names = rsq_df.index

    #offsets = np.linspace(-.18, .18, len(model_names))
    offsets = np.zeros(len(model_names))
    order = np.argsort(rsq_df.values[:, -1])
    for i in range(rsq_df.shape[0]):
        dark_color = np.copy(colors[i])
        dark_color[0:3] = dark_color[0:3] / 2.0
        s = 80
        markersize = 10
        if markers[i] == "*":
            s = 150
            markersize = 13
        plt.plot(pred_years + offsets[np.where(order == i)], rsq_df.values[i,:], color=colors[i],
                 label=model_names[i], marker=markers[i], markersize=markersize, markerfacecolor=dark_color,
                 markeredgecolor=dark_color, markeredgewidth=0.5, zorder=1, lw=3)
        plt.scatter(pred_years + offsets[np.where(order == i)], rsq_df.values[i,:], color=dark_color,
                    marker=markers[i],
                    s=s, zorder=2, lw=0.5, edgecolor=dark_color)

    plt.margins(x=0.05)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, labels, loc=0, prop={'size':20}, fancybox=True, framealpha=1.0)
    ymin, ymax = plt.gca().get_ylim()
    plt.ylim(top=min(1.0, ymax))
    plt.xlabel("Year", fontsize=20)
    plt.ylabel(xlabel, fontsize=20)
    reformat_axes()
    if name != None:
        plt.savefig("plots/" + name + ".pdf")
    else:
        plt.show()
    plt.close()

def reformat_axes():
    ax = plt.gca()
    try:
        ax.ticklabel_format(use_offset=False)
    except AttributeError:
        1 # Do nothing
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()