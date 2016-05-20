import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import seaborn as sns
sns.set_style("whitegrid", {'legend.frameon': True})
from scipy.stats import gaussian_kde

def plotMAPE(mapesDf, errorsDf, name = None, colors = None, markers = None):
    predYears = mapesDf.columns.values
    modelNames = list(mapesDf.index)
    if colors is None:
        colors = cm.rainbow(np.linspace(0, 1, len(modelNames)))
    if markers is None:
        markers = np.repeat("o", mapesDf.shape[0])

    order = np.argsort(mapesDf.values[:,-1])
    offsets = np.linspace(-.18, .18, len(modelNames))
    for i in range(len(modelNames)):
        darkColor = np.copy(colors[i])
        darkColor[0:3] = darkColor[0:3] / 2.0
        s = 80
        markersize = 10
        if markers[i] == "*":
            s = 150
            markersize = 13
        plt.errorbar(predYears + offsets[np.where(order == i)], mapesDf.values[i,:], yerr = 2 * errorsDf.values[i,:],
                     color = colors[i], label = modelNames[i], marker = markers[i],
                     markerfacecolor=darkColor, markeredgecolor="black", markersize = markersize,
                     zorder=1, lw=3)
        plt.scatter(predYears + offsets[np.where(order == i)], mapesDf.values[i, :], color=darkColor,
                    marker=markers[i], s=s, zorder=2, edgecolor=darkColor)

    plt.margins(x = 0.05)
    plt.legend(loc=0, prop={'size': 20}, fancybox=True, framealpha=1.0)
    plt.xlabel("Year", fontsize=20)
    plt.ylabel("Mean % Error", fontsize=20)
    #ymin, ymax = plt.gca().get_ylim()
    #plt.ylim(0, max(ymax, 0.7))
    reformatAxes()
    if name != None:
        plt.savefig("plots/" + name + ".pdf")
    else:
        plt.show()
    plt.close()

def plotMAPEPerCount(model, X, y, year, baseFeature, name = None):
    baseValues = X[[baseFeature]].values[:,0]
    minBaseValue = max(int(np.min(baseValues)), 1)
    maxBaseValue = int(np.max(baseValues))
    mapeForValue = {}
    numObsForValue = {}
    baseRange = range(minBaseValue, maxBaseValue + 1)
    preds = model.predict(X, year)
    for i in baseRange:
        inds = (baseValues == i)
        if np.any(inds):
            mapeForValue[i] = np.mean(np.abs(preds[inds] - y[inds]) / y[inds])
            numObsForValue[i] = inds.sum()
    s = [4720 * numObsForValue[k] / (1.0 * X.shape[0]) for k in mapeForValue.keys()]
    plt.scatter(np.array(mapeForValue.keys()),
                [mapeForValue[i] for i in mapeForValue.keys()],
                 s = s)

    if "citation" in baseFeature.lower():
        xlab = "# Citations in 2005"
        xscale = "log"
    elif "hind" in baseFeature.lower():
        xlab = "H-Index in 2005"
        xscale = "linear"
        plt.xlim(left = 0)
    elif "age" in baseFeature.lower():
        xlab = "Age in 2005"
        xscale = "linear"
        plt.xlim(left=0)

    plt.xlabel(xlab, fontsize=20)
    plt.ylabel("% Error", fontsize=20)
    #plt.ylim(bottom=0)
    plt.xscale(xscale)
    reformatAxes()
    if name != None:
        plt.savefig("plots/" + name + ".pdf")
        plt.close()
    else:
        plt.show()
        plt.close()
        
def plotAPEScatter(model, X, y, year, baseFeature, name=None, heatMap=False):
    baseValues = X[[baseFeature]].values[:,0]
    preds = model.predict(X, year)
    nonZeroInds = np.where(y != 0)
    allApes = (preds - y) / y
    allApes = allApes[nonZeroInds]

    if "citation" in baseFeature.lower():
        xlab = "Citations in 2005"
        xscale = "log"
    elif "hind" in baseFeature.lower():
        xlab = "H-Index in 2005"
        xscale = "linear"
    elif "authorage" in baseFeature.lower():
        xlab = "Length of Career in 2005"
        xscale = "linear"
    elif "paperage" in baseFeature.lower():
        xlab = "Age in 2005"
        xscale = "linear"
    else:
        raise Exception("Invalid base feature.")

    baseValues = baseValues[nonZeroInds]
    if heatMap:
        xyAll = np.vstack([baseValues, allApes])
        uniqueBases = np.unique(baseValues)
        for base in uniqueBases:
            y = xyAll[:, xyAll[0,:] == base][1]
            z = gaussian_kde(y)(y)
            idx = z.argsort()
            y, z = y[idx], z[idx]
            plt.scatter(np.repeat(base, len(y)), y, c=cm.jet(z / np.max(z)), s=20, edgecolor='')
    else:
        plt.scatter(baseValues, allApes, s=20, edgecolor='')
    plt.xlabel(xlab, fontsize=20)
    plt.ylabel("% Error", fontsize=20)
    plt.xscale(xscale)
    reformatAxes()
    if name != None:
        if heatMap:
            plt.savefig("plots/heat-" + name + ".pdf")
        else:
            plt.savefig("plots/" + name + ".pdf")
        plt.close()
    else:
        plt.show()
        plt.close()

def plotRSquared(rsqDf, name = None, colors = None, markers = None):
    predYears = rsqDf.columns.values
    if colors is None:
        colors = cm.rainbow(np.linspace(0, 1, rsqDf.shape[0]))
    if markers is None:
        markers = np.repeat("o", rsqDf.shape[0])
    modelNames = rsqDf.index

    offsets = np.linspace(-.18, .18, len(modelNames))
    order = np.argsort(rsqDf.values[:, -1])
    for i in range(rsqDf.shape[0]):
        darkColor = np.copy(colors[i])
        darkColor[0:3] = darkColor[0:3] / 2.0
        s = 80
        markersize = 10
        if markers[i] == "*":
            s = 150
            markersize = 13
        plt.plot(predYears + offsets[np.where(order == i)], rsqDf.values[i,:], color=colors[i],
                 label=modelNames[i], marker=markers[i], markersize=markersize, markerfacecolor=darkColor,
                 markeredgecolor=darkColor, markeredgewidth=0.5, zorder=1, lw=3)
        plt.scatter(predYears + offsets[np.where(order == i)], rsqDf.values[i,:], color=darkColor,
                    marker=markers[i],
                    s=s, zorder=2, lw=0.5, edgecolor=darkColor)

    plt.margins(x=0.05)
    plt.legend(loc=0, prop={'size':20}, fancybox=True, framealpha=1.0)
    ymin, _ = plt.gca().get_ylim()
    plt.ylim(bottom=min(ymin, 0.0))
    plt.xlabel("Year", fontsize=20)
    plt.ylabel("Past Adjusted $R^2$", fontsize=20)
    reformatAxes()
    if name != None:
        plt.savefig("plots/" + name + ".pdf")
    else:
        plt.show()
    plt.close()

def reformatAxes():
    ax = plt.gca()
    try:
        ax.ticklabel_format(useOffset=False)
    except AttributeError:
        1 # Do nothing
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()