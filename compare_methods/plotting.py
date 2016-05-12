import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import seaborn as sns
sns.set_style("whitegrid", {'legend.frameon': True})

def plotMAPE(mapesDf, errorsDf, name = None):
    predYears = mapesDf.columns.values
    modelNames = list(mapesDf.index)
    colors = cm.rainbow(np.linspace(0, 1, len(modelNames)))

    for i in range(len(modelNames)):
        plt.errorbar(predYears, mapesDf.values[i,:], yerr = 2 * errorsDf.values[i,:], color = colors[i],
                 label = modelNames[i], marker = 'o', markeredgecolor='black', markeredgewidth=0.5)
    plt.margins(x = 0.05)
    plt.legend(loc=0, prop={'size': 20})
    plt.xlabel("Year", fontsize=20)
    plt.ylabel("Mean % Error", fontsize=20)
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
        xlab = "Starting # Citations"
        xscale = "log"
    else:
        xlab = "Starting H-Index"
        xscale = "linear"

    plt.xlabel(xlab, fontsize=20)
    plt.ylabel("% Error", fontsize=20)
    plt.ylim(bottom=0)
    plt.xscale(xscale)
    reformatAxes()
    if name != None:
        plt.savefig("plots/" + name + ".pdf")
        plt.close()
    else:
        plt.show()
        plt.close()
        
def plotAPEScatter(model, X, y, year, baseFeature, name = None):
    baseValues = X[[baseFeature]].values[:,0]
    preds = model.predict(X, year)
    nonZeroInds = np.where(y != 0)
    allApes = (preds - y) / y
    allApes = allApes[nonZeroInds]

    if "citation" in baseFeature.lower():
        xlab = "Starting # Citations"
        xscale = "log"
    else:
        xlab = "Starting H-Index"
        xscale = "linear"

    plt.scatter(baseValues[nonZeroInds], allApes, s=20)
    plt.xlabel(xlab, fontsize=20)
    plt.ylabel("% Error", fontsize=20)
    plt.xscale(xscale)
    reformatAxes()
    if name != None:
        plt.savefig("plots/" + name + ".pdf")
        plt.close()
    else:
        plt.show()
        plt.close()

def plotRSquared(rsqDf, name = None):
    rsqDf = rsqDf[rsqDf.index != "Constant"]
    predYears = rsqDf.columns.values
    colors = cm.rainbow(np.linspace(0, 1, rsqDf.shape[0]))
    modelNames = rsqDf.index

    for i in range(rsqDf.shape[0]):
        plt.plot(predYears, rsqDf.values[i,:], color=colors[i], label=modelNames[i], marker='o',
                 markeredgecolor='black', markeredgewidth=0.5)

    plt.margins(x=0.05)
    plt.legend(loc=0, prop={'size':20})
    plt.ylim(top = 1 + .05)
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
        1
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()