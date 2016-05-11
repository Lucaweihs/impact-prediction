import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style("whitegrid", {'legend.frameon': True})

def plotMAPE(models, X, Y, name = None, startingYear = 1):
    numYears = Y.shape[1]
    colors = cm.rainbow(np.linspace(0, 1, len(models)))
    predYears = range(startingYear, numYears + startingYear)

    mapesTable = np.zeros((len(models), numYears))
    for i in range(len(models)):
        mapesTable[i, :], errors = models[i].mapeAllWithErrors(X, Y)
        plt.errorbar(predYears, mapesTable[i,:], yerr = 2 * errors, color = colors[i],
                 label = models[i].name, marker = 'o', markeredgecolor='black',
                     markeredgewidth=0.5)
    mapesDf = pd.DataFrame(data=mapesTable, index=[m.name for m in models], columns=range(1, numYears + 1))
    plt.margins(x = 0.05)
    plt.legend(loc=0, prop={'size': 20})
    plt.xlabel("Year", fontsize=20)
    plt.ylabel("Mean % Error", fontsize=20)
    #plt.title("Mean Absolute Percent Error")
    reformatAxes()
    if name != None:
        plt.savefig("plots/" + name + ".pdf")
        plt.close()
        mapesDf.to_csv("tables/" + name + ".tsv", sep = "\t")
    else:
        plt.show()
        plt.close()
    return mapesDf

def plotMAPEPerCount(model, X, y, year, baseFeature, name = None):
    baseValues = X[[baseFeature]].values[:,0]
    minBaseValue = int(np.min(baseValues))
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
    allApes = (preds - y) / y

    if "citation" in baseFeature.lower():
        xlab = "Starting # Citations"
        xscale = "log"
    else:
        xlab = "Starting H-Index"
        xscale = "linear"

    plt.scatter(baseValues, allApes, s=20)
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

def plotResError(models, X, Y, baseFeature, name = None, removeOutliers = False,
                 title = "",
                 startYear = 1):
    models = [m for m in models if m.name.lower() != "constant"]
    baseValues = X[[baseFeature]].values[:, 0]

    numYears = Y.shape[1]
    colors = cm.rainbow(np.linspace(0, 1, len(models)))
    predYears = range(startYear, numYears + startYear)
    errorsList = []
    flawedErrorsList = []
    indsToRemove = set()
    for i in range(len(models)):
        preds = np.array(models[i].predictAll(X))
        errorsPerYear = np.square(preds.T - Y.values)
        indsToRemove |= set([np.argmax(errorsPerYear[:,i]) for i in range(numYears)])
        errorsList.append(errorsPerYear)
        flawedErrorsList.append(np.square(preds.T - np.mean(Y.values, axis=0)))

    if removeOutliers:
        indsToRemove = list(indsToRemove)
        print "Inds removed: " + str(indsToRemove)
    else:
        indsToRemove = []

    baseErrors = np.var(np.delete((Y.values.T - baseValues).T, indsToRemove, axis=0), axis=0)
    baseErrorsInflated = np.var(np.delete(Y.values, indsToRemove, axis=0), axis=0)

    r2Table = np.zeros((len(models), numYears))
    r2InflatedTable = np.zeros((len(models), numYears))
    r2FlawedTable = np.zeros((len(models), numYears))
    for i in range(len(models)):
        errors = np.delete(errorsList[i], indsToRemove, axis=0)
        errorMeans = np.mean(errors, axis=0)
        r2Table[i,:] = 1.0 - errorMeans / baseErrors
        r2InflatedTable[i,:] = 1.0 - errorMeans / baseErrorsInflated
        r2FlawedTable[i,:] = np.mean(flawedErrorsList[i], axis=0) / baseErrorsInflated
        plt.plot(predYears, r2Table[i,:], color=colors[i], label=models[i].name, marker='o',
                 markeredgecolor='black', markeredgewidth=0.5)

    r2Df = pd.DataFrame(data=r2Table, index=[m.name for m in models], columns=range(1, numYears + 1))
    r2InflatedDf = pd.DataFrame(data=r2InflatedTable, index=[m.name for m in models], columns=range(1, numYears + 1))
    r2FlawedDf = pd.DataFrame(data=r2FlawedTable, index=[m.name for m in models], columns=range(1, numYears + 1))
    plt.margins(x=0.05)
    plt.legend(loc=0, prop={'size':20})
    plt.ylim(top = 1 + .05)
    plt.xlabel("Year", fontsize=20)
    plt.ylabel("% Variation Explained ($R^2$)", fontsize=20)
    plt.title(title)
    reformatAxes()
    if name != None:
        r2Df.to_csv("tables/" + name + ".tsv", sep="\t")
        r2InflatedDf.to_csv("tables/" + "inflated-" + name + ".tsv", sep="\t")
        r2FlawedDf.to_csv("tables/" + "flawed-" + name + ".tsv", sep="\t")
        plt.savefig("plots/" + name + ".pdf")
        plt.close()
    else:
        plt.show()
        plt.close()
    return (r2Df, r2InflatedDf)

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