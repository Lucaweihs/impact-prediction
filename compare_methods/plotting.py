import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def plotMAPE(models, X, Y, fileName = None):
    numYears = Y.shape[1]
    colors = cm.rainbow(np.linspace(0, 1, len(models)))
    predYears = range(1, numYears + 1)
    for i in range(len(models)):
        mapes, errors = models[i].mapeAllWithErrors(X, Y)
        plt.errorbar(predYears, mapes, yerr = 2 * errors, color = colors[i],
                 label = models[i].name, marker = 'o')
    plt.margins(x = 0.05)
    plt.legend(loc = 2)
    plt.xlabel("Years Out")
    plt.ylabel("% Error")
    plt.title("Mean Absolute Percent Error")
    if fileName != None:
        plt.savefig("plots/" + fileName)
        plt.close()
    else:
        plt.show()
        plt.close()

def plotMAPEPerCount(model, X, y, year, baseFeature, fileName = None,
                     title="Mean Absolute Percent Error per Starting Citations",
                     xlabel="Number of Citations"):
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
    sizes = []
    sizes.append([4720 * numObsForValue[k] / (1.0 * X.shape[0]) for k in mapeForValue.keys()])
    for s in sizes:
        plt.scatter(np.array(mapeForValue.keys()) + 1,
                    [mapeForValue[i] for i in mapeForValue.keys()],
                     s = s)
        plt.xlabel(xlabel)
        plt.title(title)
        plt.ylabel("% Error")
        plt.xscale("log")
        if fileName != None:
            plt.savefig("plots/" + fileName)
            plt.close()
        else:
            plt.show()
            plt.close()
        
def plotAPEScatter(model, X, y, year, baseFeature, fileName = None):
    baseValues = X[[baseFeature]].values[:,0]
    preds = model.predict(X, year)
    allApes = (preds - y) / y

    if "paper" in fileName.lower():
        type = "Paper"
    else:
        type = "Author"

    plt.scatter(baseValues,allApes, s = 5)
    plt.xlabel("Number of Citations")
    plt.title("Percent Error per " + type)
    plt.ylabel("% Error")
    plt.xscale("log")
    if fileName != None:
        plt.savefig("plots/" + fileName)
        plt.close()
    else:
        plt.show()
        plt.close()

def plotResError(models, X, Y, baseFeature, fileName = None):
    baseValues = X[[baseFeature]].values[:, 0]
    baseErrors = np.var((Y.values.T - baseValues).T, axis = 0)

    numYears = Y.shape[1]
    colors = cm.rainbow(np.linspace(0, 1, len(models)))
    predYears = range(1, numYears + 1)
    for i in range(len(models)):
        preds = models[i].predictAll(X)
        errors = np.mean(np.square(np.array(preds).T - Y.values), axis = 0)
        relErrors = errors /  baseErrors
        plt.plot(predYears, relErrors, color=colors[i],
                 label=models[i].name, marker='o')
    plt.margins(x=0.05)
    plt.legend(loc=2)
    plt.xlabel("Years Out")
    plt.ylabel("Residual Squared Error")
    plt.title("Squared error")
    if fileName != None:
        plt.savefig("plots/" + fileName)
        plt.close()
    else:
        plt.show()
        plt.close()

