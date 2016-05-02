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
    plt.title("Mean Absolute Precent Error")
    if fileName != None:
        plt.savefig("plots/" + fileName)
        plt.close()
    else:
        plt.show()
        plt.close()

def plotMAPEPerCount(model, X, y, year, baseFeature, fileName = None):
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
    #sizes.append([40 for k in mapeForValue.keys()])
    #k = 0
    for s in sizes:
        #k = k + 1
        plt.scatter(np.array(mapeForValue.keys()) + 1,
                    [mapeForValue[i] for i in mapeForValue.keys()],
                     s = s)
        plt.xlabel("Number of Citations")
        plt.title("Mean Absolute Percent Error per Starting Citations")
        plt.ylabel("% Error")
        plt.xscale("log")
        if fileName != None:
            #plt.savefig("plots/" + str(k) + "-" + fileName)
            plt.savefig("plots/" + fileName)
            plt.close()
        else:
            plt.show()
            plt.close()
        
def plotAPEScatter(model, X, y, year, baseFeature, fileName = None):
    baseValues = X[[baseFeature]].values[:,0]
    preds = model.predict(X, year)
    allApes = (preds - y) / y

    plt.scatter(baseValues,allApes, s = 5)
    plt.xlabel("Number of Citations")
    plt.title("Mean Percent Error per Citation Count")
    plt.ylabel("% Error")
    plt.xscale("log")
    if fileName != None:
        plt.savefig("plots/" + fileName)
        plt.close()
    else:
        plt.show()
        plt.close()