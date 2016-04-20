import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def plotMAPE(models, X, Y, fileName = None):
    numYears = Y.shape[1]
    colors = cm.rainbow(np.linspace(0, 1, numYears))
    predYears = range(1, numYears + 1)
    for i in range(len(models)):
        plt.plot(predYears, models[i].mapeAll(X, Y), color = colors[i],
                 label = models[i].name, marker = 'o')
    #plt.xlim(-.1, np.max(tau) + .1)
    plt.margins(x = 0.05)
    plt.legend(loc = 2)
    plt.xlabel("Years Out")
    plt.ylabel("MAPE")
    plt.title("Mean Absolute Relative Error")
    if fileName != None:
        plt.savefig(fileName)
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
    for i in baseRange:
        inds = (baseValues == i)
        if np.any(inds):
            mapeForValue[i] = model.mape(X.loc[inds], y[inds], year)
            numObsForValue[i] = inds.sum()
    sizes = []
    sizes.append([4720 * numObsForValue[k] / (1.0 * X.shape[0]) for k in mapeForValue.keys()])
    sizes.append([40 for k in mapeForValue.keys()])
    k = 0
    for s in sizes:
        k = k + 1
        plt.scatter(np.array(mapeForValue.keys()) + 1,
                    [mapeForValue[i] for i in mapeForValue.keys()],
                     s = s)
        plt.xlabel("Number of Citations")
        plt.title("Mean Absolute Relative Error per Starting Citations")
        plt.ylabel("MAPE")
        plt.xscale("log")
        if fileName != None:
            plt.savefig(str(k) + "-" + fileName)
            plt.close()
        else:
            plt.show()
            plt.close()
        
    