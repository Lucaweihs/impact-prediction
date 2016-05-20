import numpy as np
import pandas as pd

def mape_table(models, X, Y, startingYear = 1, name = None):
    numYears = Y.shape[1]
    predYears = range(startingYear, numYears + startingYear)

    mapesTable = np.zeros((len(models), numYears))
    errorsTable = np.zeros((len(models), numYears))
    for i in range(len(models)):
        mapesTable[i, :], errorsTable[i, :] = models[i].mapeAllWithErrors(X, Y)
    mapesDf = pd.DataFrame(data=mapesTable, index=[m.name for m in models], columns=predYears)
    errorsDf = pd.DataFrame(data=errorsTable, index=[m.name for m in models], columns=predYears)
    if name != None:
        mapesDf.to_csv("tables/" + name + ".tsv", sep="\t")
        errorsDf.to_csv("tables/" + "errors-" + name + ".tsv", sep="\t")
    return (mapesDf, errorsDf)

def rsquared_tables(models, X, Y, baseFeature, startingYear = 1, name = None, removeOutliers = False):
    baseValues = X[[baseFeature]].values[:, 0]

    numYears = Y.shape[1]
    predYears = range(startingYear, numYears + startingYear)
    errorsList = []
    flawedErrorsList = []
    indsToRemove = set()
    for i in range(len(models)):
        preds = np.array(models[i].predictAll(X)).T
        errorsPerYear = np.square(preds - Y.values)
        indsToRemove |= set([np.argmax(errorsPerYear[:, i]) for i in range(numYears)])
        errorsList.append(errorsPerYear)
        flawedErrorsList.append(np.square(preds - np.mean(Y.values, axis=0)))

    if removeOutliers:
        indsToRemove = list(indsToRemove)
        removeString = "removed-"
        print "Inds removed: " + str(indsToRemove)
    else:
        indsToRemove = []
        removeString = ""

    baseErrors = np.var(np.delete((Y.values.T - baseValues).T, indsToRemove, axis=0), axis=0)
    baseErrorsInflated = np.var(np.delete(Y.values, indsToRemove, axis=0), axis=0)

    r2Table = np.zeros((len(models), numYears))
    r2InflatedTable = np.zeros((len(models), numYears))
    r2FlawedTable = np.zeros((len(models), numYears))
    for i in range(len(models)):
        errorsPerYear = np.delete(errorsList[i], indsToRemove, axis=0)
        errorMeans = np.mean(errorsPerYear, axis=0)
        r2Table[i, :] = 1.0 - errorMeans / baseErrors
        r2InflatedTable[i, :] = 1.0 - errorMeans / baseErrorsInflated
        r2FlawedTable[i, :] = np.mean(flawedErrorsList[i], axis=0) / baseErrorsInflated

    modelNames = [m.name for m in models]
    r2Df = pd.DataFrame(data=r2Table, index=modelNames, columns=predYears)
    r2InflatedDf = pd.DataFrame(data=r2InflatedTable, index=modelNames, columns=predYears)
    r2FlawedDf = pd.DataFrame(data=r2FlawedTable, index=modelNames, columns=predYears)

    if name is not None:
        r2Df.to_csv("tables/" + removeString + name + ".tsv", sep="\t")
        r2InflatedDf.to_csv("tables/" + "inflated-" + removeString + name + ".tsv", sep="\t")
        r2FlawedDf.to_csv("tables/" + "flawed-" + removeString + name + ".tsv", sep="\t")
    return {"rsquare" : r2Df, "rsquare-inflated" : r2InflatedDf, "rsquare-flawed" : r2FlawedDf}