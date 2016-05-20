from models import *
from plotting import *
import math
import os
import cPickle as pickle
import data_manipulation as dm
from error_tables import *
from misc_functions import anyOpen

def dumpPickleWithZip(obj, filePath):
    with anyOpen(filePath, 'w') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def readPickleWithZip(filePath):
    with anyOpen(filePath, 'r') as f:
        obj = pickle.load(f)
    return obj

def runTests(config):
    X = dm.readData(config.featuresPath)
    Y = dm.readData(config.responsesPath)
    Y = Y.select(lambda x: config.measure in x.lower(), axis=1)

    trainInds, validInds, testInds = dm.getTrainValidTestIndsFromConfig(config)
    trainX, validX, testX = dm.getTrainValidTest(X, trainInds, validInds, testInds)
    trainY, validY, testY = dm.getTrainValidTest(Y, trainInds, validInds, testInds)
    pickleSuffix = config.fullSuffix + ".pickle"
    baseFeature = config.baseFeature
    averageFeature = config.averageFeature
    deltaFeature = config.deltaFeature

    protocol = pickle.HIGHEST_PROTOCOL

    print("Training simple model.\n")
    if not os.path.exists("data/simpleLinear-" + pickleSuffix):
        simpleLinear = SimpleLinearModel(trainX, trainY, baseFeature, deltaFeature)
        pickle.dump(simpleLinear, open("data/simpleLinear-" + pickleSuffix, "wb"), protocol)
    else:
        simpleLinear = pickle.load(open("data/simpleLinear-" + pickleSuffix, "rb"))

    print("Training lasso model.\n")
    if not os.path.exists("data/lasso-" + pickleSuffix):
        lasso = LassoModel(trainX, trainY, baseFeature)
        pickle.dump(lasso, open("data/lasso-" + pickleSuffix, "wb"), protocol)
    else:
        lasso = pickle.load(open("data/lasso-" + pickleSuffix, "rb"))

    print("Training random forest model.\n")
    rfPath = "data/rf-" + pickleSuffix + ".bz2"
    if not os.path.exists(rfPath):
        rf = RandomForestModel(trainX, trainY, baseFeature)
        dumpPickleWithZip(rf, rfPath)
    else:
        rf = readPickleWithZip(rfPath)

    print("Training gradient boost model.\n")
    if not os.path.exists("data/gb-" + pickleSuffix):
        gb = GradientBoostModel(trainX, trainY, baseFeature)
        pickle.dump(gb, open("data/gb-" + pickleSuffix, "wb"), protocol)
    else:
        gb = pickle.load(open("data/gb-" + pickleSuffix, "rb"))

    print("Training constant model.\n")
    constant = ConstantModel(trainX, trainY, baseFeature)

    deltaModels = [constant, simpleLinear]
    mlModels = [lasso, gb, rf]

    if config.docType == "paper":
        print("Training RPPNet models.\n")
        rppWith = RPPStub(config, trainX, validX, testX)
        rppWithout = RPPStub(config, trainX, validX, testX, False)
        mlModels.append(rppWith)
        deltaModels.append(rppWithout)

    colors = cm.rainbow(np.linspace(0, 1, 12))
    markers = np.array(["o", "v", "s", "*", "h", "D", "^"])
    deltaColors = list(colors[[0, 11, 1]][range(len(deltaModels))])
    deltaMarkers = list(markers[[0, 1, 2]][range(len(deltaModels))])
    mlColors = list(colors[[10, 2, 9, 3]][range(len(mlModels))])
    mlMarkers = list(markers[[3, 4, 5, 6]][range(len(mlModels))])

    predStartYear = config.sourceYear + 1

    mapeTestName = "mape-test-delta-" + config.fullSuffix
    mapesDf, errorsDf = mape_table(deltaModels, testX, testY, predStartYear, mapeTestName)
    plotMAPE(mapesDf, errorsDf, mapeTestName, colors=deltaColors, markers=deltaMarkers)

    bestDeltaInd = np.argmin(mapesDf.values[:,-1])
    mlPlus = mlModels[:] + [deltaModels[bestDeltaInd]]
    mlPlusColors = mlColors + [deltaColors[bestDeltaInd]]
    mlPlusMarkers = mlMarkers + [deltaMarkers[bestDeltaInd]]
    mapeTestName = "mape-test-mlplus-" + config.fullSuffix
    mapesDf, errorsDf = mape_table(mlPlus, testX, testY, predStartYear, mapeTestName)
    plotMAPE(mapesDf, errorsDf, mapeTestName, colors=mlPlusColors, markers=mlPlusMarkers)

    mapeTrainName = "mape-train-" + config.fullSuffix
    mapesDf, errorsDf = mape_table(deltaModels + mlModels, trainX, trainY, predStartYear, mapeTrainName)
    plotMAPE(mapesDf, errorsDf, mapeTrainName, colors=deltaColors + mlColors,
             markers=deltaMarkers + mlMarkers)


    rsqTestName = "rsq-test-delta-" + config.fullSuffix
    deltaRsqDfMap = rsquared_tables(deltaModels, testX, testY, baseFeature, predStartYear, rsqTestName)
    plotRSquared(deltaRsqDfMap["rsquare"], rsqTestName, deltaColors, deltaMarkers)

    bestDeltaInd = np.argmax(deltaRsqDfMap["rsquare"].values[:, -1])
    mlPlus = mlModels[:] + [deltaModels[bestDeltaInd]]
    mlPlusColors = mlColors + [deltaColors[bestDeltaInd]]
    mlPlusMarkers = mlMarkers + [deltaMarkers[bestDeltaInd]]
    rsqTestName = "rsq-test-mlplus-" + config.fullSuffix
    rsqDfMap = rsquared_tables(mlPlus, testX, testY, baseFeature, predStartYear, rsqTestName)
    plotRSquared(rsqDfMap["rsquare"], rsqTestName, mlPlusColors, mlPlusMarkers)

    rsqTrainName = "rsq-train-" + config.fullSuffix
    rsqDfMap = rsquared_tables(deltaModels + mlModels, trainX, trainY, baseFeature, predStartYear, rsqTrainName)
    plotRSquared(rsqDfMap["rsquare"], rsqTrainName, colors=deltaColors + mlColors,
             markers=deltaMarkers + mlMarkers)

    year = Y.shape[1]

    apeScatterFileName = "ape-" + config.fullSuffix
    plotAPEScatter(rf, testX, testY.values[:, year - 1], year, config.ageFeature, apeScatterFileName, heatMap=False)
    plotAPEScatter(rf, testX, testY.values[:, year - 1], year, config.ageFeature, apeScatterFileName, heatMap=True)

    mapePlotFileName = "mapePerCountGB-" + config.fullSuffix
    plotMAPEPerCount(gb, testX, testY.values[:, year - 1], year, baseFeature, mapePlotFileName)

    mapePlotFileName = "mapePerAgeGB-" + config.fullSuffix
    plotMAPEPerCount(gb, testX, testY.values[:, year - 1], year, config.ageFeature, mapePlotFileName)

    if config.docType == "paper":
        mapePlotFileName = "mapePerAgeRPP-" + config.fullSuffix
        plotMAPEPerCount(rppWith, testX, testY.values[:, year - 1], year, config.ageFeature, mapePlotFileName)