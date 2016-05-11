from models import *
from plotting import *
import math
import os
import cPickle as pickle
import data_manipulation as dm

def runTests(config):
    X = dm.readData(config.featuresPath)
    Y = dm.readData(config.responsesPath)
    Y = Y.select(lambda x: config.measure in x.lower(), axis=1)

    trainInds, validInds, testInds = dm.getTrainValidTestIndsFromConfig(config)
    trainX, validX, _ = dm.getTrainValidTest(X, trainInds, validInds, testInds)
    trainY, validY, _ = dm.getTrainValidTest(Y, trainInds, validInds, testInds)
    pickleSuffix = config.fullSuffix + ".pickle"
    baseFeature = config.baseFeature
    averageFeature = config.averageFeature
    deltaFeature = config.deltaFeature

    protocol = pickle.HIGHEST_PROTOCOL

    if not os.path.exists("data/simpleLinear" + pickleSuffix):
        simpleLinear = SimpleLinearModel(trainX, trainY, baseFeature, deltaFeature)
        pickle.dump(simpleLinear, open("data/simpleLinear" + pickleSuffix, "wb"), protocol)
    else:
        simpleLinear = pickle.load(open("data/simpleLinear" + pickleSuffix, "rb"))

    print("Training lasso model.\n")
    if not os.path.exists("data/lasso" + pickleSuffix):
        lasso = LassoModel(trainX, trainY, baseFeature)
        pickle.dump(lasso, open("data/lasso" + pickleSuffix, "wb"), protocol)
    else:
        lasso = pickle.load(open("data/lasso" + pickleSuffix, "rb"))

    print("Training random forest model.\n")
    if not os.path.exists("data/rf" + pickleSuffix):
        rf = RandomForestModel(trainX, trainY, baseFeature)
        pickle.dump(rf, open("data/rf" + pickleSuffix, "wb"), protocol)
    else:
        rf = pickle.load(open("data/rf" + pickleSuffix, "rb"))

    print("Training gradient boost model.\n")
    if not os.path.exists("data/gb" + pickleSuffix):
        gb = GradientBoostModel(trainX, trainY, baseFeature)
        pickle.dump(gb, open("data/gb" + pickleSuffix, "wb"), protocol)
    else:
        gb = pickle.load(open("data/gb" + pickleSuffix, "rb"))

    print("Training constant model.\n")
    constant = ConstantModel(trainX, trainY, baseFeature)

    models = [constant, gb, simpleLinear, lasso, rf]

    if config.docType == "paper":
        print("Training RPPNet models.\n")
        rppWith = RPPStub(config, trainX, validX)
        rppWithout = RPPStub(config, trainX, validX, False)
        models.append(rppWith)
        models.append(rppWithout)

    mapeValidPlotFileName = "mape-valid" + pickleSuffix.split(".")[0]
    plotMAPE(models, validX, validY, mapeValidPlotFileName,
             startingYear = config.sourceYear + 1)

    mapeTrainPlotFileName = "mape-train" + pickleSuffix.split(".")[0]
    plotMAPE(models, trainX, trainY, mapeTrainPlotFileName,
             startingYear = config.sourceYear + 1)

    year = Y.shape[1]

    apeScatterFileName = "ape" + pickleSuffix.split(".")[0]
    plotAPEScatter(rf, validX, validY.values[:, year - 1], year, baseFeature, apeScatterFileName)

    mapePlotFileName = "mapePerCountXGB" + pickleSuffix.split(".")[0]
    plotMAPEPerCount(rf, validX, validY.values[:, year - 1], year, baseFeature, mapePlotFileName)

    perAgeTitle = "Mean Absolute Percent Error per Paper Age"
    perAgeXLabel = "Paper Age"
    mapePlotFileName = "mapePerAgeXGB" + pickleSuffix.split(".")[0]
    #plotMAPEPerCount(xgBoost, validX, validY.values[:, year - 1], year, config.ageFeature, mapePlotFileName,
    #                 title=perAgeTitle, xlabel=perAgeXLabel)

    if config.docType == "paper":
        mapePlotFileName = "mapePerAgeRPP" + pickleSuffix.split(".")[0]
        plotMAPEPerCount(rppWith, validX, validY.values[:, year - 1], year, config.ageFeature, mapePlotFileName,
                         title=perAgeTitle, xlab=perAgeXLabel)

    resPlotFileName = "rsq-valid" + pickleSuffix.split(".")[0]
    plotResError(models, validX, validY, baseFeature, resPlotFileName,
                 startYear = config.sourceYear + 1)

    resPlotFileName = "rsq-train" + pickleSuffix.split(".")[0]
    plotResError(models, trainX, trainY, baseFeature, resPlotFileName,
                 startYear=config.sourceYear + 1)