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
    # print("Training variable k model.\n")
    # if not os.path.exists("data/plusVariableK" + pickleSuffix):
    #     plusVariableK = PlusVariableKBaselineModel(trainX, trainY, baseFeature, averageFeature)
    #     pickle.dump(plusVariableK, open("data/plusVariableK" + pickleSuffix, "wb"), protocol)
    # else:
    #     plusVariableK = pickle.load(open("data/plusVariableK" + pickleSuffix, "rb"))
    #
    # print("Training fixed k model.\n")
    # if not os.path.exists("data/plusFixedK" + pickleSuffix):
    #     plusFixedK = PlusFixedKBaselineModel(trainX, trainY, baseFeature)
    #     pickle.dump(plusFixedK, open("data/plusFixedK" + pickleSuffix, "wb"), protocol)
    # else:
    #     plusFixedK = pickle.load(open("data/plusFixedK" + pickleSuffix, "rb"))

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

    print("Training xgboost model.\n")
    if not os.path.exists("data/xgBoost" + pickleSuffix):
        xgBoost = XGBoostModel(trainX, trainY, baseFeature, tuneWithCV=False)
        pickle.dump(xgBoost, open("data/xgBoost" + pickleSuffix, "wb"), protocol)
    else:
        xgBoost = pickle.load(open("data/xgBoost" + pickleSuffix, "rb"))

    print("Training constant model.\n")
    constant = ConstantModel(trainX, trainY, baseFeature)

    models = [constant, xgBoost, gb, simpleLinear, lasso, rf]

    if config.docType == "paper":
        print("Training RPPNet models.\n")
        rppWith = RPPStub(config, trainX, validX)
        rppWithout = RPPStub(config, trainX, validX, False)
        models.append(rppWith)
        models.append(rppWithout)

    mapeValidPlotFileName = "mape-valid" + pickleSuffix.split(".")[0] + ".pdf"
    plotMAPE(models, validX, validY, mapeValidPlotFileName)

    mapeTrainPlotFileName = "mape-train" + pickleSuffix.split(".")[0] + ".pdf"
    plotMAPE(models, trainX, trainY, mapeTrainPlotFileName)

    year = Y.shape[1]

    apeScatterFileName = "ape" + pickleSuffix.split(".")[0] + ".pdf"
    plotAPEScatter(rf, validX, validY.values[:, year - 1], year, baseFeature, apeScatterFileName)

    mapePlotFileName = "mapePerCountXGB" + pickleSuffix.split(".")[0] + ".pdf"
    plotMAPEPerCount(rf, validX, validY.values[:, year - 1], year, baseFeature, mapePlotFileName)

    perAgeTitle = "Mean Absolute Percent Error per Paper Age"
    perAgeXLabel = "Paper Age"
    mapePlotFileName = "mapePerAgeXGB" + pickleSuffix.split(".")[0] + ".pdf"
    plotMAPEPerCount(xgBoost, validX, validY.values[:, year - 1], year, config.ageFeature, mapePlotFileName,
                     title=perAgeTitle, xlabel=perAgeXLabel)

    if config.docType == "paper":
        mapePlotFileName = "mapePerAgeRPP" + pickleSuffix.split(".")[0] + ".pdf"
        plotMAPEPerCount(rppWith, validX, validY.values[:, year - 1], year, config.ageFeature, mapePlotFileName,
                         title=perAgeTitle, xlabel=perAgeXLabel)

    resPlotFileName = "rsq-valid" + pickleSuffix.split(".")[0] + ".pdf"
    plotResError(models, validX, validY, baseFeature, resPlotFileName)

    resPlotFileName = "rsq-train" + pickleSuffix.split(".")[0] + ".pdf"
    plotResError(models, trainX, trainY, baseFeature, resPlotFileName)
    # if config.docType == "paper":
    #     mapePlotFileName = "mapePerCountRppWith" + pickleSuffix.split(".")[0] + ".pdf"
    #     plotMAPEPerCount(rppWith, validX, validY.values[:, year - 1], year,
    #                      baseFeature, mapePlotFileName)
    #     mapePlotFileName = "mapePerCountRppWithout" + pickleSuffix.split(".")[0] + ".pdf"
    #     plotMAPEPerCount(rppWithout, validX, validY.values[:, year - 1], year,
    #                      baseFeature, mapePlotFileName)

