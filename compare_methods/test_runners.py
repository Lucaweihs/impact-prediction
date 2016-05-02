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

    protocol = pickle.HIGHEST_PROTOCOL
    if not os.path.exists("data/plusVariableK" + pickleSuffix):
        plusVariableK = PlusVariableKBaselineModel(trainX, trainY, baseFeature, averageFeature)
        pickle.dump(plusVariableK, open("data/plusVariableK" + pickleSuffix, "wb"), protocol)
    else:
        plusVariableK = pickle.load(open("data/plusVariableK" + pickleSuffix, "rb"))

    if not os.path.exists("data/plusFixedK" + pickleSuffix):
        plusFixedK = PlusFixedKBaselineModel(trainX, trainY, baseFeature)
        pickle.dump(plusFixedK, open("data/plusFixedK" + pickleSuffix, "wb"), protocol)
    else:
        plusFixedK = pickle.load(open("data/plusFixedK" + pickleSuffix, "rb"))
        
    # if not os.path.exists("data/simpleLinear" + pickleSuffix):
    #     simpleLinear = SimpleLinearModel(trainX, trainY, baseFeature)
    #     pickle.dump(simpleLinear, open("data/simpleLinear" + pickleSuffix, "wb"), protocol)
    # else:
    #     simpleLinear = pickle.load(open("data/simpleLinear" + pickleSuffix, "rb"))
        
    if not os.path.exists("data/lasso" + pickleSuffix):
        lasso = LassoModel(trainX, trainY, baseFeature)
        pickle.dump(lasso, open("data/lasso" + pickleSuffix, "wb"), protocol)
    else:
        lasso = pickle.load(open("data/lasso" + pickleSuffix, "rb"))
        
    if not os.path.exists("data/rf" + pickleSuffix):
        rf = RandomForestModel(trainX, trainY, baseFeature)
        pickle.dump(rf, open("data/rf" + pickleSuffix, "wb"), protocol)
    else:
        rf = pickle.load(open("data/rf" + pickleSuffix, "rb"))

    constant = ConstantModel(trainX, trainY, baseFeature)

    models = [constant, plusVariableK, lasso, rf]

    if config.docType == "paper":
        rppWith = RPPStub(config)
        #rppWithout = RPPStub(config, False)
        models.append(rppWith)
        #models.append(rppWithout)
    
    mapePlotFileName = "mape" + pickleSuffix.split(".")[0] + ".pdf"
    plotMAPE(models, validX, validY, mapePlotFileName)

    year = Y.shape[1]

    apeScatterFileName = "ape" + pickleSuffix.split(".")[0] + ".pdf"
    plotAPEScatter(rf, validX, validY.values[:, year - 1], year, baseFeature, apeScatterFileName)

    mapePlotFileName = "mapePerCountRf" + pickleSuffix.split(".")[0] + ".pdf"
    plotMAPEPerCount(rf, validX, validY.values[:, year - 1], year, baseFeature, mapePlotFileName)
    # if config.docType == "paper":
    #     mapePlotFileName = "mapePerCountRppWith" + pickleSuffix.split(".")[0] + ".pdf"
    #     plotMAPEPerCount(rppWith, validX, validY.values[:, year - 1], year,
    #                      baseFeature, mapePlotFileName)
    #     mapePlotFileName = "mapePerCountRppWithout" + pickleSuffix.split(".")[0] + ".pdf"
    #     plotMAPEPerCount(rppWithout, validX, validY.values[:, year - 1], year,
    #                      baseFeature, mapePlotFileName)

