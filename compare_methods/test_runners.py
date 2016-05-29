from models import *
from plotting import *
import os
import cPickle as pickle
import data_manipulation as dm
from error_tables import *
from misc_functions import *

def runTests(config):
    X = dm.readData(config.featuresPath)
    Y = dm.readData(config.responsesPath)
    Y = Y.select(lambda x: config.measure in x.lower(), axis=1)
    histories = dm.readHistories(config.historyPath)
    for history in histories:
        history[1:] = history[1:] - history[:-1]

    trainInds, validInds, testInds = dm.getTrainValidTestIndsFromConfig(config)
    trainX, validX, testX = dm.getTrainValidTestData(X, trainInds, validInds, testInds)
    trainY, validY, testY = dm.getTrainValidTestData(Y, trainInds, validInds, testInds)
    trainHistories, validHistories, testHistories = \
        dm.getTrainValidTestHistories(histories, trainInds, validInds, testInds)
    pickleSuffix = config.fullSuffix + ".pickle"
    baseFeature = config.baseFeature
    deltaFeature = config.deltaFeature

    protocol = pickle.HIGHEST_PROTOCOL

    print("Training constant model.\n")
    constant = ConstantModel(trainX, trainY, baseFeature)

    print("Training optimal plus fixed k model.\n")
    fixedKOptimal = PlusKBaselineModel(trainX, trainY, baseFeature)

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

    baselineModels = [constant, fixedKOptimal, simpleLinear]
    mlModels = [lasso, rf, gb]

    if config.docType == "paper":
        print("Training RPPNet models.\n")
        rpp_suffix = pickleSuffix.split(".")[0]
        rpp_net = RPPNetWrapper(trainX, trainHistories, trainY, "data/rpp-tf-" + rpp_suffix)
        mlModels.insert(0, rpp_net)
        #rppWith = RPPStub(config, trainX, validX, testX)
        #rppWithout = RPPStub(config, trainX, validX, testX, False)
        #mlModels.insert(0, rppWith)
        #mlModels.insert(0, rppWithout)

    numBaseline = len(baselineModels)
    numMl = len(mlModels)
    np.random.seed(23498)
    colors = cm.rainbow(np.linspace(0, 1, 12))
    markers = np.array(['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd'])
    np.random.shuffle(colors)
    np.random.shuffle(markers)
    baselineColors = list(colors[range(numBaseline)])
    baselineMarkers = list(markers[range(numBaseline)])
    mlColors = list(colors[range(numBaseline, numBaseline + numMl)])
    mlMarkers = list(markers[range(numBaseline, numBaseline + numMl)])

    predStartYear = config.sourceYear + 1
    plotSuffix = config.fullSuffix.replace(":", "_")

    # MAPE tables and plots
    if config.docType == "paper":
        rpp_net.set_prediction_histories(testHistories)
    mapeTestName = "mape-test-ml-" + plotSuffix
    mapesDf, errorsDf = mape_table(mlModels, testX, testY, predStartYear, mapeTestName)
    plotMAPE(mapesDf, errorsDf, mapeTestName, colors=mlColors, markers=mlMarkers)

    if config.docType == "paper":
        rpp_net.set_prediction_histories(validHistories)
    mapeValidName = "mape-valid-ml-" + plotSuffix
    validMapesDf, _ = mape_table(mlModels, validX, validY, predStartYear, mapeValidName)

    if config.docType == "paper":
        rpp_net.set_prediction_histories(testHistories)
    top1Ml = nArgMin(validMapesDf.values[:,-1], 1)
    models = baselineModels + listInds(mlModels, top1Ml)
    colors = baselineColors + listInds(mlColors, top1Ml)
    markers = baselineMarkers + listInds(mlMarkers, top1Ml)
    mapeTestName = "mape-test-baseline-" + plotSuffix
    mapesDf, errorsDf = mape_table(models, testX, testY, predStartYear, mapeTestName)
    plotMAPE(mapesDf, errorsDf, mapeTestName, colors=colors, markers=markers)

    if config.docType == "paper":
        rpp_net.set_prediction_histories(trainHistories)
    mapeTrainName = "mape-train-" + plotSuffix
    mapesDf, errorsDf = mape_table(baselineModels + mlModels, trainX, trainY, predStartYear, mapeTrainName)
    plotMAPE(mapesDf, errorsDf, mapeTrainName, colors=baselineColors + mlColors,
             markers=baselineMarkers + mlMarkers)

    # PA-R^2 tables and plots
    if config.docType == "paper":
        rpp_net.set_prediction_histories(testHistories)
    rsqTestName = "rsq-test-ml-" + plotSuffix
    rsqDfMap = rsquared_tables(mlModels, testX, testY, baseFeature, predStartYear, rsqTestName)
    plotRSquared(rsqDfMap["rsquare"], rsqTestName, mlColors, mlMarkers)
    plotRSquared(rsqDfMap["rsquare-inflated"], "inflated-" + rsqTestName, mlColors, mlMarkers, xlabel="$R^2$")

    if config.docType == "paper":
        rpp_net.set_prediction_histories(validHistories)
    rsqValidName = "rsq-valid-ml-" + plotSuffix
    validRsqDfMap = rsquared_tables(mlModels, validX, validY, baseFeature, predStartYear, rsqValidName)

    if config.docType == "paper":
        rpp_net.set_prediction_histories(testHistories)
    rsqTestName = "rsq-test-baseline-" + plotSuffix
    top1Ml = nArgMax(validRsqDfMap["rsquare"].values[:, -1], 1)
    models = baselineModels + listInds(mlModels, top1Ml)
    colors = baselineColors + listInds(mlColors, top1Ml)
    markers = baselineMarkers + listInds(mlMarkers, top1Ml)
    baselineRsqDfMap = rsquared_tables(models, testX, testY, baseFeature, predStartYear, rsqTestName)
    plotRSquared(baselineRsqDfMap["rsquare"], rsqTestName, colors, markers)
    plotRSquared(baselineRsqDfMap["rsquare-inflated"], "inflated-" + rsqTestName, colors, markers, xlabel="$R^2$")

    if config.docType == "paper":
        rpp_net.set_prediction_histories(trainHistories)
    rsqTrainName = "rsq-train-" + plotSuffix
    rsqDfMap = rsquared_tables(baselineModels + mlModels, trainX, trainY, baseFeature, predStartYear, rsqTrainName)
    plotRSquared(rsqDfMap["rsquare"], rsqTrainName, colors=baselineColors + mlColors,
             markers=baselineMarkers + mlMarkers)

    year = Y.shape[1]

    apeScatterFileName = "ape-" + config.fullSuffix
    plotAPEScatter(gb, testX, testY.values[:, year - 1], year, config.ageFeature, apeScatterFileName, heatMap=False)
    plotAPEScatter(gb, testX, testY.values[:, year - 1], year, config.ageFeature, apeScatterFileName, heatMap=True)

    mapePlotFileName = "mapePerCountGB-" + config.fullSuffix
    plotMAPEPerCount(gb, testX, testY.values[:, year - 1], year, baseFeature, mapePlotFileName)

    mapePlotFileName = "mapePerAgeGB-" + config.fullSuffix
    plotMAPEPerCount(gb, testX, testY.values[:, year - 1], year, config.ageFeature, mapePlotFileName)

    if config.docType == "paper":
        mapePlotFileName = "mapePerAgeRPP-" + config.fullSuffix
        plotMAPEPerCount(rppWith, testX, testY.values[:, year - 1], year, config.ageFeature, mapePlotFileName)