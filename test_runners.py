import numpy as np
from Models import *
from data_manipulation import *
from plotting import *
import math
import os
import cPickle as pickle

def runTests(X, Y, history, docType, dateConfig, baseFeature, averageFeature, citationFeature, minNumCitations):    
    np.random.seed(123)
    indsWithMin = X[[citationFeature]].values[:,0] >= minNumCitations
    
    X = X.iloc[indsWithMin]
    Y = Y.iloc[indsWithMin]
    history = history.iloc[indsWithMin]
    
    (numSamples, numFeatures) = X.shape
    numResponses = Y.shape[1]
    numHistory = history.shape[1]
    
    numTrain = min(50000, int(math.floor(numSamples / 2.0)))
    numValid = min(10000, numSamples - numTrain)
    
    inds = np.random.choice(range(numSamples), numTrain + numValid)
    trainInds = np.sort(inds[range(numTrain)])
    validInds = np.sort(inds[numTrain:(numTrain + numValid - 1)])
    
    trainX = X.iloc[trainInds]
    trainY = Y.iloc[trainInds]
    
    validX = X.iloc[validInds]
    validY = Y.iloc[validInds]
    
    suffixWithMin = "-" + docType + "-" + \
                    "-".join(map(str, [dateConfig['startYear'],
                                       dateConfig['sourceYear'],
                                       dateConfig['targetYear'],
                                       dateConfig['window'],
                                       minNumCitations])) + \
                    ".pickle"

    if not os.path.exists("plusVariableK" + suffixWithMin):
        plusVariableK = PlusVariableKBaselineModel(trainX, trainY, baseFeature, averageFeature)
        pickle.dump(plusVariableK, open("plusVariableK" + suffixWithMin, "wb"))
    else:
        plusVariableK = pickle.load(open("plusVariableK" + suffixWithMin, "rb"))

    if not os.path.exists("plusFixedK" + suffixWithMin):
        plusFixedK = PlusFixedKBaselineModel(trainX, trainY, baseFeature)
        pickle.dump(plusFixedK, open("plusFixedK" + suffixWithMin, "wb"))
    else:
        plusFixedK = pickle.load(open("plusFixedK" + suffixWithMin, "rb"))
        
    if not os.path.exists("simpleLinear" + suffixWithMin):
        simpleLinear = SimpleLinearModel(trainX, trainY, baseFeature)
        pickle.dump(simpleLinear, open("simpleLinear" + suffixWithMin, "wb"))
    else:
        simpleLinear = pickle.load(open("simpleLinear" + suffixWithMin, "rb"))
        
    if not os.path.exists("lasso" + suffixWithMin):
        lasso = LassoModel(trainX, trainY, baseFeature)
        pickle.dump(lasso, open("lasso" + suffixWithMin, "wb"))
    else:
        lasso = pickle.load(open("lasso" + suffixWithMin, "rb"))
        
    if not os.path.exists("rf" + suffixWithMin):
        rf = RandomForestModel(trainX, trainY, baseFeature)
        pickle.dump(rf, open("rf" + suffixWithMin, "wb"))
    else:
        rf = pickle.load(open("rf" + suffixWithMin, "rb"))
    
    mapePlotFileName = "mape" + suffixWithMin.split(".")[0] + ".pdf"
    plotMAPE([plusVariableK, plusFixedK, simpleLinear, lasso, rf], validX, validY, mapePlotFileName)
    
    year = numResponses
    mapePlotFileName = "mapePerCount" + suffixWithMin.split(".")[0] + ".pdf"
    plotMAPEPerCount(rf, validX, validY.values[:, year - 1], year, baseFeature, mapePlotFileName)

