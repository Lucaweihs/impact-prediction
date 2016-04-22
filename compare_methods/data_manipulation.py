import pandas
import numpy as np

def readData(filePath, header = 0):
    return pandas.read_csv(filePath, sep = "\t", header = header)

def getTrainValidTest(data, trainInds, validInds, testInds):
    trainData = data.iloc[trainInds]
    validData = data.iloc[validInds]
    testData = data.iloc[testInds]
    return (trainData, validData, testData)

def getTrainValidTestIndsFromConfig(config):
    trainInds = np.genfromtxt(config.trainIndsPath, delimiter = "\t", dtype = int)
    validInds = np.genfromtxt(config.validIndsPath, delimiter = "\t", dtype = int)
    testInds = np.genfromtxt(config.testIndsPath, delimiter = "\t", dtype = int)
    return (trainInds, validInds, testInds)
