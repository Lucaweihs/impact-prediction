import pandas
import numpy as np

def readData(filePath, header = 0):
    return pandas.read_csv(filePath, sep = "\t", header = header)

def readHistories(filePath):
    histories = []
    with open(filePath, "r") as f:
        for line in f:
            histories.append(np.array([int(i) for i in line.split(("\t"))]))
    return histories

def getTrainValidTestData(data, trainInds, validInds, testInds):
    trainData = data.iloc[trainInds]
    validData = data.iloc[validInds]
    testData = data.iloc[testInds]
    return (trainData, validData, testData)

def getTrainValidTestHistories(histories, trainInds, validInds, testInds):
    trainHists = [histories[i] for i in trainInds]
    validHists = [histories[i] for i in validInds]
    testHists = [histories[i] for i in testInds]
    return (trainHists, validHists, testHists)

def getTrainValidTestIndsFromConfig(config):
    trainInds = np.genfromtxt(config.trainIndsPath, delimiter = "\t", dtype = int)
    validInds = np.genfromtxt(config.validIndsPath, delimiter = "\t", dtype = int)
    testInds = np.genfromtxt(config.testIndsPath, delimiter = "\t", dtype = int)
    return (trainInds, validInds, testInds)
