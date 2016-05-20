from config_reader import ConfigReader
from data_manipulation import readData
import numpy as np
import sys
from misc_functions import *

assert(len(sys.argv) >= 3 and len(sys.argv) <= 4)

if len(sys.argv) == 3:
    _, docType, measure = sys.argv
    filtersString = ""
else:
    _, docType, measure, filtersString = sys.argv

filters, filterId = filtersStringToFilters(filtersString)

config = ConfigReader("config.ini", docType, measure, filterId)

X = readData(config.featuresPath)

allowedIndsBoolArray = np.repeat(True, X.shape[0])
for field, minMax in filters.iteritems():
    allowedIndsBoolArray &= (X[[field]].values[:, 0] >= minMax[0]) & (X[[field]].values[:, 0] <= minMax[1])

allInds = np.where(allowedIndsBoolArray)[0]
sampleSize = len(allInds)
testSize = min(int(.2 * sampleSize), 10000)
validSize = min(int(.1 * sampleSize), 10000)
trainSize =  sampleSize - testSize - validSize

np.random.seed(12)
np.random.shuffle(allInds)
trainInds = sorted(allInds[0:trainSize])
validInds = sorted(allInds[trainSize:(trainSize + validSize)])
testInds = sorted(allInds[(trainSize + validSize):len(allInds)])

np.array(trainInds).tofile(config.trainIndsPath, sep ="\t", format ='%d')
np.array(validInds).tofile(config.validIndsPath, sep ="\t", format ='%d')
np.array(testInds).tofile(config.testIndsPath, sep ="\t", format ='%d')