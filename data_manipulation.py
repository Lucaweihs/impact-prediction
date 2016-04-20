import pandas

def readData(filePath):
    return pandas.read_csv(filePath, sep = "\t", header = 0)