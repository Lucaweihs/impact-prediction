import configparser

class ConfigReader:
    def __init__(self, fileName, docType, measure):
        cp = configparser.ConfigParser()
        cp._interpolation = configparser.ExtendedInterpolation()
        cp.read(fileName)
        
        self.window = int(cp.get("General", "window"))
        self.startYear = int(cp.get("General", "startYear"))
        self.sourceYear = int(cp.get("General", "sourceYear"))
        self.targetYear = int(cp.get("General", "targetYear"))
        self.minNumCitations = int(cp.get("General", "minNumCitations"))
        
        self.relPath = str(cp.get("General", "path"))
        self.suffix = "-" + "-".join(map(str, [self.startYear, self.sourceYear,
                                         self.targetYear, self.window]))
        self.suffixWithMinCites = "-" + str(self.minNumCitations) + self.suffix
        self.fullSuffix = "-" + docType + "-" + measure + self.suffixWithMinCites

        self.docType = docType
        self.featuresPath = self.relPath + self.docType + "Features" + self.suffix + ".tsv"
        self.responsesPath = self.relPath + self.docType + "Responses" + self.suffix + ".tsv"
        self.historyPath = self.relPath + self.docType + "Histories" + self.suffix + ".tsv"

        self.measure = measure

        docTypeCap = self.docType.capitalize()
        measureCap = self.measure.capitalize()
        configSection = docTypeCap + measureCap

        self.baseFeature = str(cp.get(configSection, "baseFeature"))
        self.citationFeature = str(cp.get(configSection, "citationFeature"))
        self.averageFeature = str(cp.get(configSection, "averageFeature"))

        self.trainIndsPath = self.relPath + "trainInds" + self.fullSuffix + ".tsv"
        self.testIndsPath = self.relPath + "testInds" + self.fullSuffix + ".tsv"
        self.validIndsPath = self.relPath + "validInds" + self.fullSuffix + ".tsv"
