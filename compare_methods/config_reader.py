import configparser

class ConfigReader:
    def __init__(self, fileName, docType, measure, idString):
        cp = configparser.ConfigParser()
        cp._interpolation = configparser.ExtendedInterpolation()
        cp.read(fileName)

        self.window = int(cp.get("General", "window"))
        self.startYear = int(cp.get("General", "startYear"))
        self.sourceYear = int(cp.get("General", "sourceYear"))
        self.targetYear = int(cp.get("General", "targetYear"))

        self.relPath = str(cp.get("General", "path"))

        yearPartOfSuffix = "-".join(map(str, [self.startYear, self.sourceYear,
                                                    self.targetYear, self.window]))

        self.fullSuffix = docType + "-" + measure + "-" + idString + "-" + yearPartOfSuffix
        self.fullSuffixNoMeasure = docType + "-" + idString + "-" + yearPartOfSuffix

        self.docType = docType
        self.featuresPath = self.relPath + self.docType + "Features-" + yearPartOfSuffix + ".tsv"
        self.responsesPath = self.relPath + self.docType + "Responses-" + yearPartOfSuffix + ".tsv"
        self.historyPath = self.relPath + self.docType + "Histories-" + yearPartOfSuffix + ".tsv"

        self.measure = measure

        docTypeCap = self.docType.capitalize()
        measureCap = self.measure.capitalize()
        configSection = docTypeCap + measureCap

        self.baseFeature = str(cp.get(configSection, "baseFeature"))
        self.citationFeature = str(cp.get(configSection, "citationFeature"))
        self.averageFeature = str(cp.get(configSection, "averageFeature"))
        self.deltaFeature = str(cp.get(configSection, "deltaFeature"))
        self.ageFeature = str(cp.get(configSection, "ageFeature"))

        self.trainIndsPath = self.relPath + "trainInds-" + self.fullSuffixNoMeasure + ".tsv"
        self.testIndsPath = self.relPath + "testInds-" + self.fullSuffixNoMeasure + ".tsv"
        self.validIndsPath = self.relPath + "validInds-" + self.fullSuffixNoMeasure + ".tsv"
