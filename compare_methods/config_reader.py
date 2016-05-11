import configparser

class ConfigReader:
    def __init__(self, fileName, docType, measure, minNumCitations, minAge = 0, minBase = None):
        cp = configparser.ConfigParser()
        cp._interpolation = configparser.ExtendedInterpolation()
        cp.read(fileName)

        self.window = int(cp.get("General", "window"))
        self.startYear = int(cp.get("General", "startYear"))
        self.sourceYear = int(cp.get("General", "sourceYear"))
        self.targetYear = int(cp.get("General", "targetYear"))
        self.minNumCitations = minNumCitations
        self.minAge = minAge
        self.minBase = minBase

        self.relPath = str(cp.get("General", "path"))
        self.suffix = "-" + "-".join(map(str, [self.startYear, self.sourceYear,
                                         self.targetYear, self.window]))
        if minBase != None:
            minBaseString = "-" + str(minBase)
        else:
            minBaseString = ""

        self.fullSuffix = "-" + docType + "-" + measure + "-" + str(self.minNumCitations) + "-" + str(minAge) + minBaseString + self.suffix
        self.fullSuffixNoMeasure = "-" + docType + "-" + str(self.minNumCitations) + "-" + str(minAge) + minBaseString + self.suffix

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
        self.deltaFeature = str(cp.get(configSection, "deltaFeature"))
        self.ageFeature = str(cp.get(configSection, "ageFeature"))

        self.trainIndsPath = self.relPath + "trainInds" + self.fullSuffixNoMeasure + ".tsv"
        self.testIndsPath = self.relPath + "testInds" + self.fullSuffixNoMeasure + ".tsv"
        self.validIndsPath = self.relPath + "validInds" + self.fullSuffixNoMeasure + ".tsv"
