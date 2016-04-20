from test_runners import *

dateConfig = { "window": 2, "startYear": 2005, "sourceYear": 2010, "targetYear": 2015}
minNumCitations = 20

baseFeature = "paperCitationCount"
citationFeature = "paperCitationCount"
ageFeature = "paperAge"
averageFeature = "paperMeanCitationsPerYear"

path = "../../../../../../ai2/scholar/prediction/src/main/resources/org/allenai/prediction/"
docType = "paper"

suffix = "-" + "-".join(map(str,
                            [dateConfig['startYear'], dateConfig['sourceYear'],
                             dateConfig['targetYear'], dateConfig['window']])) + ".tsv"

featuresPath = path + docType + "Features" + suffix
responsesPath = path + docType + "Responses" + suffix
historyPath = path + docType + "Histories" + suffix

X = readData(featuresPath)
Y = readData(responsesPath)
history = readData(historyPath)

runTests(X, Y, history, docType, dateConfig, baseFeature, averageFeature, citationFeature, minNumCitations)
