from test_runners import *

dateConfig = { "window": 2, "startYear": 2005, "sourceYear": 2010, "targetYear": 2015}
minNumCitations = 5

measure = "hindex"
baseFeature = "authorHInds"
citationFeature = "authorCitationCount"
ageFeature = "authorAge"
averageFeature = "authorHIndDelta"

path = "../../../../../../ai2/scholar/prediction/src/main/resources/org/allenai/prediction/"
docType = "author"

suffix = "-" + "-".join(map(str,
                            [dateConfig['startYear'], dateConfig['sourceYear'],
                             dateConfig['targetYear'], dateConfig['window']])) + ".tsv"

featuresPath = path + docType + "Features" + suffix
responsesPath = path + docType + "Responses" + suffix
historyPath = path + docType + "Histories" + suffix

X = readData(featuresPath)
Y = readData(responsesPath)
history = readData(historyPath)

Y = Y.select(lambda x: measure in x.lower(), axis = 1)
history = history.select(lambda x: measure in x.lower(), axis = 1)

runTests(X, Y, history, docType + "-" + measure, dateConfig, baseFeature, averageFeature, citationFeature, minNumCitations)

