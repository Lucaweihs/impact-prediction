from sklearn import linear_model, preprocessing, ensemble, pipeline
import numpy as np
import multiprocessing
import data_manipulation as dm

class CitationModel:       
    def predict(self, X, year):
        raise Exception("Unimplemented error.")
    
    def predictAll(self, X):
        preds = []
        for i in range(self.numYears):
            preds.append(self.predict(X, i + 1))
        return preds
        
    def mape(self, X, y, year):
        inds = (y != 0)
        return np.mean(np.abs(y - self.predict(X, year))[inds] / (1.0 * y[inds]))

    def mapeWithError(self, X, y, year):
        inds = (y != 0)
        absDiffs = np.abs(y - self.predict(X, year))[inds] / (1.0 * y[inds])
        mape = np.mean(absDiffs)
        sd = np.sqrt(np.var(absDiffs) / len(absDiffs))
        return (mape, sd)
        
    def mapeAll(self, X, Y):
        mapes = np.zeros(self.numYears)
        for i in range(self.numYears):
            mapes[i] = self.mape(X, Y.values[:, i], i + 1)
        return mapes

    def mapeAllWithErrors(self, X, Y):
        mapes = []
        errors = []
        for i in range(self.numYears):
            mape, error = self.mapeWithError(X, Y.values[:, i], i + 1)
            mapes.append(mape)
            errors.append(error)
        return (mapes, np.array(errors))

class ConstantModel(CitationModel):
    def __init__(self, X, Y, baseFeature):
        self.name = "Constant"
        self.baseFeature = baseFeature
        self.numYears = Y.shape[1]

    def predict(self, X, year):
        return X[[self.baseFeature]].values[:,0]

class PlusVariableKBaselineModel(CitationModel):
    def __init__(self, X, Y, baseFeature, averageFeature):
        self.name = "Variable k"
        self.baseFeature = baseFeature
        self.averageFeature = averageFeature
        self.numYears = Y.shape[1]
        
        newXs = []
        newYs = []
        for i in range(Y.shape[1]):
            newX = X[[baseFeature, averageFeature]].values
            newX[:,1] = newX[:,1] * (i + 1)
            newXs.append(newX)
            newYs.append(Y[[i]].values)
        
        newX = np.concatenate(tuple(newXs))
        newY = np.concatenate(tuple(newYs))
        
        self.linModel = linear_model.LinearRegression(copy_X = False)
        self.linModel.fit(newX, newY)
        
    def predict(self, X, year):
        newX = X[[self.baseFeature, self.averageFeature]].values
        newX[:,1] = newX[:,1] * year
        return np.maximum(self.linModel.predict(newX)[:,0], X[[self.baseFeature]].values[:,0])

class PlusFixedKBaselineModel(CitationModel):
    def __init__(self, X, Y, baseFeature):
        self.name = "Fixed k"
        self.baseFeature = baseFeature
        self.numYears = Y.shape[1]
        
        newXs = []
        newYs = []
        for i in range(Y.shape[1]):
            newX = np.column_stack((X[[baseFeature]].values, np.repeat(i + 1, X.shape[0])))
            newXs.append(newX)
            newYs.append(Y[[i]].values)
        
        newX = np.concatenate(tuple(newXs))
        newY = np.concatenate(tuple(newYs))
        
        self.linModel = linear_model.LinearRegression(copy_X = False)
        self.linModel.fit(newX, newY)
        
    def predict(self, X, year):
        newX = np.column_stack((X[[self.baseFeature]].values, np.repeat(year, X.shape[0])))
        return np.maximum(self.linModel.predict(newX)[:,0], X[[self.baseFeature]].values[:,0])
        
class SimpleLinearModel(CitationModel):
    def __init__(self, X, Y, baseFeature):
        self.name = "Simple Linear"
        self.baseFeature = baseFeature
        self.numYears = Y.shape[1]
        self.linModels = []
        for i in range(Y.shape[1]):
            model = linear_model.LinearRegression()
            model.fit(X.values, Y.values[:,i])
            self.linModels.append(model)
            
    def predict(self, X, year):
        return np.maximum(self.linModels[year - 1].predict(X), X[[self.baseFeature]].values[:,0])
        
class LassoModel(CitationModel):
    def __init__(self, X, Y, baseFeature):
        self.name = "Lasso"
        self.baseFeature = baseFeature
        self.numYears = Y.shape[1]
        self.lassoModels = []
        for i in range(Y.shape[1]):
            rescaler = preprocessing.StandardScaler()
            model = linear_model.LassoCV(cv = 10)
            pipe = pipeline.Pipeline([('rescale', rescaler), ('model', model)])
            pipe.fit(X.values, Y.values[:,i])
            self.lassoModels.append(pipe)
            
    def predict(self, X, year):
        return np.maximum(self.lassoModels[year - 1].predict(X), X[[self.baseFeature]].values[:,0])
            
class RandomForestModel(CitationModel):
    def __init__(self, X, Y, baseFeature,
                 rfParams = {"n_estimators": 300, "max_features": .333,
                 "min_samples_leaf": 5, "n_jobs": multiprocessing.cpu_count() - 1}):
        self.name = "Random Forest"
        self.baseFeature = baseFeature
        self.numYears = Y.shape[1]
        self.rfModels = []
        for i in range(Y.shape[1]):
            rfModel = ensemble.RandomForestRegressor(**rfParams)
            rfModel.fit(X.values, Y.values[:,i])
            self.rfModels.append(rfModel)
    
    def predict(self, X, year):
        return np.maximum(self.rfModels[year - 1].predict(X), X[[self.baseFeature]].values[:,0])

class RPPStub(CitationModel):
    def __init__(self, config, withFeatures = True):
        if withFeatures:
            self.name = "RPP With"
        else:
            self.name = "RPP Without"
        if withFeatures:
            predsFilePath = config.relPath + "rppPredictionsWithFeatures" + config.fullSuffix + ".tsv"
        else:
            predsFilePath = config.relPath + "rppPredictionsWithoutFeatures" + config.fullSuffix + ".tsv"
        self.predictions = dm.readData(predsFilePath, header = None)
        self.numYears = self.predictions.shape[1]
        self.baseFeature = config.baseFeature

    def predict(self, X, year):
        return self.predictions[[year - 1]].values[:,0]
