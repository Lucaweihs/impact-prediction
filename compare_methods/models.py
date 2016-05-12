from sklearn import linear_model, preprocessing, ensemble, pipeline
from sklearn.grid_search import GridSearchCV 
import numpy as np
import multiprocessing
import data_manipulation as dm
import xgboost as xgb
import multiprocessing as mp
import os
import cPickle as pickle

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
    def __init__(self, X, Y, baseFeature, deltaFeature):
        self.name = "Simple Linear"
        self.baseFeature = baseFeature
        self.deltaFeature = deltaFeature
        self.numYears = Y.shape[1]
        self.linModels = []
        for i in range(Y.shape[1]):
            model = linear_model.LinearRegression()
            model.fit(X[[baseFeature, deltaFeature]].values, Y.values[:,i])
            self.linModels.append(model)
            
    def predict(self, X, year):
        return np.maximum(self.linModels[year - 1].predict(X[[self.baseFeature, self.deltaFeature]]),
                          X[[self.baseFeature]].values[:,0])
        
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
                 rfParams = {"n_estimators": 1500, "max_features": .3333,
                 "min_samples_leaf": 25, "n_jobs": multiprocessing.cpu_count() - 1}):
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

class GradientBoostModel(CitationModel):
    def __init__(self, X, Y, baseFeature,
                 params = {"loss": "lad", "n_estimators": 500, "verbose" : 1},
                 tuneWithCV = False):
        self.name = "Gradient Boost"
        self.baseFeature = baseFeature
        self.numYears = Y.shape[1]
        self.gbModels = []
        if tuneWithCV:
            gbModel = ensemble.GradientBoostingRegressor(**params)
            print("Tuning max depth\n")
            searchWithParameters(gbModel, {"max_depth": [2, 3, 4, 5, 6]}, X.values, Y.values[:, self.numYears - 1])
            print("Tuning min samples\n")
            searchWithParameters(gbModel, {"min_samples_split": [2, 3, 4],
                                           "min_samples_leaf": [1, 2]}, X.values, Y.values[:, self.numYears - 1])
            print("Tuning subsample percent\n")
            searchWithParameters(gbModel, {"subsample": [.6, .7, .8, .9, 1.0]}, X.values, Y.values[:, self.numYears - 1])
            params = gbModel.get_params()
        for i in range(Y.shape[1]):
            gbModel = ensemble.GradientBoostingRegressor(**params)
            gbModel.fit(X.values, Y.values[:,i])
            self.gbModels.append(gbModel)

    def predict(self, X, year):
        return np.maximum(self.gbModels[year - 1].predict(X),
                          X[[self.baseFeature]].values[:,0])


def mapeError(preds, trueDMatrix):
    return 'mape', np.mean(np.abs(preds - trueDMatrix.get_label()) / (1.0 * trueDMatrix.get_label()))


def setNEstimatorsByCV(model, X, y, cv_folds=5, early_stopping_rounds=50):
    model.set_params(n_estimators=1000, nthread=mp.cpu_count())
    xgTrain = xgb.DMatrix(X, label=y)
    params = model.get_params()
    cvResult = xgb.cv(params, xgTrain, num_boost_round=params['n_estimators'],
                      nfold=cv_folds, feval=mapeError,
                      early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
    model.set_params(n_estimators=cvResult.shape[0])


def mapeScorer(estimator, X, y):
    preds = estimator.predict(X)
    return float(-np.mean(np.abs(preds - y) / (1.0 * y)))


def searchWithParameters(model, paramsToSearch, X, y):
    gsearch = GridSearchCV(estimator=model, param_grid=paramsToSearch,
                           scoring=mapeScorer, n_jobs=mp.cpu_count(),
                           iid=False, cv=5, verbose=0)
    gsearch.fit(X, y)
    print(gsearch.grid_scores_)
    model.set_params(**gsearch.best_params_)

class XGBoostModel(CitationModel):
    def __init__(self, X, Y, baseFeature,
                 params = {"learning_rate" : 0.01, "objective" : "reg:linear",
                           "max_depth" : 6, "min_child_weight" : 1,
                           "gamma" : 1.0, "subsample" : 0.8,
                           "colsample_bytree" : 0.8},
                 tuneWithCV = True):
        params['nthread'] = 0
        self.name = "XGB"
        self.baseFeature = baseFeature
        self.numYears = Y.shape[1]
        self.params = params
        self.xgModels = []
        for i in range(Y.shape[1]):
            xgbModel = xgb.sklearn.XGBRegressor(**params)
            if tuneWithCV:
                self.__fitModelByCV(xgbModel, X, Y[[i]].values[:,0])
            elif os.path.exists("data/xgb-params.pickle"):
                paramsList = pickle.load(open("data/xgb-params.pickle", "rb"))
                xgbModel.set_params(**(paramsList[i]))
            else:
                setNEstimatorsByCV(xgbModel, X, Y[[i]].values[:,0])
            xgbModel.set_params(nthread = mp.cpu_count())
            xgbModel.fit(X, Y[[i]].values[:,0])
            self.xgModels.append(xgbModel)

    def __fitModelByCV(self, xgbModel, X, y):
        xgbModel.set_params(nthread=0)
        setNEstimatorsByCV(xgbModel, X, y)

        params1 = {'max_depth': [3, 4, 5, 6, 7],
                   'min_child_weight': [1, 2, 4, 8]}

        searchWithParameters(xgbModel, params1, X, y)

        params2 = {'gamma': [0] + (10 ** np.linspace(-3, 0, 9)).tolist()}
        searchWithParameters(xgbModel, params2, X, y)

        xgbModel.set_params(nthread=0)
        setNEstimatorsByCV(xgbModel, X, y)

        params3 = {'subsample': [i / 10.0 for i in range(6, 10)],
                   'colsample_bytree': [i / 10.0 for i in range(6, 10)]}
        searchWithParameters(xgbModel, params3, X, y)

        params4 = {'reg_alpha': [0] + (10 ** np.linspace(-3, 2, 3)).tolist(),
                   'reg_lambda': [0] + (10 ** np.linspace(-3, 2, 3)).tolist()}
        searchWithParameters(xgbModel, params4, X, y)

    def predict(self, X, year):
        return np.maximum(self.xgModels[year - 1].predict(X), X[[self.baseFeature]].values[:, 0])

class RPPStub(CitationModel):
    def __init__(self, config, trainX, validX, withFeatures = True, customSuffix = None):
        if withFeatures:
            self.name = "RPPNet"
            toAppend = "-all"
        else:
            self.name = "RPPNet Intercept"
            toAppend = "-none"

        if customSuffix != None:
            suffix = toAppend + customSuffix
        else:
            suffix = toAppend + config.fullSuffix + ".tsv"

        validPredsFilePath = config.relPath + "rppPredictions-valid" + suffix
        trainPredsFilePath = config.relPath + "rppPredictions-train" + suffix
        self.trainX = trainX
        self.validX = validX
        self.validPreds = dm.readData(validPredsFilePath, header = None)
        self.trainPreds = dm.readData(trainPredsFilePath, header = None)
        self.numYears = self.validPreds.shape[1]
        self.baseFeature = config.baseFeature

    def predict(self, X, year):
        if self.trainX.shape == X.shape and np.all(self.trainX.values == X.values):
            return self.trainPreds[[year - 1]].values[:,0]
        elif self.validX.shape == X.shape and np.all(self.validX.values == X.values):
            return self.validPreds[[year - 1]].values[:,0]
        else:
            raise Exception("Unimplemented error.")
