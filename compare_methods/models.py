from sklearn import linear_model, preprocessing, ensemble, pipeline
from sklearn.grid_search import GridSearchCV 
import numpy as np
import multiprocessing
import data_manipulation as dm
import xgboost as xgb
import multiprocessing as mp
import os
import cPickle as pickle
import rpp

class CitationModel:
    def predict(self, X, year):
        raise Exception("Unimplemented error.")
    
    def predictAll(self, X):
        preds = []
        for i in range(self.numYears):
            preds.append(self.predict(X, i + 1))
        return preds
        
    def mape(self, X, y, year):
        return self._mape_with_error(self.predict(X, year), y)

    def _mape_with_error(self, preds, truth):
        assert(not np.any(truth == 0))
        abs_diffs = np.abs((preds - truth) / (1.0 * truth))
        mape = np.mean(abs_diffs, axis=0)
        sd = np.sqrt(np.var(abs_diffs, axis=0) / abs_diffs.shape[0])
        return (mape, sd)

    def _mape(self, preds, truth):
        assert (not np.any(truth == 0))
        abs_diffs = np.abs((preds - truth) / (1.0 * truth))
        mape = np.mean(abs_diffs, axis=0)
        return mape

    def mapeWithError(self, X, y, year):
        return self._mape_with_error(self.predict(X, year), y)
        
    def mapeAll(self, X, Y):
        return self._mape(self.predictAll(X), Y.values)

    def mapeAllWithErrors(self, X, Y):
        return self._mape_with_error(self.predictAll(X), Y.values)

class ConstantModel(CitationModel):
    def __init__(self, X, Y, baseFeature):
        self.name = "F"
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

class PlusKBaselineModel(CitationModel):
    def __init__(self, X, Y, baseFeature, k = None):
        self.name = "PK"
        self.baseFeature = baseFeature
        self.numYears = Y.shape[1]

        if k is None:
            newXs = []
            newYs = []
            for i in range(Y.shape[1]):
                newXs.append(np.repeat(i + 1, X.shape[0]))
                newYs.append(Y.values[:,i] - X[[baseFeature]].values[:,0])

            newX = np.concatenate(tuple(newXs))
            newY = np.concatenate(tuple(newYs))

            linModel = linear_model.SGDRegressor(loss="huber", epsilon = 1, penalty="none",
                                                fit_intercept=False)
            self.k = linModel.fit(newX.reshape((len(newX), 1)), newY).coef_[0]
        else:
            self.k = k
        print "Plus-k model has constant k = " + str(self.k)
        
    def predict(self, X, year):
        return X[[self.baseFeature]].values[:,0] + year * self.k
        
class SimpleLinearModel(CitationModel):
    def __init__(self, X, Y, baseFeature, deltaFeature):
        self.name = "SM"
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
        self.name = "LAS"
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
                 "min_samples_leaf": 25, "n_jobs": multiprocessing.cpu_count() - 1, "verbose" : 1}):
        self.name = "RF"
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
                 params = { "loss": "lad", "n_estimators": 500, "verbose" : 1,
                            "min_samples_leaf" : 2, },
                 tuneWithCV = False):
        self.name = "GBRT"
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

class RPPNetWrapper(CitationModel):
    def __init__(self, X, histories, Y, model_save_path = None):
        self.name = "RPPNet-tf"
        self.num_years = Y.shape[1]
        self.numYears = Y.shape[1]
        self.rpp_net = rpp.RPPNet(0.1, model_save_path, maxiter=1)
        if not self.rpp_net.is_fit():
            self.rpp_net.fit(X, histories)

    def set_prediction_histories(self, histories):
        self._pred_histories = histories

    def predict(self, X, year):
        return self.rpp_net.predict(X, self._pred_histories, year)[:,-1]

    def predictAll(self, X):
        return self.rpp_net.predict(X, self._pred_histories, self.num_years)

class RPPStub(CitationModel):
    def __init__(self, config, trainX, validX, testX, withFeatures = True, customSuffix = None):
        if withFeatures:
            self.name = "RPPNet"
            toAppend = "-all"
        else:
            self.name = "RPP"
            toAppend = "-none"

        if customSuffix != None:
            suffix = toAppend + customSuffix
        else:
            suffix = toAppend + "-" + config.fullSuffix + ".tsv"

        validPredsFilePath = config.relPath + "rppPredictions-valid" + suffix
        trainPredsFilePath = config.relPath + "rppPredictions-train" + suffix
        testPredsFilePath = config.relPath + "rppPredictions-test" + suffix
        self.trainX = trainX
        self.validX = validX
        self.testX = testX
        self.trainPreds = dm.readData(trainPredsFilePath, header = None)
        self.validPreds = dm.readData(validPredsFilePath, header=None)
        self.testPreds = dm.readData(testPredsFilePath, header=None)
        self.numYears = self.validPreds.shape[1]
        self.baseFeature = config.baseFeature

    def predict(self, X, year):
        if self.trainX.shape == X.shape and np.all(self.trainX.values == X.values):
            return self.trainPreds[[year - 1]].values[:,0]
        elif self.validX.shape == X.shape and np.all(self.validX.values == X.values):
            return self.validPreds[[year - 1]].values[:,0]
        elif self.testX.shape == X.shape and np.all(self.testX.values == X.values):
            return self.testPreds[[year - 1]].values[:, 0]
        else:
            raise Exception("Unimplemented error.")
