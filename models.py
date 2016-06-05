from sklearn import linear_model, preprocessing, ensemble, pipeline
from sklearn.grid_search import GridSearchCV 
import numpy as np
import pandas
import multiprocessing
import data_manipulation as dm
import xgboost as xgb
import multiprocessing as mp
import os
import cPickle as pickle
import rpp
from misc_functions import mape, mape_with_error

class CitationModel:
    def predict(self, X, year):
        raise Exception("Unimplemented error.")

    def predict_all(self, X):
        preds = []
        for i in range(self.num_years):
            preds.append(self.predict(X, i + 1))
        return np.array(preds).T

    def mape(self, X, y, year):
        return mape_with_error(self.predict(X, year), y)

    def mape_with_error(self, X, y, year):
        return mape_with_error(self.predict(X, year), y)

    def mape_all(self, X, Y):
        return mape(self.predict_all(X), Y.values)

    def mape_all_with_errors(self, X, Y):
        return mape_with_error(self.predict_all(X), Y.values)

class ConstantModel(CitationModel):
    def __init__(self, X, Y, base_feature):
        self.name = "F"
        self.base_feature = base_feature
        self.num_years = Y.shape[1]

    def predict(self, X, year):
        return X[[self.base_feature]].values[:,0]

class PlusVariableKBaselineModel(CitationModel):
    def __init__(self, X, Y, base_feature, average_feature):
        self.name = "Variable k"
        self.base_feature = base_feature
        self.average_feature = average_feature
        self.num_years = Y.shape[1]

        new_xs = []
        new_ys = []
        for i in range(Y.shape[1]):
            new_x = X[[base_feature, average_feature]].values
            new_x[:,1] = new_x[:,1] * (i + 1)
            new_xs.append(new_x)
            new_ys.append(Y[[i]].values)

        new_x = np.concatenate(tuple(new_xs))
        new_y = np.concatenate(tuple(new_ys))

        self.lin_model = linear_model.LinearRegression(copy_X = False)
        self.lin_model.fit(new_x, new_y)

    def predict(self, X, year):
        new_x = X[[self.base_feature, self.average_feature]].values
        new_x[:,1] = new_x[:,1] * year
        return np.maximum(self.lin_model.predict(new_x)[:,0], X[[self.base_feature]].values[:,0])

class PlusKBaselineModel(CitationModel):
    def __init__(self, X, Y, base_feature, k = None):
        self.name = "PK"
        self.base_feature = base_feature
        self.num_years = Y.shape[1]

        if k is None:
            new_xs = []
            new_ys = []
            for i in range(Y.shape[1]):
                new_xs.append(np.repeat(i + 1, X.shape[0]))
                new_ys.append(Y.values[:,i] - X[[base_feature]].values[:,0])

            new_x = np.concatenate(tuple(new_xs))
            new_y = np.concatenate(tuple(new_ys))
            random_state = np.random.get_state() # Prior random state
            np.random.seed(23498) # To make sure SGD always returns the same result
            lin_model = linear_model.SGDRegressor(loss="huber", epsilon=1, penalty="none",
                                                fit_intercept=False)
            self.k = lin_model.fit(new_x.reshape((len(new_x), 1)), new_y).coef_[0]
            np.random.set_state(random_state) # Reset to prior state
        else:
            self.k = k

    def predict(self, X, year):
        return X[[self.base_feature]].values[:,0] + year * self.k

class SimpleLinearModel(CitationModel):
    def __init__(self, X, Y, base_feature, delta_feature):
        self.name = "SM"
        self.base_feature = base_feature
        self.delta_feature = delta_feature
        self.num_years = Y.shape[1]
        self.lin_models = []
        for i in range(Y.shape[1]):
            model = linear_model.LinearRegression()
            model.fit(X[[base_feature, delta_feature]].values, Y.values[:,i])
            self.lin_models.append(model)

    def predict(self, X, year):
        return np.maximum(self.lin_models[year - 1].predict(X[[self.base_feature, self.delta_feature]]),
                          X[[self.base_feature]].values[:,0])

class LassoModel(CitationModel):
    def __init__(self, X, Y, base_feature):
        self.name = "LAS"
        self.base_feature = base_feature
        self.num_years = Y.shape[1]
        self.lasso_models = []
        for i in range(Y.shape[1]):
            rescaler = preprocessing.StandardScaler()
            model = linear_model.LassoCV(cv=10)
            pipe = pipeline.Pipeline([('rescale', rescaler), ('model', model)])
            pipe.fit(X.values, Y.values[:,i])
            self.lasso_models.append(pipe)

    def predict(self, X, year):
        return np.maximum(self.lasso_models[year - 1].predict(X), X[[self.base_feature]].values[:,0])

class RandomForestModel(CitationModel):
    def __init__(self, X, Y, base_feature,
                 rf_params = {"n_estimators": 1500, "max_features": .3333,
                 "min_samples_leaf": 25, "n_jobs": multiprocessing.cpu_count() - 1, "verbose" : 1}):
        self.name = "RF"
        self.base_feature = base_feature
        self.num_years = Y.shape[1]
        self.rf_models = []
        for i in range(Y.shape[1]):
            rf_model = ensemble.RandomForestRegressor(**rf_params)
            rf_model.fit(X.values, Y.values[:,i])
            self.rf_models.append(rf_model)

    def predict(self, X, year):
        return np.maximum(self.rf_models[year - 1].predict(X), X[[self.base_feature]].values[:,0])

    def set_verbose(self, level):
        for rf_model in self.rf_models:
            rf_model.set_params(verbose=level)

class GradientBoostModel(CitationModel):
    def __init__(self, X, Y, base_feature,
                 params = { "loss": "lad", "n_estimators": 500, "verbose" : 1,
                            "min_samples_leaf" : 2, },
                 tune_with_cv = False):
        self.name = "GBRT"
        self.base_feature = base_feature
        self.num_years = Y.shape[1]
        self.gb_models = []
        if tune_with_cv:
            gb_model = ensemble.GradientBoostingRegressor(**params)
            print("Tuning max depth")
            search_with_parameters(gb_model, {"max_depth": [2, 3, 4, 5, 6]}, X.values, Y.values[:, self.num_years - 1])
            print("Tuning min samples")
            search_with_parameters(gb_model, {"min_samples_split": [2, 3, 4],
                                           "min_samples_leaf": [1, 2]}, X.values, Y.values[:, self.num_years - 1])
            print("Tuning subsample percent")
            search_with_parameters(gb_model, {"subsample": [.6, .7, .8, .9, 1.0]}, X.values, Y.values[:, self.num_years - 1])
            params = gb_model.get_params()
        for i in range(Y.shape[1]):
            gb_model = ensemble.GradientBoostingRegressor(**params)
            gb_model.fit(X.values, Y.values[:,i])
            self.gb_models.append(gb_model)

    def predict(self, X, year):
        return np.maximum(self.gb_models[year - 1].predict(X),
                          X[[self.base_feature]].values[:,0])


def mape_error(preds, true_d_matrix):
    return 'mape', np.mean(np.abs(preds - true_d_matrix.get_label()) / (1.0 * true_d_matrix.get_label()))


def set_n_estimators_by_cv(model, X, y, cv_folds=5, early_stopping_rounds=50):
    model.set_params(n_estimators=1000, nthread=mp.cpu_count())
    xg_train = xgb.DMatrix(X, label=y)
    params = model.get_params()
    cv_result = xgb.cv(params, xg_train, num_boost_round=params['n_estimators'],
                      nfold=cv_folds, feval=mape_error,
                      early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
    model.set_params(n_estimators=cv_result.shape[0])


def mape_scorer(estimator, X, y):
    preds = estimator.predict(X)
    return float(-np.mean(np.abs(preds - y) / (1.0 * y)))


def search_with_parameters(model, params_to_search, X, y):
    gsearch = GridSearchCV(estimator=model, param_grid=params_to_search,
                           scoring=mape_scorer, n_jobs=mp.cpu_count(),
                           iid=False, cv=5, verbose=0)
    gsearch.fit(X, y)
    print(gsearch.grid_scores_)
    model.set_params(**gsearch.best_params_)

class XGBoostModel(CitationModel):
    def __init__(self, X, Y, base_feature,
                 params = {"learning_rate" : 0.01, "objective" : "reg:linear",
                           "max_depth" : 6, "min_child_weight" : 1,
                           "gamma" : 1.0, "subsample" : 0.8,
                           "colsample_bytree" : 0.8},
                 tune_with_cv = True):
        params['nthread'] = 0
        self.name = "XGB"
        self.base_feature = base_feature
        self.num_years = Y.shape[1]
        self.params = params
        self.xg_models = []
        for i in range(Y.shape[1]):
            xgb_model = xgb.sklearn.XGBRegressor(**params)
            if tune_with_cv:
                self.__fit_model_by_cv(xgb_model, X, Y[[i]].values[:,0])
            elif os.path.exists("data/xgb-params.pickle"):
                params_list = pickle.load(open("data/xgb-params.pickle", "rb"))
                xgb_model.set_params(**(params_list[i]))
            else:
                set_n_estimators_by_cv(xgb_model, X, Y[[i]].values[:,0])
            xgb_model.set_params(nthread = mp.cpu_count())
            xgb_model.fit(X, Y[[i]].values[:,0])
            self.xg_models.append(xgb_model)

    def __fit_model_by_cv(self, xgb_model, X, y):
        xgb_model.set_params(nthread=0)
        set_n_estimators_by_cv(xgb_model, X, y)

        params1 = {'max_depth': [3, 4, 5, 6, 7],
                   'min_child_weight': [1, 2, 4, 8]}

        search_with_parameters(xgb_model, params1, X, y)

        params2 = {'gamma': [0] + (10 ** np.linspace(-3, 0, 9)).tolist()}
        search_with_parameters(xgb_model, params2, X, y)

        xgb_model.set_params(nthread=0)
        set_n_estimators_by_cv(xgb_model, X, y)

        params3 = {'subsample': [i / 10.0 for i in range(6, 10)],
                   'colsample_bytree': [i / 10.0 for i in range(6, 10)]}
        search_with_parameters(xgb_model, params3, X, y)

        params4 = {'reg_alpha': [0] + (10 ** np.linspace(-3, 2, 3)).tolist(),
                   'reg_lambda': [0] + (10 ** np.linspace(-3, 2, 3)).tolist()}
        search_with_parameters(xgb_model, params4, X, y)

    def predict(self, X, year):
        return np.maximum(self.xg_models[year - 1].predict(X), X[[self.base_feature]].values[:, 0])

class RPPNetWrapper(CitationModel):
    def __init__(self, X, histories, Y, model_save_path = None, gamma=.05,
                 maxiter=10, with_features = True):
        self.name = "RPPNet" if with_features else "RPP"
        self.num_years = Y.shape[1]
        self.gamma = gamma
        self._with_features = with_features
        self.rpp_net = rpp.RPPNet(gamma, model_save_path, maxiter=maxiter)
        if not self.rpp_net.is_fit():
            if not with_features:
                self.rpp_net.fit(pandas.DataFrame(np.zeros((X.shape[0], 0))), histories)
            else:
                self.rpp_net.fit(X, histories)

    def set_prediction_histories(self, histories):
        self._pred_histories = histories

    def predict(self, X, year):
        if not self._with_features:
            return self.rpp_net.predict(pandas.DataFrame(np.zeros((X.shape[0], 0))),
                                        self._pred_histories, year)[:, -1]
        else:
            return self.rpp_net.predict(X, self._pred_histories, year)[:, -1]

    def predict_all(self, X):
        if not self._with_features:
            return self.rpp_net.predict(pandas.DataFrame(np.zeros((X.shape[0], 0))),
                                        self._pred_histories, self.num_years)
        else:
            return self.rpp_net.predict(X, self._pred_histories, self.num_years)

class RPPStub(CitationModel):
    def __init__(self, config, train_x, valid_x, test_x, with_features = True, custom_suffix = None):
        if with_features:
            self.name = "RPPNet"
            to_append = "-all"
        else:
            self.name = "RPP"
            to_append = "-none"

        if custom_suffix != None:
            suffix = to_append + custom_suffix
        else:
            suffix = to_append + "-" + config.full_suffix + ".tsv"

        valid_preds_file_path = config.rel_path + "rppPredictions-valid" + suffix
        train_preds_file_path = config.rel_path + "rppPredictions-train" + suffix
        test_preds_file_path = config.rel_path + "rppPredictions-test" + suffix
        self.train_x = train_x
        self.valid_x = valid_x
        self.test_x = test_x
        self.train_preds = dm.read_data(train_preds_file_path, header = None)
        self.valid_preds = dm.read_data(valid_preds_file_path, header=None)
        self.test_preds = dm.read_data(test_preds_file_path, header=None)
        self.num_years = self.valid_preds.shape[1]
        self.base_feature = config.base_feature

    def predict(self, X, year):
        if self.train_x.shape == X.shape and np.all(self.train_x.values == X.values):
            return self.train_preds[[year - 1]].values[:,0]
        elif self.valid_x.shape == X.shape and np.all(self.valid_x.values == X.values):
            return self.valid_preds[[year - 1]].values[:,0]
        elif self.test_x.shape == X.shape and np.all(self.test_x.values == X.values):
            return self.test_preds[[year - 1]].values[:, 0]
        else:
            raise Exception("Unimplemented error.")