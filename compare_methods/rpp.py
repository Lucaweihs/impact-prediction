import scipy.stats as stats
from scipy.special import gammaln as log_gamma
from scipy.special import psi as digamma
import scipy.optimize as optimize
import numpy as np
from sklearn import preprocessing
from joblib import Parallel, delayed
import multiprocessing
import math
import os
import cPickle as pickle

import tensorflow as tf

def _rpp_neg_loglike(alpha, beta, N, cite_diff_sum, const_wrt_alpha_beta):
    return tf.reduce_mean(-(const_wrt_alpha_beta + alpha * tf.log(beta) -
                            tf.lgamma(alpha) + tf.lgamma(alpha + N) -
                            (alpha + N) * tf.log(beta + cite_diff_sum)), name="rpp_neg_loglike")

def _rpp_squared_prior_mean_penalty(alpha, beta, gamma):
    return tf.reduce_mean(gamma * (alpha / beta) ** 2)

def _rpp_loss(alpha, beta, N, cite_diff_sum, const_wrt_alpha_beta, gamma):
    return tf.add(_rpp_neg_loglike(alpha, beta, N, cite_diff_sum, const_wrt_alpha_beta),
                  _rpp_squared_prior_mean_penalty(alpha, beta, gamma),
                  name="rpp_loss")

def _rpp_one_layer_alpha_beta(x, num_features):
    with tf.name_scope('only_layer'):
        w1 = tf.Variable(
            tf.zeros([num_features, 2]),
            name="w1"
        )
        b1 = tf.Variable(tf.zeros([2]) + 1, name="b1")
        fc_1 = tf.nn.softplus(tf.matmul(x, w1) + b1) + .001
    return fc_1

def _rpp_multi_layer_alpha_beta(x, keep_prob, num_features):
    with tf.name_scope('layer1'):
        num_out1 = int(num_features / 2)
        w1 = tf.Variable(
            tf.zeros([num_features, num_out1]),
            name="w1"
        )
        b1 = tf.Variable(tf.zeros([num_out1]), name="b1")
        fc_1 = tf.nn.relu(tf.matmul(x, w1) + b1)

    with tf.name_scope('layer2'):
        num_out2 = int(num_out1 / 2)
        w2 = tf.Variable(
            tf.zeros([num_out1, num_out2]),
            name="w2"
        )
        b2 = tf.Variable(tf.zeros([num_out2]), name="b2")
        fc_2 = tf.nn.relu(tf.matmul(fc_1, w2) + b2)

    with tf.name_scope('layer3'):
        num_out3 = int(num_out2 / 2)
        w3 = tf.Variable(
            tf.zeros([num_out2, num_out3]),
            name="w3"
        )
        b3 = tf.Variable(tf.zeros([num_out3]), name="b3")
        fc_3 = tf.nn.relu(tf.matmul(fc_2, w3) + b3)

    fc_3_drop = tf.nn.dropout(fc_3, keep_prob)

    with tf.name_scope('layer4'):
        num_out4 = 2
        w4 = tf.Variable(
            tf.zeros([num_out3, num_out4]),
            name="w4"
        )
        b4 = tf.Variable(tf.zeros([num_out4]) + 1, name="b3")
        fc_4 = tf.nn.softplus(tf.matmul(fc_3_drop, w4) + b4) + .001
    return fc_4

def _rpp_train(loss):
    optimizer = tf.train.AdamOptimizer()
    global_step = tf.Variable(0, name='global_step', trainable=False)
    return optimizer.minimize(loss, global_step=global_step)

class RPP(object):
    def __init__(self, history):
        self._history = history
        self.alpha = 10.0
        self.beta = 12.0
        self.m = 10.0
        self._init_cum_hist_with_m = np.cumsum(np.hstack(((self.m,), history)))[:-1]
        self._mean = 1.5
        self._sd = 0.5
        self._cache = {}

        # Constants in the log-likelihood
        self._len_history = len(history)
        self._N = np.sum(history)
        self._c_const = np.dot(np.log(self._init_cum_hist_with_m), history)
        self._d_const = np.sum(log_gamma(history + 1))
        self._const = self._c_const - self._d_const

    def _clear_cache(self):
        self._cache.clear()

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, mean):
        if mean != self._mean:
            self._clear_cache()
        self._mean = mean

    @property
    def sd(self):
        return self._sd

    @sd.setter
    def sd(self, sd):
        if sd != self._sd:
            self._clear_cache()
        self._sd = sd

    def _lognorm_cdf(self, x, mean, sd):
        return stats.norm.cdf(np.log(x), loc=mean, scale=sd)

    def _lognorm_cdf_diff(self, x, mean, sd):
        return self._lognorm_cdf(x, mean, sd) - self._lognorm_cdf(x - 1.0, mean, sd)

    def _norm_pdf(self, x, mean, sd):
        return stats.norm.pdf(x, mean, sd)

    def _lognorm_cdf_diffs(self, mean, sd):
        lognorm_vals = self._lognorm_cdf(np.arange(0, self._len_history + 1), mean, sd)
        return lognorm_vals[1:] - lognorm_vals[:-1]

    def _lognorm_cdf_diffs_cache(self):
        if not self._cache.has_key("lognorm_cdf_diffs"):
            self._cache["lognorm_cdf_diffs"] = self._lognorm_cdf_diffs(self.mean, self.sd)
        return self._cache["lognorm_cdf_diffs"]

    def _pdf_diffs_cache(self):
        if not self._cache.has_key("pdf_diffs"):
            pdf_vals = self._norm_pdf(np.log(np.arange(0, self._len_history + 1)), self.mean, self.sd)
            self._cache["pdf_diffs"] = pdf_vals[1:] - pdf_vals[:-1]
        return self._cache["pdf_diffs"]

    def _scaled_pdf_diffs_cache(self):
        if not self._cache.has_key("scaled_pdf_diffs"):
            log_inds = np.log(np.arange(1, self._len_history + 1))
            scaled_pdf_vals = ((log_inds - self.mean) / self.sd) * self._norm_pdf(log_inds, self.mean, self.sd)
            scaled_pdf_vals[1:] = scaled_pdf_vals[1:] - scaled_pdf_vals[:-1]
            self._cache["scaled_pdf_diffs"] = scaled_pdf_vals
        return self._cache["scaled_pdf_diffs"]

    def _cite_diff_sum(self, lognorm_cdf_diffs):
        return np.dot(self._init_cum_hist_with_m, lognorm_cdf_diffs)

    def _cite_diff_sum_cache(self):
        if not self._cache.has_key("cite_diff_sum"):
            self._cache["cite_diff_sum"] = \
                self._cite_diff_sum(self._lognorm_cdf_diffs_cache())
        return self._cache["cite_diff_sum"]

    def _log_diff_sum(self, lognorm_cdf_diffs):
        prods = np.log(lognorm_cdf_diffs) * self._history
        prods[self._history == 0.0] = 0.0
        return np.sum(prods)

    def _log_diff_sum_cache(self):
        if not self._cache.has_key("log_diff_sum"):
            self._cache["log_diff_sum"] = self._log_diff_sum(self._lognorm_cdf_diffs_cache())
        return self._cache["log_diff_sum"]

    def _loglikelihood_helper(self, alpha, beta, cite_diff_sum, log_diff_sum):
        a = alpha
        b = beta
        N = self._N
        return self._const + log_diff_sum + a * np.log(b) - log_gamma(a) + \
               log_gamma(a + N) - (a + N) * np.log(b + cite_diff_sum)

    def loglikelihood(self, alpha = None, beta = None, mean = None, sd = None):
        if alpha is None:
            alpha = self.alpha
        if beta is None:
            beta = self.beta
        if mean is not None and sd is not None:
            lognorm_cdf_diffs = self._lognorm_cdf_diffs(mean, sd)
            cite_diff_sum = self._cite_diff_sum(lognorm_cdf_diffs)
            log_diff_sum = self._log_diff_sum(lognorm_cdf_diffs)
        elif mean is None and sd is None:
            cite_diff_sum = self._cite_diff_sum_cache()
            log_diff_sum = self._log_diff_sum_cache()
        else:
            raise Exception("Unimplemented.")
        return self._loglikelihood_helper(alpha, beta, cite_diff_sum, log_diff_sum)

    def _lambda_posterior_mean(self):
        return (self.alpha + self._N) / (self.beta + self._cite_diff_sum_cache())

    def alpha_gradient(self):
        a = self.alpha
        b = self.beta
        return np.log(b) - digamma(a) + digamma(a + self._N) - \
               np.log(b + self._cite_diff_sum_cache())

    def beta_gradient(self):
        return self.alpha / self.beta - self._lambda_posterior_mean()

    def _per_year_cite_over_diffs(self):
        lognorm_cdf_diffs = self._lognorm_cdf_diffs_cache()
        per_year_cite_over_diffs = self._history / lognorm_cdf_diffs
        per_year_cite_over_diffs[lognorm_cdf_diffs == 0] = np.Inf
        per_year_cite_over_diffs[self._history == 0] = 0.0
        return per_year_cite_over_diffs

    def mean_gradient(self):
        lam = self._lambda_posterior_mean()
        pdf_diffs = self._pdf_diffs_cache()
        per_year_cite_over_diffs = self._per_year_cite_over_diffs()
        lam_cumcite_diff_cite_delta = lam * self._init_cum_hist_with_m - per_year_cite_over_diffs
        return np.dot(pdf_diffs, lam_cumcite_diff_cite_delta)

    def sd_gradient(self):
        lam = self._lambda_posterior_mean()
        scaled_pdf_diffs = self._scaled_pdf_diffs_cache()
        per_year_cite_over_diffs = self._per_year_cite_over_diffs()
        lam_cumcite_diff_cite_delta = lam * self._init_cum_hist_with_m - per_year_cite_over_diffs
        return np.dot(scaled_pdf_diffs, lam_cumcite_diff_cite_delta)

    def _predict_n_years_fixed_lam(self, n, lam, size):
        preds = np.zeros((size, n))
        mean = self.mean
        sd = self.sd
        year_offset = self._len_history + 1
        for i in range(n):
            if i == 0:
                preds[:,i] = \
                    (1 + lam * self._lognorm_cdf_diff(i + year_offset, mean, sd)) * \
                    (self.m + self._N) - self.m
            else:
                preds[:,i] = \
                    (1 + lam * self._lognorm_cdf_diff(i + year_offset, mean, sd)) * \
                    (preds[:, i - 1] + self.m) - self.m
        return preds

    def predict_n_years(self, n):
        lams = np.random.gamma(self.alpha + self._N,
                               1.0 / (self.beta + self._cite_diff_sum_cache()),
                               500)
        preds = self._predict_n_years_fixed_lam(n, lams, len(lams))
        return np.mean(preds, axis = 0)


def _set_rpp_alpha_beta_and_optimize_mean_and_sd(rpp, alpha_beta, maxiter=10):
    rpp.alpha = alpha_beta[0]
    rpp.beta = alpha_beta[1]
    return RPPNet.optimize_rpp_mean_sd(rpp, maxiter)

def _rpp_predict_in_n_years(rpp, n):
    return rpp.predict_n_years(n)

class RPPNet(object):
    def __init__(self, gamma, model_save_path = None, maxiter=1000):
        self.maxiter = maxiter
        self._gamma = gamma
        self._scaler = None
        self._params = None
        self._sess = None
        self._is_fit = False
        if model_save_path is not None:
            self._model_save_path = os.path.abspath(model_save_path)
            if os.path.exists(self._tf_saver_file_path()):
                self._scaler = pickle.load(open(self._scaler_file_path(), "rb"))
                self._initialize_tf(len(self._scaler.mean_))
                self._is_fit = True
        else:
            self._model_save_path = model_save_path

    def __del__(self):
        if self._sess is not None:
            self._sess.close()

    def is_fit(self):
        return self._is_fit

    @staticmethod
    def optimize_rpp_mean_sd(rpp, maxiter=10):
        def obj(x):
            return -rpp.loglikelihood(alpha=rpp.alpha, beta=rpp.beta,
                                      mean=x[0], sd=x[1])
        def grad(x):
            rpp.mean = x[0]
            rpp.sd = x[1]
            return -np.array([rpp.mean_gradient(), rpp.sd_gradient()])

        result = optimize.minimize(
            obj,
            np.array([rpp.mean, rpp.sd]),
            method="L-BFGS-B",
            jac=grad,
            bounds=[(-1, None), (.1, None)],
            options={"disp": False, "maxiter": maxiter})
        if not result.success:
            Warning("RPP mean/sd optimization did not converge.")
        rpp.mean = result.x[0]
        rpp.sd = result.x[1]
        return rpp

    def _optimize_alpha_beta_params(self, params, grad, obj):
        result = optimize.minimize(obj,
                                   params,
                                   method="L-BFGS-B",
                                   jac=grad,
                                   options={"disp": False, "maxiter": 10})
        if not result.success:
            Warning("RPP alpha/beta optimization did not converge.")
        return result.x

    def _optimize_means_sds(self, rpps, alpha_beta):
        new_rpps = Parallel(n_jobs=multiprocessing.cpu_count())(
            delayed(_set_rpp_alpha_beta_and_optimize_mean_and_sd)(rpps[i],
                                                                  alpha_beta[i, :]) \
            for i in range(len(rpps))
        )
        return new_rpps

    def predict(self, X, histories, num_years):
        X = self._scaler.transform(X)

        rpps = [RPP(histories[i]) for i in range(len(histories))]
        sess = self._sess
        feed_dict = {self._X_ph: X,
                     self._N_ph: np.zeros(len(rpps)),
                     self._cite_diff_sum_ph: np.zeros(len(rpps)),
                     self._keep_prob_ph: 1.0,
                     self._const_wrt_alpha_beta_ph: 0,
                     self._gamma_ph: self._gamma}
        rpps = self._optimize_means_sds(rpps, sess.run(self._alpha_beta_op, feed_dict=feed_dict))

        preds = np.array(Parallel(n_jobs=multiprocessing.cpu_count())(
            delayed(_rpp_predict_in_n_years)(rpps[i], num_years) for i in range(len(rpps))))

        return preds.T

    def _initialize_tf(self, num_features):
        if self._sess is not None:
            self._sess.close()
        # Placeholders
        self._X_ph = tf.placeholder("float", [None, num_features])
        self._N_ph = tf.placeholder("float", [None])
        self._cite_diff_sum_ph = tf.placeholder("float", [None])
        self._const_wrt_alpha_beta_ph = tf.placeholder("float", [])
        self._gamma_ph = tf.placeholder("float", [])
        self._keep_prob_ph = tf.placeholder("float", [])
        # Operations
        self._alpha_beta_op = _rpp_multi_layer_alpha_beta(self._X_ph,
                                                          self._keep_prob_ph,
                                                          num_features)
        self._alpha_op = self._alpha_beta_op[:, 0]
        self._beta_op = self._alpha_beta_op[:, 1]
        self._loss_op = _rpp_loss(self._alpha_op, self._beta_op, self._N_ph,
                                  self._cite_diff_sum_ph, self._const_wrt_alpha_beta_ph, self._gamma_ph)
        self._train_op = _rpp_train(self._loss_op)
        self._saver = tf.train.Saver()
        self._init_op = tf.initialize_all_variables()
        # Start the session
        self._sess = tf.Session()
        if self._model_save_path is not None and \
                os.path.exists(self._tf_saver_file_path()):
            self._saver.restore(self._sess, self._tf_saver_file_path())
        else:
            self._sess.run(self._init_op)

    def _tf_saver_file_path(self):
        return self._model_save_path + ".tf"

    def _scaler_file_path(self):
        return self._model_save_path.split(".")[0] + "-scaler.pickle"

    def fit(self, X, histories):
        assert (X.shape[0] == len(histories))
        self._scaler = preprocessing.StandardScaler().fit(X)
        X = self._scaler.transform(X)
        self._initialize_tf(X.shape[1])

        rpps = [RPP(histories[i]) for i in range(len(histories))]

        k = 0
        N = np.array([rpp._N for rpp in rpps])
        const = np.mean([rpp._const for rpp in rpps])
        feed_dict_train = {self._X_ph: X, self._N_ph: N, self._keep_prob_ph: .95, self._gamma_ph: self._gamma}
        sess = self._sess
        while k == 0 or k < self.maxiter:
            cite_diff_sum = np.array([rpp._cite_diff_sum_cache() for rpp in rpps])
            mean_log_diff_sum = np.mean([rpp._log_diff_sum_cache() for rpp in rpps])

            feed_dict_train[self._cite_diff_sum_ph] = cite_diff_sum
            feed_dict_train[self._const_wrt_alpha_beta_ph] = const + mean_log_diff_sum
            feed_dict_test = feed_dict_train.copy()
            feed_dict_test[self._keep_prob_ph] = 1.0
            if k % 1 == 0:
                print "Iteration " + str(k)
                print "Current obj = " + str(sess.run(self._loss_op, feed_dict=feed_dict_test))
                #alpha_beta = sess.run(self._alpha_beta_op, feed_dict=feed_dict_test)
                #print "Current obj 2 = " + str(np.mean([rpps[i].value_new(alpha_beta[i,]) for i in range(len(rpps))]))
            print "Optimizing alpha/beta"
            for _ in range(1000):
                sess.run(self._train_op, feed_dict=feed_dict_train)
            print "Optimizing means/sds"
            rpps = self._optimize_means_sds(rpps, sess.run(self._alpha_beta_op, feed_dict=feed_dict_test))
            k += 1
        self._is_fit = True
        if self._model_save_path is not None:
            self._saver.save(sess, self._tf_saver_file_path())
            pickle.dump(self._scaler, open(self._scaler_file_path(), "wb"),
                        pickle.HIGHEST_PROTOCOL)