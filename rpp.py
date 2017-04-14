import cPickle as pickle
import multiprocessing
import os

import numpy as np
import scipy.optimize as optimize
import scipy.stats as stats
import tensorflow as tf
from joblib import Parallel, delayed
from scipy.special import gammaln as log_gamma
from scipy.special import psi as digamma
from sklearn import preprocessing
import warnings

np.seterr(all='warn')

CORES_TO_USE = multiprocessing.cpu_count()

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

def _rpp_one_layer_alpha_beta(x, keep_prob, num_features):
    #x_drop = tf.nn.dropout(x, keep_prob)
    return _fc_layer(x, num_features, 2, tf.nn.softplus, "out") + 0.001

def _rpp_one_layer_alpha_beta_bias_only(out_size):
    with tf.name_scope('layer_out'):
        alpha_beta = tf.nn.softplus(tf.Variable(tf.zeros([2]) + .1, name="b_out")) + .001
        alpha_beta = tf.reshape(alpha_beta, [1, 2])
        alpha_beta = tf.tile(alpha_beta, out_size)
    return alpha_beta

def _fc_layer(inputs, in_size, out_size, non_lin_func, id = ""):
    with tf.name_scope('layer_' + id):
        w = tf.Variable(
            tf.zeros([in_size, out_size]) + .1,
            name="w_" + id
        )
        b = tf.Variable(tf.zeros([out_size]) + .1, name="b_" + id)
        fc = non_lin_func(tf.matmul(inputs, w) + b)
    return fc

def _fc_relu_layer(inputs, in_size, out_size, id = ""):
    return _fc_layer(inputs, in_size, out_size, tf.nn.relu, id = id)

def _rpp_multi_layer_alpha_beta(x, keep_prob, num_features):
    fc_1 = _fc_relu_layer(x, num_features, 12, "1")
    fc_2 = _fc_relu_layer(fc_1, 12, 8, "2")
    fc_3 = _fc_relu_layer(fc_2, 8, 8, "3")
    fc_4 = _fc_relu_layer(fc_3, 8, 8, "4")
    fc_5 = tf.nn.dropout(_fc_relu_layer(fc_4, 8, 8, "5"), keep_prob)
    fc_out = _fc_layer(fc_5, 8, 2, tf.nn.softplus, "out") + .001
    return fc_out

def _rpp_train(loss):
    optimizer = tf.train.AdamOptimizer(learning_rate=.01)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    return optimizer.minimize(loss, global_step=global_step)

def _nan_or_inf(x):
    return np.isnan(x) or np.isinf(x)

def _line_search(init, obj, direction, lower_bounds, upper_bounds, eta):
    direction = eta * direction / np.linalg.norm(direction)

    ref_obj = obj(init)
    cur_point = init + direction
    iter = 0
    while iter < 20:
        if not np.any(np.logical_or(cur_point < lower_bounds, cur_point > upper_bounds)):
            cur_obj = obj(cur_point)
            if not _nan_or_inf(cur_obj) and cur_obj <= ref_obj:
                break
        iter += 1
        direction /= 2.0
        cur_point = init + direction

    if _nan_or_inf(obj(cur_point)):
        warnings.warn("Line search failed to find a valid point in bounds, returning" +
                      " the initial point (" + str(init) + ", with obj = " +
                      str(ref_obj))
        return init, ref_obj

    iter = 0
    last_obj = obj(cur_point)
    while iter < 20:
        direction *= 2.0
        cur_point = init + direction
        cur_obj = obj(cur_point)
        if (np.any(np.logical_or(cur_point < lower_bounds, cur_point > upper_bounds)) \
            or _nan_or_inf(cur_obj) or cur_obj > last_obj):
            direction /= 2.0
            break
        last_obj = cur_obj
        iter += 1

    return init + direction, obj(init + direction)

def _gradient_descent(init, obj, grad, lower_bounds, upper_bounds, maxiter=1000, tol=10**-6, eta=1.0):
    if np.any(init <= lower_bounds) or np.any(init >= upper_bounds):
        raise Exception("Initial point must be on interior of bounds.")
    last_obj = np.Inf
    cur_obj = obj(init)
    cur_point = init
    iter = 0
    while 1.0 - cur_obj / last_obj > tol and iter < maxiter:
        iter += 1
        cur_grad = grad(cur_point)
        last_obj = cur_obj
        cur_point, cur_obj = _line_search(cur_point, obj, -cur_grad,
                                          lower_bounds, upper_bounds,
                                          eta=eta)
    return cur_point

class RPP(object):
    def __init__(self, history):
        self._history = np.array(history, dtype=np.float64)
        self.alpha = 10.0
        self.beta = 12.0
        self.m = 10.0
        self._init_cum_hist_with_m = np.cumsum(np.hstack(((self.m,), history)))[:-1]

        history_span = self._history_span(history)
        self._mean, self._sd = self._mean_var_to_mu_sigma(
            np.mean(history_span),
            np.max([np.mean(history_span), 1]))
        self._cache = {}

        # Constants in the log-likelihood
        self._len_history = len(history)
        self._N = np.sum(history)
        self._c_const = np.dot(np.log(self._init_cum_hist_with_m), history)
        self._d_const = np.sum(log_gamma(history + 1))
        self._const = self._c_const - self._d_const

    def _history_span(self, history):
        which_non_zero = np.where(history != 0)[0] + 1
        if len(which_non_zero) == 0:
            return (1, len(history))
        else:
            return (np.min(which_non_zero), np.max(which_non_zero))

    def _mean_var_to_mu_sigma(self, mean, var):
        a = np.sqrt(mean**4 / (mean**2 + var))
        mu = np.log(a)
        sigma = np.sqrt(2) * np.sqrt(np.log(mean / a))
        return (mu, sigma)

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
        with np.errstate(divide='ignore'):
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
            with np.errstate(divide='ignore'):
                pdf_vals = np.array(self._norm_pdf(np.log(np.arange(0, self._len_history + 1)), self.mean, self.sd))
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
        with np.errstate(divide='ignore', invalid='ignore'):
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
        per_year_cite_over_diffs[self._history == 0.0] = 0.0
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


def _set_rpp_alpha_beta_and_optimize_mean_and_sd(rpp, alpha_beta, maxiter):
    rpp.alpha = alpha_beta[0]
    rpp.beta = alpha_beta[1]
    return RPPNet.optimize_rpp_mean_sd(rpp, maxiter)

def _set_rpp_alpha_beta(rpp, alpha_beta, maxiter):
    rpp.alpha = alpha_beta[0]
    rpp.beta = alpha_beta[1]
    return rpp

def _rpp_predict_in_n_years(rpp, n):
    return rpp.predict_n_years(n)

class RPPNet(object):
    def __init__(self, gamma, model_save_path = None, maxiter=10):
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
                self._initialize_tf(0 if self._scaler is None else len(self._scaler.mean_))
                self._is_fit = True
        else:
            self._model_save_path = model_save_path

    def __del__(self):
        if self._sess is not None:
            self._sess.close()

    def is_fit(self):
        return self._is_fit

    @staticmethod
    def optimize_rpp_mean_sd(rpp, maxiter):
        def obj(x):
            return -rpp.loglikelihood(alpha=rpp.alpha, beta=rpp.beta,
                                      mean=x[0], sd=x[1])
        def grad(x):
            rpp.mean = x[0]
            rpp.sd = x[1]
            return -np.array([rpp.mean_gradient(), rpp.sd_gradient()])

        m = max(rpp.mean, -.99)
        s = max(rpp.sd, .51)
        result = _gradient_descent(np.array([m, s]), obj, grad,
                              np.array([-1, .5]), np.array([np.Inf, np.Inf]), maxiter=maxiter, tol=10**-4, eta = .1)
        rpp.mean = result[0]
        rpp.sd = result[1]
        return rpp

    def _optimize_means_sds(self, rpps, alpha_beta, maxiter=1000):
        new_rpps = Parallel(n_jobs=CORES_TO_USE)(
            delayed(_set_rpp_alpha_beta_and_optimize_mean_and_sd)(rpps[i],
                                                                  alpha_beta[i, :],
                                                                  maxiter) \
            for i in range(len(rpps))
        )
        #new_rpps = [_set_rpp_alpha_beta_and_optimize_mean_and_sd(rpps[i], alpha_beta[i,:], maxiter) for i in range(len(rpps))]
        return new_rpps

    def predict(self, X, histories, num_years):
        if self._num_features != 0:
            X_scaled = self._scaler.transform(X)
        else:
            X_scaled = X.values

        rpps = [RPP(histories[i]) for i in range(len(histories))]
        sess = self._sess
        feed_dict = {self._X_ph: X_scaled,
                     self._N_ph: np.zeros(len(rpps)),
                     self._cite_diff_sum_ph: np.zeros(len(rpps)),
                     self._keep_prob_ph: 1.0,
                     self._const_wrt_alpha_beta_ph: 0,
                     self._gamma_ph: self._gamma}
        if X.shape[1] == 0:
            feed_dict[self._out_size] = np.array([X.shape[0], 2])
        rpps = self._optimize_means_sds(rpps, sess.run(self._alpha_beta_op, feed_dict=feed_dict), 10000)

        preds = np.array(Parallel(n_jobs=CORES_TO_USE)(
            delayed(_rpp_predict_in_n_years)(rpps[i], num_years) for i in range(len(rpps))))

        return preds

    def _initialize_tf(self, num_features):
        if self._sess is not None:
            self._sess.close()
        self._g = tf.Graph()
        with self._g.as_default() as g:
            with g.name_scope("g" + str(num_features)):
                self._num_features = num_features
                # Placeholders
                self._X_ph = tf.placeholder("float", [None, num_features])
                self._N_ph = tf.placeholder("float", [None])
                self._cite_diff_sum_ph = tf.placeholder("float", [None])
                self._const_wrt_alpha_beta_ph = tf.placeholder("float", [])
                self._gamma_ph = tf.placeholder("float", [])
                self._keep_prob_ph = tf.placeholder("float", [])
                # Operations
                if num_features != 0:
                    self._alpha_beta_op = _rpp_one_layer_alpha_beta(self._X_ph,
                                                                    self._keep_prob_ph,
                                                                    num_features)
                else:
                    self._out_size = tf.placeholder(tf.int32, [2])
                    self._alpha_beta_op = _rpp_one_layer_alpha_beta_bias_only(self._out_size)
                self._alpha_op = self._alpha_beta_op[:, 0]
                self._beta_op = self._alpha_beta_op[:, 1]
                self._loss_op = _rpp_loss(self._alpha_op, self._beta_op, self._N_ph,
                                          self._cite_diff_sum_ph,
                                          self._const_wrt_alpha_beta_ph, self._gamma_ph)
                self._train_op = _rpp_train(self._loss_op)
                self._saver = tf.train.Saver()
                self._init_op = tf.initialize_all_variables()
                self._bias = [v for v in tf.all_variables() if 'layer_out/b_out' in v.name][0]
        tf.reset_default_graph()
        # Start the session
        self._sess = tf.Session(graph=g)
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
        if X.shape[1] != 0:
            self._scaler = preprocessing.StandardScaler().fit(X)
            X = self._scaler.transform(X)
            self._initialize_tf(X.shape[1])
        else:
            X = X.values
            self._initialize_tf(0)

        rpps = [RPP(histories[i]) for i in range(len(histories))]

        k = 0
        N = np.array([rpp._N for rpp in rpps])
        const = np.mean([rpp._const for rpp in rpps])
        feed_dict_train = {self._X_ph: X, self._N_ph: N, self._keep_prob_ph: .95,
                           self._gamma_ph: self._gamma}
        if X.shape[1] == 0:
            feed_dict_train[self._out_size] = np.array([X.shape[0], 2])
        sess = self._sess
        cite_diff_sum = np.array([rpp._cite_diff_sum_cache() for rpp in rpps])
        mean_log_diff_sum = np.mean([rpp._log_diff_sum_cache() for rpp in rpps])
        feed_dict_train[self._cite_diff_sum_ph] = cite_diff_sum
        feed_dict_train[self._const_wrt_alpha_beta_ph] = const + mean_log_diff_sum
        feed_dict_test = feed_dict_train.copy()
        feed_dict_test[self._keep_prob_ph] = 1.0

        alpha_beta = sess.run(self._alpha_beta_op, feed_dict=feed_dict_test)
        for i in range(len(rpps)):
            rpps[i].alpha = alpha_beta[i, 0]
            rpps[i].beta = alpha_beta[i, 1]
        while k == 0 or k < self.maxiter:
            if k % 1 == 0:
                print "Iteration " + str(k)
                print "Current obj = " + str(sess.run(self._loss_op, feed_dict=feed_dict_test))
                #alpha_beta = sess.run(self._alpha_beta_op, feed_dict=feed_dict_test)
                #print "Current obj 2 = " + str(np.mean([rpps[i].value_new(alpha_beta[i,]) for i in range(len(rpps))]))

            print "Optimizing means/sds"
            rpps = self._optimize_means_sds(rpps,
                                            sess.run(self._alpha_beta_op,
                                                     feed_dict=feed_dict_test),
                                            10)
            cite_diff_sum = np.array([rpp._cite_diff_sum_cache() for rpp in rpps])
            mean_log_diff_sum = np.mean([rpp._log_diff_sum_cache() for rpp in rpps])
            feed_dict_train[self._cite_diff_sum_ph] = cite_diff_sum
            feed_dict_train[self._const_wrt_alpha_beta_ph] = const + mean_log_diff_sum
            feed_dict_test = feed_dict_train.copy()
            feed_dict_test[self._keep_prob_ph] = 1.0
            print "Optimizing alpha/beta"
            for i in range(1000):
                if i % 200 == 0:
                    print "TF iter " + str(i) + ", obj = " + str(sess.run(self._loss_op,
                                                                          feed_dict=feed_dict_test))
                    print "Current bias = " + str(sess.run(self._bias, feed_dict_test))
                sess.run(self._train_op, feed_dict=feed_dict_train)

            k += 1
        self._is_fit = True
        if self._model_save_path is not None:
            self._saver.save(sess, self._tf_saver_file_path())
            pickle.dump(self._scaler, open(self._scaler_file_path(), "wb"),
                        pickle.HIGHEST_PROTOCOL)