from models import *
from plotting import *
import os
import cPickle as pickle
import data_manipulation as dm
from error_tables import *
from misc_functions import *

def rsq_err_worst_and_best_removed(preds, base_values, Y, num_worst_remove, num_best_remove):
    num_years = Y.shape[1]
    errors_per_year = np.square(preds - Y.values)

    inds_to_remove_per_year = []
    for i in range(num_years):
        baseline_errors = Y.values[:,i] - base_values
        baseline_errors = np.square(baseline_errors - np.mean(baseline_errors))

        best_inds = n_arg_max(baseline_errors * (1.0 - errors_per_year[:, i] / baseline_errors), num_best_remove)
        worst_inds = n_arg_max(errors_per_year[:, i], num_worst_remove)
        inds_to_remove_per_year.append(list(set(best_inds + worst_inds)))

    r2_errors = np.zeros(num_years)
    for i in range(num_years):
        base_error = np.var(np.delete(Y.values[:,i] - base_values, inds_to_remove_per_year[i]))
        errors = np.delete(errors_per_year[:, i], inds_to_remove_per_year[i])
        error_mean = np.mean(errors)
        r2_errors[i] = 1.0 - error_mean / base_error

    return r2_errors

def train_test_valid_predictions(models, train_x, valid_x, test_x,
                                 train_histories, valid_histories, test_histories):
    preds_map = {}
    for model in models:
        name = model.name
        preds_map[name] = {}

        # Train predictions
        if "rpp" in name.lower():
           model.set_prediction_histories(train_histories)
        preds_map[name]["train"] = model.predict_all(train_x)

        # Valid predictions
        if "rpp" in name.lower():
            model.set_prediction_histories(valid_histories)
        preds_map[name]["valid"] = model.predict_all(valid_x)

        # Test predictions
        if "rpp" in name.lower():
            model.set_prediction_histories(test_histories)
        preds_map[name]["test"] = model.predict_all(test_x)
    return preds_map

def model_names(models):
    return [model.name for model in models]

def run_tests(config):
    print("Reading data.\n")
    X = dm.read_data(config.features_path)
    Y = dm.read_data(config.responses_path)
    Y = Y.select(lambda x: config.measure in x.lower(), axis=1)
    histories = dm.read_histories(config.history_path)
    for history in histories:
        history[1:] = history[1:] - history[:-1]

    train_inds, valid_inds, test_inds = dm.get_train_valid_test_inds_from_config(config)
    train_x, valid_x, test_x = dm.get_train_valid_test_data(X, train_inds, valid_inds, test_inds)
    train_y, valid_y, test_y = dm.get_train_valid_test_data(Y, train_inds, valid_inds, test_inds)
    train_histories, valid_histories, test_histories = \
        dm.get_train_valid_test_histories(histories, train_inds, valid_inds, test_inds)
    pickle_suffix = config.full_suffix + ".pickle"
    base_feature = config.base_feature
    delta_feature = config.delta_feature

    protocol = pickle.HIGHEST_PROTOCOL

    print("Training optimal plus fixed k model.")
    fixed_k_optimal = PlusKBaselineModel(train_x, train_y, base_feature)
    print "Plus-k model has constant k = " + str(fixed_k_optimal.k) + "\n"

    print("Training simple model.\n")
    if not os.path.exists("data/simple_linear-" + pickle_suffix):
        simple_linear = SimpleLinearModel(train_x, train_y, base_feature, delta_feature)
        pickle.dump(simple_linear, open("data/simple_linear-" + pickle_suffix, "wb"), protocol)
    else:
        simple_linear = pickle.load(open("data/simple_linear-" + pickle_suffix, "rb"))

    print("Training lasso model.\n")
    if not os.path.exists("data/lasso-" + pickle_suffix):
        lasso = LassoModel(train_x, train_y, base_feature)
        pickle.dump(lasso, open("data/lasso-" + pickle_suffix, "wb"), protocol)
    else:
        lasso = pickle.load(open("data/lasso-" + pickle_suffix, "rb"))

    print("Training random forest model.\n")
    rf_path = "data/rf-" + pickle_suffix + ".bz2"
    if not os.path.exists(rf_path):
        rf = RandomForestModel(train_x, train_y, base_feature)
        dump_pickle_with_zip(rf, rf_path)
    else:
        rf = read_pickle_with_zip(rf_path)
    rf.set_verbose(0)

    print("Training gradient boost model.\n")
    if not os.path.exists("data/gb-" + pickle_suffix):
        gb = GradientBoostModel(train_x, train_y, base_feature, tune_with_cv=True)
        pickle.dump(gb, open("data/gb-" + pickle_suffix, "wb"), protocol)
    else:
        gb = pickle.load(open("data/gb-" + pickle_suffix, "rb"))

    baseline_models = [fixed_k_optimal, simple_linear]
    ml_models = [lasso, rf, gb]

    if config.doc_type == "paper":
        print("Training RPPNet models.\n")
        rpp_suffix = pickle_suffix.split(".")[0]
        rpp_net = RPPNetWrapper(train_x, train_histories, train_y, "data/rpp-tf-" + rpp_suffix, maxiter=17, gamma=.1)
        rpp_net_without = RPPNetWrapper(train_x, train_histories, train_y, "data/rpp-tf-none-" + rpp_suffix, maxiter=4, gamma=.7, with_features=False)
        ml_models.insert(0, rpp_net)
        ml_models.insert(0, rpp_net_without)

    print("Generating predictions for training, validation, and test sets.\n")
    all_models = ml_models + baseline_models
    all_preds = train_test_valid_predictions(all_models, train_x, valid_x, test_x,
                                             train_histories, valid_histories, test_histories)

    print("Generating MAPE tables.\n")
    pred_start_year = config.source_year + 1
    plot_suffix = config.full_suffix.replace(":", "_").replace(",", "_")
    all_y = {"train" : train_y, "valid" : valid_y, "test" : test_y}
    all_model_names = [a.name for a in all_models]
    ml_model_names = [a.name for a in ml_models]
    baseline_model_names = [a.name for a in baseline_models]
    mape_tables_map = mape_tables_for_models(all_model_names, all_preds, all_y, pred_start_year, plot_suffix)

    print("Generating MAPE plots.\n")
    num_baseline = len(baseline_models)
    num_ml = len(ml_models)
    np.random.seed(23498)
    colors = cm.rainbow(np.linspace(0, 1, 12))
    markers = np.array(["o", "v", "^", "<", ">", "8", "s", "p", "*", "h", "H", "D", "d"])
    np.random.shuffle(colors)
    np.random.shuffle(markers)
    baseline_colors = list(colors[range(num_baseline)])
    baseline_markers = list(markers[range(num_baseline)])
    ml_colors = list(colors[range(num_baseline, num_baseline + num_ml)])
    ml_markers = list(markers[range(num_baseline, num_baseline + num_ml)])

    mape_test_name = "mape-test-ml-" + plot_suffix
    plot_mape(mape_tables_map["test"][0].loc[ml_model_names],
              mape_tables_map["test"][1].loc[ml_model_names],
              mape_test_name, colors=ml_colors, markers=ml_markers)

    mape_valid_name = "mape-valid-ml-" + plot_suffix
    plot_mape(mape_tables_map["valid"][0].loc[ml_model_names],
              mape_tables_map["valid"][1].loc[ml_model_names],
              mape_valid_name, colors=ml_colors, markers=ml_markers)

    top1_ml = n_arg_min(mape_tables_map["valid"][0].values[:,-1], 1)
    model_names = baseline_model_names + list_inds(ml_model_names, top1_ml)
    colors = baseline_colors + list_inds(ml_colors, top1_ml)
    markers = baseline_markers + list_inds(ml_markers, top1_ml)
    mape_test_name = "mape-test-baseline-" + plot_suffix
    plot_mape(mape_tables_map["test"][0].loc[model_names],
              mape_tables_map["test"][1].loc[model_names],
              mape_test_name, colors=colors, markers=markers)

    mape_train_name = "mape-train-ml-" + plot_suffix
    plot_mape(mape_tables_map["train"][0].loc[all_model_names],
              mape_tables_map["train"][1].loc[all_model_names],
              mape_train_name, colors=baseline_colors + ml_colors,
              markers=baseline_markers + ml_markers)

    print("Generating R^2 tables.\n")
    base_values_map = {"train" : train_x[[base_feature]].values[:,0],
                       "valid" : valid_x[[base_feature]].values[:,0],
                       "test" : test_x[[base_feature]].values[:,0]}
    pa_rsq_map, rsq_map = rsquared_tables_for_models(all_model_names, all_preds, all_y,
                                                base_values_map, pred_start_year,
                                                plot_suffix)

    print("Generating R^2 / PA-R^2 plots.\n")
    # PA-R^2 tables and plots
    rsq_test_name = "rsq-test-ml-" + plot_suffix
    plot_r_squared(pa_rsq_map["test"].loc[ml_model_names], "pa-" + rsq_test_name,
                   ml_colors, ml_markers)
    plot_r_squared(rsq_map["test"].loc[ml_model_names], "regular-" + rsq_test_name,
                   ml_colors, ml_markers, xlabel="$R^2$")

    rsq_test_name = "rsq-test-baseline-" + plot_suffix
    top1_ml = n_arg_max(pa_rsq_map["test"].loc[ml_model_names].values[:, -1], 1)
    model_names = baseline_model_names + list_inds(ml_model_names, top1_ml)
    colors = baseline_colors + list_inds(ml_colors, top1_ml)
    markers = baseline_markers + list_inds(ml_markers, top1_ml)
    plot_r_squared(pa_rsq_map["test"].loc[model_names], "pa-" + rsq_test_name,
                   colors, markers)
    plot_r_squared(rsq_map["test"].loc[model_names], "regular-" + rsq_test_name,
                   colors, markers, xlabel="$R^2$")

    rsq_train_name = "pa-rsq-train-" + plot_suffix
    plot_r_squared(pa_rsq_map["test"].loc[all_model_names], "pa-" + rsq_train_name,
                   colors=baseline_colors + ml_colors,
                   markers=baseline_markers + ml_markers)

    if config.doc_type == "paper":
        model_names = ["SM", "LAS", "RPPNet", "RPP", "GBRT"]
        best_worst_parsq = pa_rsq_map["test"].loc[model_names].copy()
        for name in model_names:
            best_worst_parsq.loc[name] = rsq_err_worst_and_best_removed(all_preds[name]["test"],
                                                                        base_values_map["test"], test_y, 50, 50)
        plot_r_squared(best_worst_parsq, "outlier-removed-pa-rsq-test-" + plot_suffix,
                       colors=[baseline_colors[1]] + [ml_colors[i] for i in [2, 1, 0, 4]],
                       markers=[baseline_markers[1]] + [ml_markers[i] for i in [2, 1, 0, 4]])

    print("Median Absolute % Error of GBRT at 10 years:")
    last_year = Y.shape[1]
    gbrt_preds = all_preds["GBRT"]["test"][:, last_year - 1]
    gbrt_errors = np.abs((gbrt_preds - test_y.values[:, last_year - 1]) / test_y.values[:, last_year - 1])
    print(str(np.median(gbrt_errors)) + "\n")
    print("Mean Absolute % Error of GBRT at 10 years:")
    print(str(np.mean(gbrt_errors)) + "\n")

    print("Generating mape per count/age plots.\n")
    ape_scatter_file_name = "ape-" + config.full_suffix
    plot_ape_scatter(all_preds["GBRT"]["test"][:, last_year - 1], test_x[[config.age_feature]].values[:,0],
                     test_y.values[:, last_year - 1], config.age_feature, ape_scatter_file_name, heat_map=False)
    plot_ape_scatter(all_preds["GBRT"]["test"][:, last_year - 1], test_x[[config.age_feature]].values[:, 0],
                     test_y.values[:, last_year - 1], config.age_feature, ape_scatter_file_name, heat_map=True)

    mape_plot_file_name = "mape_per_count_gb-" + config.full_suffix
    plot_mape_per_count(all_preds["GBRT"]["test"][:, last_year - 1], test_x[[config.base_feature]].values[:,0],
                     test_y.values[:, last_year - 1], config.base_feature, mape_plot_file_name)

    mape_plot_file_name = "mape_per_age_gb-" + config.full_suffix
    plot_mape_per_count(all_preds["GBRT"]["test"][:, last_year - 1], test_x[[config.age_feature]].values[:,0],
                     test_y.values[:, last_year - 1], config.age_feature, mape_plot_file_name)

    if config.doc_type == "paper":
        mape_plot_file_name = "mape_per_age_rpp-" + config.full_suffix
        plot_mape_per_count(all_preds["RPPNet"]["test"][:, last_year - 1], test_x[[config.age_feature]].values[:,0],
                     test_y.values[:, last_year - 1], config.age_feature, mape_plot_file_name)