from models import *
from plotting import *
import os
import cPickle as pickle
import data_manipulation as dm
from error_tables import *
from misc_functions import *

def run_tests(config):
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

    print("Training constant model.\n")
    constant = ConstantModel(train_x, train_y, base_feature)

    print("Training optimal plus fixed k model.\n")
    fixed_k_optimal = PlusKBaselineModel(train_x, train_y, base_feature)

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

    print("Training gradient boost model.\n")
    if not os.path.exists("data/gb-" + pickle_suffix):
        gb = GradientBoostModel(train_x, train_y, base_feature)
        pickle.dump(gb, open("data/gb-" + pickle_suffix, "wb"), protocol)
    else:
        gb = pickle.load(open("data/gb-" + pickle_suffix, "rb"))

    baseline_models = [constant, fixed_k_optimal, simple_linear]
    ml_models = [lasso, rf, gb]

    if config.doc_type == "paper":
        print("Training RPPNet models.\n")
        rpp_suffix = pickle_suffix.split(".")[0]
        rpp_net = RPPNetWrapper(train_x, train_histories, train_y, "data/rpp-tf-" + rpp_suffix, maxiter=17, gamma=.1)
        rpp_net_without = RPPNetWrapper(train_x, train_histories, train_y, "data/rpp-tf-none-" + rpp_suffix, maxiter=4, gamma=.7, with_features=False)
        ml_models.insert(0, rpp_net)
        ml_models.insert(0, rpp_net_without)
        #rpp_with = RPPStub(config, train_x, valid_x, test_x)
        #rpp_without = RPPStub(config, train_x, valid_x, test_x, False)
        #ml_models.insert(0, rpp_with)
        #ml_models.insert(0, rpp_without)

    num_baseline = len(baseline_models)
    num_ml = len(ml_models)
    np.random.seed(23498)
    colors = cm.rainbow(np.linspace(0, 1, 12))
    markers = np.array(['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd'])
    np.random.shuffle(colors)
    np.random.shuffle(markers)
    baseline_colors = list(colors[range(num_baseline)])
    baseline_markers = list(markers[range(num_baseline)])
    ml_colors = list(colors[range(num_baseline, num_baseline + num_ml)])
    ml_markers = list(markers[range(num_baseline, num_baseline + num_ml)])

    pred_start_year = config.source_year + 1
    plot_suffix = config.full_suffix.replace(":", "_").replace(",", "_")


    # MAPE tables and plots
    if config.doc_type == "paper":
        rpp_net.set_prediction_histories(test_histories)
        rpp_net_without.set_prediction_histories(test_histories)
    mape_test_name = "mape-test-ml-" + plot_suffix
    mapes_df, errors_df = mape_table(ml_models, test_x, test_y, pred_start_year, mape_test_name)
    plot_mape(mapes_df, errors_df, mape_test_name, colors=ml_colors, markers=ml_markers)

    if config.doc_type == "paper":
        rpp_net.set_prediction_histories(valid_histories)
        rpp_net_without.set_prediction_histories(valid_histories)
    mape_valid_name = "mape-valid-ml-" + plot_suffix
    valid_mapes_df, valid_errors_df = mape_table(ml_models, valid_x, valid_y, pred_start_year, mape_valid_name)
    plot_mape(valid_mapes_df, valid_errors_df, mape_valid_name, colors=baseline_colors + ml_colors,
             markers=baseline_markers + ml_markers)

    if config.doc_type == "paper":
        rpp_net.set_prediction_histories(test_histories)
        rpp_net_without.set_prediction_histories(test_histories)
    top1_ml = n_arg_min(valid_mapes_df.values[:,-1], 1)
    models = baseline_models + list_inds(ml_models, top1_ml)
    colors = baseline_colors + list_inds(ml_colors, top1_ml)
    markers = baseline_markers + list_inds(ml_markers, top1_ml)
    mape_test_name = "mape-test-baseline-" + plot_suffix
    mapes_df, errors_df = mape_table(models, test_x, test_y, pred_start_year, mape_test_name)
    plot_mape(mapes_df, errors_df, mape_test_name, colors=colors, markers=markers)

    if config.doc_type == "paper":
        rpp_net.set_prediction_histories(train_histories)
        rpp_net_without.set_prediction_histories(train_histories)
    mape_train_name = "mape-train-" + plot_suffix
    mapes_df, errors_df = mape_table(baseline_models + ml_models, train_x, train_y, pred_start_year, mape_train_name)
    plot_mape(mapes_df, errors_df, mape_train_name, colors=baseline_colors + ml_colors,
             markers=baseline_markers + ml_markers)

    # PA-R^2 tables and plots
    if config.doc_type == "paper":
        rpp_net.set_prediction_histories(test_histories)
        rpp_net_without.set_prediction_histories(test_histories)
    rsq_test_name = "rsq-test-ml-" + plot_suffix
    rsq_df_map = rsquared_tables(ml_models, test_x, test_y, base_feature, pred_start_year, rsq_test_name)
    plot_r_squared(rsq_df_map["rsquare"], rsq_test_name, ml_colors, ml_markers)
    plot_r_squared(rsq_df_map["rsquare-inflated"], "inflated-" + rsq_test_name, ml_colors, ml_markers, xlabel="$R^2$")

    if config.doc_type == "paper":
        rpp_net.set_prediction_histories(valid_histories)
        rpp_net_without.set_prediction_histories(valid_histories)
    rsq_valid_name = "rsq-valid-ml-" + plot_suffix
    valid_rsq_df_map = rsquared_tables(ml_models, valid_x, valid_y, base_feature, pred_start_year, rsq_valid_name)

    if config.doc_type == "paper":
        rpp_net.set_prediction_histories(test_histories)
        rpp_net_without.set_prediction_histories(test_histories)
    rsq_test_name = "rsq-test-baseline-" + plot_suffix
    top1_ml = n_arg_max(valid_rsq_df_map["rsquare"].values[:, -1], 1)
    models = baseline_models + list_inds(ml_models, top1_ml)
    colors = baseline_colors + list_inds(ml_colors, top1_ml)
    markers = baseline_markers + list_inds(ml_markers, top1_ml)
    baseline_rsq_df_map = rsquared_tables(models, test_x, test_y, base_feature, pred_start_year, rsq_test_name)
    plot_r_squared(baseline_rsq_df_map["rsquare"], rsq_test_name, colors, markers)
    plot_r_squared(baseline_rsq_df_map["rsquare-inflated"], "inflated-" + rsq_test_name, colors, markers, xlabel="$R^2$")

    if config.doc_type == "paper":
        rpp_net.set_prediction_histories(train_histories)
        rpp_net_without.set_prediction_histories(train_histories)
    rsq_train_name = "rsq-train-" + plot_suffix
    rsq_df_map = rsquared_tables(baseline_models + ml_models, train_x, train_y, base_feature, pred_start_year, rsq_train_name)
    plot_r_squared(rsq_df_map["rsquare"], rsq_train_name, colors=baseline_colors + ml_colors,
             markers=baseline_markers + ml_markers)

    year = Y.shape[1]

    ape_scatter_file_name = "ape-" + config.full_suffix
    plot_ape_scatter(gb, test_x, test_y.values[:, year - 1], year, config.age_feature, ape_scatter_file_name, heat_map=False)
    plot_ape_scatter(gb, test_x, test_y.values[:, year - 1], year, config.age_feature, ape_scatter_file_name, heat_map=True)

    mape_plot_file_name = "mape_per_count_gb-" + config.full_suffix
    plot_mape_per_count(gb, test_x, test_y.values[:, year - 1], year, base_feature, mape_plot_file_name)

    mape_plot_file_name = "mape_per_age_gb-" + config.full_suffix
    plot_mape_per_count(gb, test_x, test_y.values[:, year - 1], year, config.age_feature, mape_plot_file_name)

    if config.doc_type == "paper":
        mape_plot_file_name = "mape_per_age_rpp-" + config.full_suffix
        rpp_net.set_prediction_histories(test_histories)
        plot_mape_per_count(rpp_net, test_x, test_y.values[:, year - 1], year, config.age_feature, mape_plot_file_name)