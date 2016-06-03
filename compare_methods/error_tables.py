import numpy as np
import pandas as pd
from misc_functions import mape_with_error

def mape_tables_for_models(model_names, preds_map, y_map, starting_year = 1, suffix =""):
    data_set_types = ["train", "valid", "test"]

    tables = {}
    for data_set_type in data_set_types:
        preds_list = [preds_map[k][data_set_type] for k in model_names]
        name = "mape-" + data_set_type + ("-" + suffix if suffix != "" else "")
        tables[data_set_type] = mape_table(model_names, preds_list,
                                           y_map[data_set_type], starting_year, name)
    return tables

def mape_table(model_names, preds_list, Y,
               starting_year = 1, name = None):
    num_years = Y.shape[1]
    pred_years = range(starting_year, num_years + starting_year)

    mapes_table = np.zeros((len(model_names), num_years))
    errors_table = np.zeros((len(model_names), num_years))
    for i in range(len(model_names)):
        mapes_table[i, :], errors_table[i, :] = \
            mape_with_error(preds_list[i], Y.values)
    mapes_df = pd.DataFrame(data=mapes_table, index=model_names, columns=pred_years)
    errors_df = pd.DataFrame(data=errors_table, index=model_names, columns=pred_years)
    if name is not None:
        mapes_df.to_csv("tables/" + name + ".tsv", sep="\t")
        errors_df.to_csv("tables/" + "errors-" + name + ".tsv", sep="\t")
    return (mapes_df, errors_df)

def rsquared_tables_for_models(model_names, preds_map, y_map, base_values_map,
                               starting_year = 1, suffix = ""):
    data_set_types = ["train", "valid", "test"]

    pa_rsq_tables = {}
    rsq_tables = {}
    for data_set_type in data_set_types:
        preds_list = [preds_map[k][data_set_type] for k in model_names]
        rsq_map = rsquared_tables(model_names, preds_list, y_map[data_set_type],
                                                base_values_map[data_set_type], starting_year, suffix)

        pa_rsq_tables[data_set_type] = rsq_map["rsquare"]
        rsq_tables[data_set_type] = rsq_map["rsquare-inflated"]
    return (pa_rsq_tables, rsq_tables)


def rsquared_tables(model_names, preds_list, Y, base_values, starting_year = 1,
                    suffix = None, remove_outliers = False):
    num_years = Y.shape[1]
    pred_years = range(starting_year, num_years + starting_year)
    errors_list = []
    flawed_errors_list = []
    inds_to_remove = set()
    for i in range(len(model_names)):
        preds = preds_list[i]
        errors_per_year = np.square(preds - Y.values)
        inds_to_remove |= set([np.argmax(errors_per_year[:, i]) for i in range(num_years)])
        errors_list.append(errors_per_year)
        flawed_errors_list.append(np.square(preds - np.mean(Y.values, axis=0)))

    if remove_outliers:
        inds_to_remove = list(inds_to_remove)
        remove_string = "removed-"
        print "Inds removed: " + str(inds_to_remove)
    else:
        inds_to_remove = []
        remove_string = ""

    base_errors = np.var(np.delete((Y.values.T - base_values).T, inds_to_remove, axis=0), axis=0)
    base_errors_inflated = np.var(np.delete(Y.values, inds_to_remove, axis=0), axis=0)

    r2_table = np.zeros((len(model_names), num_years))
    r2_inflated_table = np.zeros((len(model_names), num_years))
    r2_flawed_table = np.zeros((len(model_names), num_years))
    for i in range(len(model_names)):
        errors_per_year = np.delete(errors_list[i], inds_to_remove, axis=0)
        error_means = np.mean(errors_per_year, axis=0)
        r2_table[i, :] = 1.0 - error_means / base_errors
        r2_inflated_table[i, :] = 1.0 - error_means / base_errors_inflated
        r2_flawed_table[i, :] = np.mean(flawed_errors_list[i], axis=0) / base_errors_inflated

    r2_df = pd.DataFrame(data=r2_table, index=model_names, columns=pred_years)
    r2_inflated_df = pd.DataFrame(data=r2_inflated_table, index=model_names, columns=pred_years)
    r2_flawed_df = pd.DataFrame(data=r2_flawed_table, index=model_names, columns=pred_years)

    if suffix is not None:
        r2_df.to_csv("tables/rsq-" + remove_string + suffix + ".tsv", sep="\t")
        r2_inflated_df.to_csv("tables/" + "inflated-rsq-" + remove_string + suffix + ".tsv", sep="\t")
        r2_flawed_df.to_csv("tables/" + "flawed-rsq-" + remove_string + suffix + ".tsv", sep="\t")
    return {"rsquare" : r2_df, "rsquare-inflated" : r2_inflated_df, "rsquare-flawed" : r2_flawed_df}