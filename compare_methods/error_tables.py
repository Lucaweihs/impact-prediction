import numpy as np
import pandas as pd

def mape_table(models, X, Y, starting_year = 1, name = None):
    num_years = Y.shape[1]
    pred_years = range(starting_year, num_years + starting_year)

    mapes_table = np.zeros((len(models), num_years))
    errors_table = np.zeros((len(models), num_years))
    for i in range(len(models)):
        mapes_table[i, :], errors_table[i, :] = models[i].mape_all_with_errors(X, Y)
    mapes_df = pd.DataFrame(data=mapes_table, index=[m.name for m in models], columns=pred_years)
    errors_df = pd.DataFrame(data=errors_table, index=[m.name for m in models], columns=pred_years)
    if name != None:
        mapes_df.to_csv("tables/" + name + ".tsv", sep="\t")
        errors_df.to_csv("tables/" + "errors-" + name + ".tsv", sep="\t")
    return (mapes_df, errors_df)

def rsquared_tables(models, X, Y, base_feature, starting_year = 1, name = None, remove_outliers = False):
    base_values = X[[base_feature]].values[:, 0]

    num_years = Y.shape[1]
    pred_years = range(starting_year, num_years + starting_year)
    errors_list = []
    flawed_errors_list = []
    inds_to_remove = set()
    for i in range(len(models)):
        preds = models[i].predict_all(X)
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

    r2_table = np.zeros((len(models), num_years))
    r2_inflated_table = np.zeros((len(models), num_years))
    r2_flawed_table = np.zeros((len(models), num_years))
    for i in range(len(models)):
        errors_per_year = np.delete(errors_list[i], inds_to_remove, axis=0)
        error_means = np.mean(errors_per_year, axis=0)
        r2_table[i, :] = 1.0 - error_means / base_errors
        r2_inflated_table[i, :] = 1.0 - error_means / base_errors_inflated
        r2_flawed_table[i, :] = np.mean(flawed_errors_list[i], axis=0) / base_errors_inflated

    model_names = [m.name for m in models]
    r2_df = pd.DataFrame(data=r2_table, index=model_names, columns=pred_years)
    r2_inflated_df = pd.DataFrame(data=r2_inflated_table, index=model_names, columns=pred_years)
    r2_flawed_df = pd.DataFrame(data=r2_flawed_table, index=model_names, columns=pred_years)

    if name is not None:
        r2_df.to_csv("tables/" + remove_string + name + ".tsv", sep="\t")
        r2_inflated_df.to_csv("tables/" + "inflated-" + remove_string + name + ".tsv", sep="\t")
        r2_flawed_df.to_csv("tables/" + "flawed-" + remove_string + name + ".tsv", sep="\t")
    return {"rsquare" : r2_df, "rsquare-inflated" : r2_inflated_df, "rsquare-flawed" : r2_flawed_df}