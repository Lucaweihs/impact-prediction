import pandas
import numpy as np
import cPickle as pickle
import os

def read_data(file_path, header=0):
    if os.path.exists(file_path + ".pickle"):
        with file(file_path + ".pickle", "r") as f:
            df = pickle.load(f)
    else:
        df = pandas.read_csv(file_path, sep="\t", header=header)
        with file(file_path + ".pickle", "w") as f:
            pickle.dump(df, f, pickle.HIGHEST_PROTOCOL)
    return df

def read_histories(file_path):
    if os.path.exists(file_path + ".pickle"):
         with file(file_path + ".pickle", "r") as f:
             return pickle.load(f)
    histories = []
    with open(file_path, "r") as f:
        for line in f:
            histories.append(np.array([int(i) for i in line.split(("\t"))]))
    with file(file_path + ".pickle", "w") as f:
        pickle.dump(histories, f, pickle.HIGHEST_PROTOCOL)
    return histories

def get_train_valid_test_data(data, train_inds, valid_inds, test_inds):
    train_data = data.iloc[train_inds]
    valid_data = data.iloc[valid_inds]
    test_data = data.iloc[test_inds]
    return (train_data, valid_data, test_data)

def get_train_valid_test_histories(histories, train_inds, valid_inds, test_inds):
    train_hists = [histories[i] for i in train_inds]
    valid_hists = [histories[i] for i in valid_inds]
    test_hists = [histories[i] for i in test_inds]
    return (train_hists, valid_hists, test_hists)

def get_train_valid_test_inds_from_config(config):
    train_inds = np.genfromtxt(config.train_inds_path, delimiter="\t", dtype=int)
    valid_inds = np.genfromtxt(config.valid_inds_path, delimiter="\t", dtype=int)
    test_inds = np.genfromtxt(config.test_inds_path, delimiter="\t", dtype=int)
    return (train_inds, valid_inds, test_inds)