from config_reader import ConfigReader
from data_manipulation import read_data
import numpy as np
import sys
from misc_functions import *

assert(len(sys.argv) >= 3 and len(sys.argv) <= 4)

if len(sys.argv) == 3:
    _, doc_type, measure = sys.argv
    filters_string = ""
else:
    _, doc_type, measure, filters_string = sys.argv

filters, filter_id = filters_string_to_filters(filters_string)

config = ConfigReader("config.ini", doc_type, measure, filter_id)

X = read_data(config.features_path)

allowed_inds_bool_array = np.repeat(True, X.shape[0])
for field, min_max in filters.iteritems():
    allowed_inds_bool_array &= (X[[field]].values[:, 0] >= min_max[0]) & (X[[field]].values[:, 0] <= min_max[1])

all_inds = np.where(allowed_inds_bool_array)[0]
sample_size = len(all_inds)
test_size = min(int(.2 * sample_size), 10000)
valid_size = min(int(.1 * sample_size), 10000)
train_size =  sample_size - test_size - valid_size

np.random.seed(12)
np.random.shuffle(all_inds)
train_inds = sorted(all_inds[0:train_size])
valid_inds = sorted(all_inds[train_size:(train_size + valid_size)])
test_inds = sorted(all_inds[(train_size + valid_size):len(all_inds)])

np.array(train_inds).tofile(config.train_inds_path, sep ="\t", format ='%d')
np.array(valid_inds).tofile(config.valid_inds_path, sep ="\t", format ='%d')
np.array(test_inds).tofile(config.test_inds_path, sep ="\t", format ='%d')