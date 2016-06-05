from test_runners import *
import sys
from config_reader import ConfigReader
import os
from misc_functions import filters_string_to_filters

assert(len(sys.argv) >= 2 and len(sys.argv) <= 3)

if len(sys.argv) == 2:
    _, measure = sys.argv
    filters_string = ""
else:
    _, measure, filters_string = sys.argv

filters, filter_id = filters_string_to_filters(filters_string)

config = ConfigReader("config.ini", "author", measure, filter_id)

if not os.path.exists(config.train_inds_path):
    command = "python split_data.py author " + measure + ' "' + filters_string + '"'
    os.system(command)

run_tests(config)

