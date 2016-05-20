from test_runners import *
import sys
from config_reader import ConfigReader
import os
from misc_functions import filtersStringToFilters

assert(len(sys.argv) >= 2 and len(sys.argv) <= 3)

if len(sys.argv) == 2:
    _, measure = sys.argv
    filtersString = ""
else:
    _, measure, filtersString = sys.argv

filters, filterId = filtersStringToFilters(filtersString)

config = ConfigReader("config.ini", "author", measure, filterId)

if not os.path.exists(config.trainIndsPath):
    command = "python split_data.py author " + measure + ' "' + filtersString + '"'
    os.system(command)

runTests(config)

