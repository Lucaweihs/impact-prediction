from test_runners import *
from config_reader import ConfigReader
import sys
from misc_functions import filtersStringToFilters

assert(len(sys.argv) >= 2 and len(sys.argv) <= 3)

if len(sys.argv) == 2:
    _, measure = sys.argv
    filtersString = ""
else:
    _, measure, filtersString = sys.argv

filters, filterId = filtersStringToFilters(filtersString)

config = ConfigReader("config.ini", "paper", measure, filterId)

if not os.path.exists(config.trainIndsPath):
    command = "python split_data.py paper " + measure + ' "' + filtersString + '"'
    os.system(command)

runTests(config)
