from test_runners import *
import sys
from config_reader import ConfigReader
import os

assert(len(sys.argv) >= 4 and len(sys.argv) <= 5)

if len(sys.argv) == 4:
    config = ConfigReader("config.ini", "author",
                          sys.argv[1].lower(), int(sys.argv[2]), int(sys.argv[3]))
    minBaseString = ""
else:
    config = ConfigReader("config.ini", "author",
                          sys.argv[1].lower(), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    minBaseString = " " + str(config.minBase)

if not os.path.exists(config.trainIndsPath):
    command = "python split_data.py " + config.docType + " " + config.measure + \
              " " + str(config.minNumCitations) + " " + str(config.minAge) + minBaseString
    os.system(command)

runTests(config)

