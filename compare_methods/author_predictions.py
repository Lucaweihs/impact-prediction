from test_runners import *
import sys
from config_reader import ConfigReader

assert(len(sys.argv) == 2)

config = ConfigReader("config.ini", "author", sys.argv[1].lower())

runTests(config)

