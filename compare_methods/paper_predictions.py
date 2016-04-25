from test_runners import *
from config_reader import ConfigReader
import sys

assert(len(sys.argv) == 2)

config = ConfigReader("config.ini", "paper", "citation", int(sys.argv[1]))

runTests(config)
