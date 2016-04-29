from test_runners import *
from config_reader import ConfigReader
import sys

assert(len(sys.argv) >= 2 and len(sys.argv) <= 3)

if len(sys.argv) == 2:
    config = ConfigReader("config.ini", "paper", "citation", int(sys.argv[1]))
else:
    config = ConfigReader("config.ini", "paper", "citation", int(sys.argv[1]), int(sys.argv[2]))

runTests(config)
