from test_runners import *
import sys
from config_reader import ConfigReader

assert(len(sys.argv) == 3)

config = ConfigReader("config.ini", "author",
                      sys.argv[1].lower(), int(sys.argv[2]))

runTests(config)

