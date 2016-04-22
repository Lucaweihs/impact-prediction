from test_runners import *
from config_reader import ConfigReader

config = ConfigReader("config.ini", "paper", "citation")

runTests(config)
