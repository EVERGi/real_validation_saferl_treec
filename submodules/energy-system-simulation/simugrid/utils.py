import os, sys

# Definition of paths
PROJ_DIR = (os.path.dirname(os.path.realpath(sys.argv[0])) + "/").replace("\\", "/")
SRC_DIR = (os.path.dirname(os.path.abspath(__file__))+"/").replace("\\", "/")
ROOT_DIR = (os.path.dirname(SRC_DIR[:-1])+"/").replace("\\", "/")
DATA_DIR = (ROOT_DIR+"data/").replace("\\", "/")
