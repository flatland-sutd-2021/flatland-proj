import os
import sys

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
STARTER_PATH = os.path.join(DIR_PATH, "..", "neurips2020-flatland-starter-kit")
sys.path.append(os.path.normpath(STARTER_PATH))