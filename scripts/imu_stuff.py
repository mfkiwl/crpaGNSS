
import numpy as np
from tqdm import tqdm
from pathlib import Path
from log_utils import *

from navtools.constants import SPEED_OF_LIGHT
from navtools.conversions.coordinates import ecef2lla, ecef2enu, ecef2enuv, ecef2enuDcm
from navsim.configuration import get_configuration
from navsim.simulations.ins import INSSimulation

PROJECT_PATH = Path(__file__).parents[1]
CONFIG_PATH = PROJECT_PATH / "configs"
DATA_PATH = PROJECT_PATH / "data"
RESULTS_PATH = PROJECT_PATH / "results"

DISABLE_PROGRESS = False

config = get_configuration(CONFIG_PATH)

ins_sim = INSSimulation(config)
ins_sim.motion_commands(DATA_PATH)
ins_sim.simulate()

print()