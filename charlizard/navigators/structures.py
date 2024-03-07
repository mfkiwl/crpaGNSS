'''
|========================================= structures.py ==========================================|
|                                                                                                  |
|   Property of Daniel Sturdivant. Unauthorized copying of this file via any medium would be       |
|   super sad and unfortunate for me. Proprietary and confidential.                                |
|                                                                                                  |
| ------------------------------------------------------------------------------------------------ |
|                                                                                                  |
|   @file     charlizard/navigators/structures.py                                                  |
|   @brief    Common structures used for various navigators.                                       |
|   @author   Daniel Sturdivant <sturdivant20@gmail.com>                                           |
|   @date     February 2024                                                                        |
|                                                                                                  |
|==================================================================================================|
'''

import numpy as np
from navtools.signals.signals import SatelliteSignal
from dataclasses import dataclass

# === Receiver State ===============================================================================
@dataclass
class RxState:
  pos: np.ndarray         #! receiver ecef position
  vel: np.ndarray         #! receiver ecef velocity
  clock_bias: float       #! receiver clock bias
  clock_drift: float      #! receiver clock drift

@dataclass
class RxCovariance:
  pos: np.ndarray         #! receiver ecef position covariance
  vel: np.ndarray         #! receiver ecef velocity covariance
  clock_bias: float       #! receiver clock bias variance
  clock_drift: float      #! receiver clock drift variance
  
  
# === Navigator Configs ============================================================================
@dataclass
class VDFLLConfig:
  T: float                        #! integration period [s]
  order: int                      #! 2: constant velocity, 3: constant acceleration
  process_noise_stdev: float      #! x,y,z process noise standard deviation
  tap_spacing: float              #! early,prompt,late correlator tap/chip spacing
  innovation_stdev: float         #! normalized innovation filter threshold
  cn0_buffer_len: int             #! number of correlator outputs to use in cn0 estimation
  cn0: np.ndarray                 #! initial receiver cn0
  pos: np.ndarray                 #! initial receiver ecef position
  vel: np.ndarray                 #! initial receiver ecef velocity
  clock_bias: float               #! initial receiver clock bias
  clock_drift: float              #! initial receiver clock drift
  clock_type: str                 #! receiver oscillator type
  is_signal_level: bool           #! state if operating at a signal level or correlator level
  
@dataclass
class GNSSINSConfig:
  T: float                        #! integration period [s]
  tap_spacing: float              #! early,prompt,late correlator tap/chip spacing
  innovation_stdev: float         #! normalized innovation filter threshold
  cn0_buffer_len: int             #! number of correlator outputs to use in cn0 estimation
  cn0: np.ndarray                 #! initial receiver cn0
  pos: np.ndarray                 #! initial receiver ecef position
  vel: np.ndarray                 #! initial receiver ecef velocity
  att: np.ndarray                 #! initial receiver roll, pitch, yaw [deg]
  clock_bias: float               #! initial receiver clock bias
  clock_drift: float              #! initial receiver clock drift
  clock_type: str                 #! receiver oscillator type
  imu_model: str                  #! IMU specifications
  coupling: str                   #! navigator coupling scheme ('tight' or 'deep')

  