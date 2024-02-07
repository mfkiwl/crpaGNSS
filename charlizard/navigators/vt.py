'''
|============================================= vt.py ==============================================|
|                                                                                                  |
|   Property of Daniel Sturdivant. Unauthorized copying of this file via any medium would be       |
|   super sad and unfortunate for me. Proprietary and confidential.                                |
|                                                                                                  |
| ------------------------------------------------------------------------------------------------ |
|                                                                                                  |
|   @file     charlizard/navigators/vt.py                                                          |
|   @brief    Vector tracking (VDFLL + PLL) architecture.                                          |
|   @author   Daniel Sturdivant <sturdivant20@gmail.com>                                           |
|   @date     February 2024                                                                        |
|                                                                                                  |
|==================================================================================================|
'''

import numpy as np
from logutils import default_logger as _log

from charlizard.navigators.structures import VDFLLConfig, Correlators
from charlizard.models.discriminator import *
from charlizard.models.lock_detectors import *

from navsim.error_models.clock import get_clock_allan_variance_values

from navtools.conversions.coordinates import ecef2enuDcm, ecef2enu, ecef2enuv, ecef2lla
from navtools.common import compute_range_and_unit_vector, compute_range_rate
from navtools.constants import *

# easy conversions
R2D, D2R = 180 / np.pi, np.pi / 180
lla_rad2deg = np.array([R2D, R2D, 1.0], dtype=float)
I3, Z33, Z32, Z23 = np.eye(3), np.zeros((3,3)), np.zeros((3,2)), np.zeros((2,3))

class VectorTrack:
  def __init__(self, config: VDFLLConfig):
    # grab config
    self._is_signal_level = config.is_signal_level
    self._clock_config = get_clock_allan_variance_values(config.clock_type)
    self._tap_spacing = config.tap_spacing
    self._innovation_stdev = config.innovation_stdev
    
    # signal settings
    # sig_config = config.signal_config.get('gps'.casefold())
    # self._chip_width = SPEED_OF_LIGHT / sig_config.fchip_data
    # self._wavelength = SPEED_OF_LIGHT / sig_config.fcarrier
    self._chip_width = SPEED_OF_LIGHT / 1.023e6
    self._wavelength = SPEED_OF_LIGHT / 1575.42e6
    
    # correlators
    self._correlators = Correlators(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    # cn0 estimation config
    self.cn0 = config.cn0
    self._cn0_counter = 1
    self._cn0_buffer_len = config.cn0_buffer_len
    self._cn0_prompt_buffer = []
    
    # initialize kalman filter
    self._is_cov_initialized = False
    self._order = config.order
    self.T = config.T
    if config.order == 2:
      self._I = np.eye(8)
      self.rx_state = np.array([config.pos, config.vel, config.clock_bias, config.clock_drift], dtype=float)
    elif config.order == 3:
      self._I = np.eye(11)
      self.rx_state = np.array([config.pos, config.vel, 0,0,0, config.clock_bias, config.clock_drift], dtype=float)
    else:
      _log.Error("VectorTrack::init -> invalid filter order specified. Must be 2 or 3.")
      raise ValueError()
    self.__generate_A(config.order)
    self.__generate_Q(config.order, config.process_noise_stdev)
    self.rx_cov = self._I
    
    # generate ecef-2-enu rotation matrix
    self._lla0 = ecef2lla(config.pos)
    self._C_e_n = ecef2enuDcm(self._lla0)
  

#--------------------------------------------------------------------------------------------------#
  #! === Time Update (prediction) ===
  def time_update(self):
    # assumed constant T
    self.rx_state = self._A @ self.rx_state
    self.rx_cov = self._A @ self.rx_cov @ self._A.T + self._Q
  
  #! === Measurement Update (correction) ===
  def measurement_update(self, prange_meas: np.ndarray, prange_rate_meas: np.ndarray, emitter_states: dict):
    # predicted measurements from kalman filter
    prange_pred, prange_rate_pred = self.__predict_observables(emitter_states)
    
    # observation and covariance matrix
    self.__estimate_cn0()
    self.__generate_C(self._order)
    self.__generate_R()
    
    # compute initial covariance if necessary
    if not self._is_cov_initialized:
      self.__initialize_covariance()
      self._is_cov_initialized = True
      
    # measurement residuals
    chip_error = dll_error(self._early, self._late, 0.5) * self._chip_width
    freq_error = -fll_error(self._ip1, self._ip2, self._qp1, self._qp2, self.T) * self._wavelength
    dy = np.concatenate((chip_error + prange_meas - prange_pred, 
                         freq_error + prange_rate_meas - prange_rate_pred))
    
    # innovation filtering
    S = self._C @ self.rx_cov @ self._C.T + self._R
    norm_z = np.abs(dy / np.linalg.cholesky(S))
    fault_idx = np.nonzero(norm_z > self._innovation_stdev)
    self._C = self._C[fault_idx,:]
    self._R = np.diag(self._R[fault_idx,fault_idx])
    dy = dy[fault_idx]
    
    # update
    PCt = self.rx_cov @ self._C.T
    L = PCt @ np.linalg.inv(self._C @ PCt + self._R)
    I_LC = (self._I - L @ self._C)
    self.rx_cov = (I_LC @ self.rx_cov @ I_LC.T) + (L @ self._R @ L.T)
    self.rx_state = self.rx_state + L @ dy
  
  #! === Predict Observables ===
  def __predict_observables(self, emitter_states: dict) -> tuple[np.ndarray, np.ndarray]:
    self._num_channels = len(emitter_states)
    self._emitters = list(emitter_states.keys())
    
    self._unit_vectors = np.zeros((self._num_channels, 3))
    pranges, prange_rates = np.zeros(self._num_channels), np.zeros(self._num_channels)
    
    i = 0
    for emitter in emitter_states.values():
      pranges[i], self._unit_vectors[i,:] = compute_range_and_unit_vector(self.rx_state[0:3], emitter.pos)
      prange_rates[i] = compute_range_rate(self.rx_state[3:6], emitter.vel, self._unit_vectors[i,:])
      i += 1
    
    pranges += self.rx_state[-2]
    prange_rates += self.rx_state[-1]
    return pranges, prange_rates
  
  #! === Predict NCO Frequencies ===


#--------------------------------------------------------------------------------------------------#
  #! === Update Correlators ===
  def update_correlators(self, correlators: Correlators):
    self._correlators = correlators

  #! === Estimate CN0 ===
  def __estimate_cn0(self):
    self._cn0_prompt_buffer.append(self._correlators.IP)
    self._cn0_counter += 1
    if self._cn0_counter == self._cn0_buffer_len:
      prompt_array = np.array(self._cn0_prompt_buffer)
      for i in self.cn0.size:
        self.cn0[i] = CN0_m2m4_estimator(prompt_array[i,:], self.cn0[i], self.T)
      self._cn0_counter = 1
      self._cn0_prompt_buffer = []
    
  
#--------------------------------------------------------------------------------------------------#
  #! === Initialize Receiver Covariance Matrix ===
  def __initialize_covariance(self):
    prev_diag_P = np.diag(self._I)
    new_diag_P = np.zeros(prev_diag_P.size)
    
    # loop until approximate steady-state is achieved
    while np.any(np.abs(prev_diag_P - new_diag_P) > 1e-4):
      prev_diag_P = new_diag_P

      self.rx_cov = self._A @ self.rx_cov @ self._A.T + self._Q
      PCt = self.rx_cov @ self._C.T
      L = PCt @ np.linalg.inv(self._C @ PCt + self._R)
      ILC = (self._I - L @ self._C)
      self.rx_cov = (ILC @ self.rx_cov @ ILC.T) + (L @ self._R @ L.T)
      
      new_diag_P = np.diag(self.rx_cov)
  
  #! === State Transition Matrix ===
  def __generate_A(self, order: int):
    clk_a = np.array([[1.0, 0.0],[0.0, self.T]])
    if order == 2:
      self._A = np.block((( I3, self.T*I3,   Z32), 
                          (Z33,        I3,   Z32),
                          (Z23,       Z23, clk_a)))
    elif order == 3:
      self._A = np.block((( I3, self.T*I3, 0.5*self.T**2,   Z32), 
                          (Z33,        I3,     self.T*I3,   Z32), 
                          (Z33,       Z33,            I3,   Z32),
                          (Z23,       Z23,           Z23, clk_a)))
      
  #! === Process Noise Covariance ===
  def __generate_Q(self, order: int, sigma: float):
    # clock
    sf = self._clock_config.h0 / 2
    sg = self._clock_config.h2 * 2 * np.pi**2
    clk_q = SPEED_OF_LIGHT**2 * np.array([[sf*self.T + sg*self.T**3/3, sg*self.T**2/2],
                                          [            sg*self.T**2/2,      sg*self.T]])
    if order == 2:
      xyz_pp = sigma * self.T**3/3 * I3
      xyz_vv = sigma * self.T * I3
      xyz_pv = sigma * self.T**2/2 * I3
      self._Q = np.block(((xyz_pp, xyz_pv,   Z32), 
                          (xyz_pv, xyz_vv,   Z32), 
                          (   Z23,    Z23, clk_q)))
    elif order == 3:
      xyz_pp = sigma * self.T**5/20 * I3
      xyz_vv = sigma * self.T**3/3 * I3
      xyz_aa = sigma * self.T * I3
      xyz_pv = sigma * self.T**4/8 * I3
      xyz_pa = sigma * self.T**3/6 * I3
      xyz_va = sigma * self.T**2/2 * I3
      self._Q = np.block(((xyz_pp, xyz_pv, xyz_pa,   Z32), 
                          (xyz_pv, xyz_vv, xyz_va,   Z32), 
                          (xyz_pa, xyz_va, xyz_aa,   Z32),
                          (   Z23,    Z23,    Z23, clk_q)))
      
  #! === Observation Matrix ===
  def __generate_C(self, order: int):
    Z = np.zeros(self._unit_vectors.shape)
    Z1, I1 = np.zeros(self._unit_vectors.shape[0]), np.ones(self._unit_vectors.shape[0])
    if order == 2:
      self._C = np.block(((-self._unit_vectors, Z, I1, Z1),
                          (Z, -self._unit_vectors, Z1, I1)))
    elif order == 3:
      self._C = np.block(((-self._unit_vectors, Z, Z, I1, Z1),
                          (Z, -self._unit_vectors, Z, Z1, I1)))
      
  #! === Measurement Noise Covariance ===
  def __generate_R(self):
    self._R = np.diag( 
                np.concatenate((prange_residual_var(self.cn0, self.T, self._chip_width), 
                                prange_rate_residual_var(self.cn0, self.T, self._wavelength))) 
              )
      