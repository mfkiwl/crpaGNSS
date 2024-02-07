'''
|========================================= correlators.py =========================================|
|                                                                                                  |
|   Property of Daniel Sturdivant. Unauthorized copying of this file via any medium would be       |
|   super sad and unfortunate for me. Proprietary and confidential.                                |
|                                                                                                  |
| ------------------------------------------------------------------------------------------------ |
|                                                                                                  |
|   @file     charlizard/models/correlators.py                                                     |
|   @brief    GNSS tracking correlator models.                                                     |
|   @author   Daniel Sturdivant <sturdivant20@gmail.com>                                           |
|   @date     February 2024                                                                        |
|                                                                                                  |
|==================================================================================================|
'''

import numpy as np
from numba import njit
from dataclasses import dataclass

# === Correlators ==================================================================================
@dataclass
class Correlators:
  IE: np.ndarray
  IP: np.ndarray
  IL: np.ndarray
  QE: np.ndarray
  QP: np.ndarray
  QL: np.ndarray
  ip1: np.ndarray
  qp1: np.ndarray
  ip2: np.ndarray
  qp2: np.ndarray
  
@dataclass()
class CorrelatorErrors:
  pseudorange: np.ndarray
  pseudorange_rate: np.ndarray
  carrier_pseudorange: np.ndarray
  chip: np.ndarray
  freq: np.ndarray
  phase: np.ndarray


# === Signal Functions =============================================================================

# === correlate ===
@njit(cache=True, fastmath=True)
def correlate(code_rep: np.ndarray, carr_wiped: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """generate early prompt and late correlators

  Parameters
  ----------
  code_rep : np.ndarray
      early, prompt, and late local code replicas
  carr_wiped : np.ndarray
      local carrier replica

  Returns
  -------
  tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
      Early, Prompt, Late correlators and Prompt subcorrelators
  """
  half_len = int(carr_wiped.size / 2)

  first_half = complex(code_rep[:,:half_len]) @ carr_wiped[:half_len]
  second_half = complex(code_rep[:,half_len:]) @ carr_wiped[half_len:]
  early, prompt, late = first_half + second_half

  #      early  prompt  late    prompt1         prompt2
  return early, prompt, late, first_half[1], second_half[1]


# === Simulation Functions =========================================================================

# === correlator_error ===
def correlator_error(
    observables: dict, est_prange: np.ndarray, est_prange_rate: np.ndarray, chip_width: float, wavelength: float
  ) -> CorrelatorErrors:
  """calculates correlator error based on simulated and estimated observables

  Parameters
  ----------
  observables : dict
      contains simulated observable data for each emitter (pseudoranges and rates)
  est_prange : np.ndarray
      estimated pseudorange for each emitter from navigator
  est_prange_rate : np.ndarray
      estimated pseudorange-rate for each emitter from navigator
  chip_width : float
      c/a code chip wavelength [m]
  wavelength : float
      carrier wavelength [m]

  Returns
  -------
  CorrelatorErrors
      measurement and tracking errors
  """
  meas_prange = np.array([emitter.code_pseudorange for emitter in observables.values()])
  meas_carr_prange = np.array([emitter.carrier_pseudorange for emitter in observables.values()])
  meas_prange_rate = np.array([emitter.pseudorange_rate for emitter in observables.values()])
  
  # code errors
  range_error = meas_prange - est_prange
  chip_error = range_error / chip_width
  
  # carrier errors
  range_rate_error = meas_prange_rate - est_prange_rate
  carrier_range_error = meas_carr_prange - est_prange   # TODO: check this, dirty
  freq_error = -range_rate_error / wavelength
  phase_error = carrier_range_error / wavelength
  
  return CorrelatorErrors(range_error, range_rate_error, carrier_range_error, chip_error, freq_error, phase_error)

# === correlator_model ===
# (pg. 386) Position, navigation, and timing technologies in the 21st century - v1
# (pg. 133/417) Matthew Lashley Dissertation
def correlator_model(err: CorrelatorErrors, cn0: np.ndarray, tau: float, T: float) -> Correlators:
  # convert out of dB-Hz
  n = cn0.size
  raw_cn0 = 10**(0.1*cn0)
  
  # amplitude
  p = np.pi * err.freq * T
  A = np.sqrt(2*raw_cn0*T) * np.sin(p) / p
  
  # data bit +/- 1
  D = 1 # if np.random.random() < 0.5 else -1
  
  # autocorrelation
  RE = 1 - np.abs(err.chip + tau)
  RP = 1 - np.abs(err.chip)
  RL = 1 - np.abs(err.chip - tau)
  
  # linear sub-phase intervals
  m = 10            # number of phase points
  subphase_time = T * np.arange(m,0,-1) / m
  subphase_offset_linear = np.outer(err.freq, subphase_time)
  subphase_error = err.phase[:,None] - subphase_offset_linear
  
  # subphase carrier replicas
  inphase = np.cos(p[:,None] + subphase_error)
  quadrature = np.sin(p[:,None] + subphase_error)
  
  # subphase correlators
  sub_ie = np.array([A[i]*RE[i]*D*inphase[i,:] + np.random.randn(m) for i in range(n)])
  sub_ip = np.array([A[i]*RP[i]*D*inphase[i,:] + np.random.randn(m) for i in range(n)])
  sub_il = np.array([A[i]*RL[i]*D*inphase[i,:] + np.random.randn(m) for i in range(n)])
  sub_qe = np.array([A[i]*RE[i]*D*quadrature[i,:] + np.random.randn(m) for i in range(n)])
  sub_qp = np.array([A[i]*RP[i]*D*quadrature[i,:] + np.random.randn(m) for i in range(n)])
  sub_ql = np.array([A[i]*RL[i]*D*quadrature[i,:] + np.random.randn(m) for i in range(n)])
  
  # correlators
  ip1 = sub_ip[:,:5].sum(axis=1)
  ip2 = sub_ip[:,5:].sum(axis=1)
  qp1 = sub_qp[:,:5].sum(axis=1)
  qp2 = sub_qp[:,5:].sum(axis=1)
  IE = sub_ie.sum(axis=1)
  IP = ip1 + ip2
  IL = sub_il.sum(axis=1)
  QE = sub_qe.sum(axis=1)
  QP = qp1 + qp2
  QL = sub_ql.sum(axis=1)
  
  return Correlators(IE, IP, IL, QE, QP, QL, ip1, qp1, ip2, qp2)
