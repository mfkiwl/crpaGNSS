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
  
  # code ranging errors
  chip_error = np.array([(emitter.code_pseudorange - est_prange[i]) / chip_width 
                            for i,emitter in enumerate(observables.values())])
  
  # carrier ranging errors # TODO: check this - dirty
  phase_error = np.array([(emitter.carrier_pseudorange - est_prange[i]) / wavelength 
                            for i,emitter in enumerate(observables.values())])
  
  # carrier frequency errors
  freq_error = np.array([(emitter.pseudorange_rate - est_prange_rate[i]) / -wavelength 
                            for i,emitter in enumerate(observables.values())])
  
  return CorrelatorErrors(chip_error, freq_error, phase_error)

# === correlator_model ===
# (pg. 386) Position, navigation, and timing technologies in the 21st century - v1
# (pg. 133/417) Matthew Lashley Dissertation
def correlator_model(err: CorrelatorErrors, cn0: np.ndarray, tau: float, T: float) -> Correlators:
  # convert out of dB-Hz
  n = cn0.size
  raw_cn0 = 10**(0.1*cn0)
  
  # amplitude
  A = np.sqrt(2 * raw_cn0 * T) * np.sinc(np.pi * err.freq * T)
  
  # data bit +/- 1
  # D = 1 # if np.random.random() < 0.5 else -1
  
  # autocorrelation
  RE = 1 - np.abs(err.chip + tau)
  RP = 1 - np.abs(err.chip)
  RL = 1 - np.abs(err.chip - tau)
  RE[RE < 0] = 0.0
  RP[RP < 0] = 0.0
  RL[RL < 0] = 0.0
  
  # number of phase points
  m = 20
  m_2 = 10
  
  # linear frequency error over integration period
  # subphase_time = T * np.arange(m,0,-1) / m
  subphase_time = T * np.arange(0,m)[::-1] / m
  subphase_delta = np.outer(err.freq, subphase_time)
  subphase_error = err.phase[:,None] - subphase_delta
  
  # mean errors
  mean_subphase_error = 2*np.pi * subphase_error.mean(axis=1)
  mean_half_1_subphase_err = 2*np.pi * subphase_error[:,:m_2].mean(axis=1)
  mean_half_2_subphase_err = 2*np.pi * subphase_error[:,m_2:].mean(axis=1)
  
  # correlators
  IE = A * RE * np.cos(np.pi * err.freq * T + mean_subphase_error) + np.random.randn(n)
  IP = A * RP * np.cos(np.pi * err.freq * T + mean_subphase_error) + np.random.randn(n)
  IL = A * RL * np.cos(np.pi * err.freq * T + mean_subphase_error) + np.random.randn(n)
  QE = A * RE * np.sin(np.pi * err.freq * T + mean_subphase_error) + np.random.randn(n)
  QP = A * RP * np.sin(np.pi * err.freq * T + mean_subphase_error) + np.random.randn(n)
  QL = A * RL * np.sin(np.pi * err.freq * T + mean_subphase_error) + np.random.randn(n)
  ip1 = A * RP * np.cos(np.pi * err.freq * T/2 + mean_half_1_subphase_err) + np.random.randn(n)/2
  qp1 = A * RP * np.sin(np.pi * err.freq * T/2 + mean_half_1_subphase_err) + np.random.randn(n)/2
  ip2 = A * RP * np.cos(np.pi * err.freq * T/2 + mean_half_2_subphase_err) + np.random.randn(n)/2
  qp2 = A * RP * np.sin(np.pi * err.freq * T/2 + mean_half_2_subphase_err) + np.random.randn(n)/2
  
  return Correlators(IE, IP, IL, QE, QP, QL, ip1, qp1, ip2, qp2)
