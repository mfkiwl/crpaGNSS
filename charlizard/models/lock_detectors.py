'''
|======================================= lock_detectors.py ========================================|
|                                                                                                  |
|   Property of Daniel Sturdivant. Unauthorized copying of this file via any medium would be       |
|   super sad and unfortunate for me. Proprietary and confidential.                                |
|                                                                                                  |
| ------------------------------------------------------------------------------------------------ |
|                                                                                                  |
|   @file     charlizard/models/lock_detectors.py                                                  |
|   @brief    CN0 and signal power estimation techniques.                                          |
|   @author   Daniel Sturdivant <sturdivant20@gmail.com>                                           |
|   @date     February 2024                                                                        |
|                                                                                                  |
|==================================================================================================|
'''

import numpy as np
from numba import njit

# # === CN0_CORR_ESTIMATOR ===
# # Noise correlator based CNO estimator -> Matthew Lashley Dissertation (Appendix B)
# @njit(cache=True, fastmath=True)
# def CN0_corr_estimator(code_rep: np.ndarray, 
#                  carr_wiped: np.ndarray, 
#                  IE: float, 
#                  QE: float, 
#                  IL: float, 
#                  QL: float,
#                  sig_noise: float,
#                  sig_power: float,
#                  cn0: float,
#                  T: float,
#                  N: np.int32):

#   # determine noise correlator spacing (evenly distributed across samples)
#   spacing = np.int32(carr_wiped.size/(T*1000)/(N+1))

#   # initialize noise correlators
#   eta = 0.0
#   shift = spacing
#   for _ in np.arange(N):
#     noise = np.roll(code_rep, shift)                # circular shift for code noise correlator
#     tmp = (carr_wiped * noise).sum()                # sum
#     eta += (tmp.real*tmp.real + tmp.imag*tmp.imag)  # noise in both phase channels
#     shift += spacing

#   sig_noise = 0.98*sig_noise + 0.02*(eta/(2*N))                 # filtered noise variance
#   amp = (IE+IL)**2 + (QE+QL)**2                                 # I**2 + Q**2, four correlators
#   sig_power = 0.98*sig_power + 0.02*(amp + 4*sig_noise)         # intentionally track biased estimate (power)
#   cn0 = 10 * np.log10((sig_power-4*sig_noise) / (2*T*sig_noise))# filtered cn0 in db

#   return sig_noise, sig_power, cn0


# === CN0_M2M4_ESTIMATOR ===
@njit(cache=True, fastmath=True)
def CN0_m2m4_estimator(IP: np.ndarray, QP: np.ndarray, cn0: float, T: float) -> float:
  """Moment based CN0 estimator -> GNSS-SDR (Tracking), implements moving average filter

  Parameters
  ----------
  prompt : np.ndarray
      last N prompt tracking correlators
  cn0 : float
      previous CN0 estimate [dB-Hz]
  T : float
      integration period [s]

  Returns
  -------
  float
      new CN0 estimate [dB-Hz]
  """
  # 2nd and 4th moments
  tmp = (IP*IP) + (QP*QP)
  m_2 = tmp.sum() / IP.size
  m_4 = (tmp*tmp).sum() / IP.size

  # signal to noise ratio
  tmp = 2*m_2*m_2 - m_4
  if tmp > 0:
    tmp = np.sqrt(2*m_2*m_2 - m_4)
  else:
    tmp = np.abs(IP).sum()**2
  SNR = tmp / (m_2 - tmp)

  # cn0 moving average
  CN0 = 10*np.log10(SNR/T)
  cn0 = 0.95*cn0 + 0.05*CN0

  return cn0


# === CN0_BEAULEUI_ESTIMATOR ===
@njit(cache=True, fastmath=True)
def CN0_beaulieu_estimator(IP: np.ndarray, QP: np.ndarray, o_IP: np.ndarray, o_QP: np.ndarray, 
                           cn0: float, T: float, phase_track: bool=True) -> float:
  """Beaulieu CN0 estimator, implements moving average filter

  Parameters
  ----------
  prompt : np.ndarray
      last N prompt tracking correlators
  prev_prompt : np.ndarray
      last N prompt tracking correlators before 'prompt'
  cn0 : float
      previous CN0 estimate [dB-Hz]
  T : float
      integration period [s]
  phase_track: bool, optional
      frequency lock (False) or phase lock (True), by default True

  Returns
  -------
  float
      new CN0 estimate [dB-Hz]
  """
  # account for correlator power without phase lock
  if phase_track:
    prompt = IP
    prev_prompt = o_IP
  else:
    prompt = np.sqrt(IP**2 + QP**2)
    prev_prompt = np.sqrt(o_IP**2 + o_QP**2)
  
  # power estimators
  p_d = 0.5 * (prompt**2 + prev_prompt**2)
  p_n = (np.abs(prompt) - np.abs(prev_prompt))**2

  # signal to noise ratio
  SNR = 1 / ((p_n / p_d).sum() / prompt.size)

  # cn0 moving average
  CN0 = 10*np.log10(SNR/T)
  cn0 = 0.95*cn0 + 0.05*CN0

  return cn0
  

# === CARRIER_LOCK_DETECTOR ===
# carrier lock detector from GNSS-SDR
@njit(cache=True, fastmath=True)
def carrier_lock_detector(IP: np.ndarray, QP: np.ndarray, cos2Phi: float) -> float:
  IP = IP.sum()
  QP = QP.sum()
  IP2 = IP*IP
  QP2 = QP*QP
  NBD = IP2 - QP2
  NBP = IP2 + QP2
  return 0.95*cos2Phi + 0.05*(NBD/NBP)