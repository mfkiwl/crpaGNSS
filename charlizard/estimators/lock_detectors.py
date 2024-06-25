"""
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
"""

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


# * === cn0_m2m4_estimate ===
@njit(cache=True, fastmath=True)
def cn0_m2m4_estimate(IPs: np.ndarray, QPs: np.ndarray, T: float) -> float:
    # 2nd and 4th moments
    m2 = 0.0
    m4 = 0.0
    N = IPs.size
    for ii in range(N):
        tmp = IPs[ii] * IPs[ii] + QPs[ii] * QPs[ii]
        m2 += tmp
        m4 += tmp * tmp
    m2 /= N
    m4 /= N
    Pd = np.sqrt(2.0 * m2 * m2 - m4)
    Pn = m2 - Pd

    cn0_pow = (Pd / Pn) / T
    cn0_dB = 10.0 * np.log10(cn0_pow)
    return cn0_dB, cn0_pow


@njit(cache=True, fastmath=True)
def cn0_m2m4_estimate2d(IPs: np.ndarray, QPs: np.ndarray, T: float) -> float:
    # 2nd and 4th moments
    M = IPs.shape[0]
    N = IPs.shape[1]
    m2 = np.zeros(M)
    m4 = np.zeros(M)
    for ii in range(N):
        tmp = IPs[:, ii] * IPs[:, ii] + QPs[:, ii] * QPs[:, ii]
        m2 += tmp
        m4 += tmp * tmp
    m2 /= N
    m4 /= N
    Pd = np.sqrt(2.0 * m2 * m2 - m4)
    Pn = m2 - Pd

    cn0_pow = (Pd / Pn) / T
    return cn0_pow


# * === cn0_beaulieu_estimate ===
@njit(cache=True, fastmath=True)
def cn0_beaulieu_estimate(IPs: np.ndarray, prev_IPs: np.ndarray, T: float) -> float:
    Pd = 0.0
    Pn = 0.0
    snr = 0.0
    N = IPs.size
    for ii in range(IPs.size):
        tmp = np.abs(IPs[ii]) - np.abs(prev_IPs[ii])
        Pn = tmp * tmp
        Pd = 0.5 * (IPs[ii] * IPs[ii] + prev_IPs[ii] * prev_IPs[ii])
        snr += Pn / Pd

    cn0_pow = (1.0 / (snr / N)) / T
    cn0_dB = 10.0 * np.log10(cn0_pow)
    return cn0_dB, cn0_pow


# * === carrier_lock_detector ===
@njit(cache=True, fastmath=True)
def carrier_lock_detector(IPs: np.ndarray, QPs: np.ndarray) -> float:
    I = 0.0
    Q = 0.0
    for ii in range(IPs.size):
        I += IPs[ii]
        Q += QPs[ii]
    I2 = I * I
    Q2 = Q * Q
    # NBD / NBP
    return (I2 - Q2) / (I2 + Q2)
