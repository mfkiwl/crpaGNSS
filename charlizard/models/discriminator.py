"""
|======================================= discriminator.py =========================================|
|                                                                                                  |
|   Property of Daniel Sturdivant. Unauthorized copying of this file via any medium would be       |
|   super sad and unfortunate for me. Proprietary and confidential.                                |
|                                                                                                  |
| ------------------------------------------------------------------------------------------------ |
|                                                                                                  |
|   @file                                                                                          |
|   @brief    SDR tool to generate PLL, DLL, and FLL discriminators and their noise variances      |
|   @refs     Understanding GPS/GNSS: Principles and Applications, 3rd Edition (2017)              |
|               - Elliot Kaplan, Christopher Hegarty                                               |
|   @author   Daniel Sturdivant <sturdivant20@gmail.com>                                           |
|   @date     February 2024                                                                        |
|                                                                                                  |
|==================================================================================================|
"""

import numpy as np
from numba import njit
from navtools.constants import GNSS_PI, GNSS_TWO_PI


# === pll_error ===
@njit(cache=True, fastmath=True)
def pll_error(IP: float, QP: float) -> float:
    """generates the ATAN pll discriminator

    Parameters
    ----------
    prompt : complex
        prompt correlator output

    Returns
    -------
    float
        pll discriminator
    """
    return np.arctan(QP / IP) / GNSS_TWO_PI


# === dll_error ===
@njit(cache=True, fastmath=True)
def dll_error(IE: float, QE: float, IL: float, QL: float, D: float) -> float:
    """generates the NORMALIZED EARLY-LATE dll discriminator

    Parameters
    ----------
    early : complex
        early correlator output
    late : complex
        late correlator output
    D : float
        correlator chip/tap spacing

    Returns
    -------
    float
        dll discriminator
    """
    E = np.sqrt(IE**2 + QE**2)
    L = np.sqrt(IL**2 + QL**2)
    return (1 - D) * (E - L) / (E + L)


# === fll_error ===
@njit(cache=True, fastmath=True)
def fll_error(ip1: float, ip2: float, qp1: float, qp2: float, T: float):
    """generates the ATAN2 fll discriminator

    Parameters
    ----------
    ip1 : float
        in-phase prompt correlator for the first half integration period
    ip2 : float
        in-phase prompt correlator for the second half integration period
    qp1 : float
        quadrature prompt correlator for the first half integration period
    qp2 : float
        quadrature prompt correlator for the second half integration period
    T : float
        integration period [s]

    Returns
    -------
    _type_
        fll discriminator
    """
    cross = ip1 * qp2 - ip2 * qp1
    dot = ip1 * ip2 + qp1 * qp2
    return np.arctan2(cross, dot) / (GNSS_PI * T)


# === prange_residual_var ===
@njit(cache=True, fastmath=True)
def prange_residual_var(cn0: float | np.ndarray, T: float, chip_width: float) -> float | np.ndarray:
    """calculates the pseudorange residual variance based on the DLL of a tracking loop

    Parameters
    ----------
    cn0 : float | np.ndarray
        carrier to noise ratio [dH-Hz]
    T : float
        integration period [s]
    chip_width : float
        wavelength of a prn chip [m]

    Returns
    -------
    float | np.ndarray
        pseudorange residual variance [m^2]
    """

    raw_cn0 = 10 ** (0.1 * cn0)
    bw = 2
    d = 0.5

    # return chip_width**2 * ( (1 / (2 * T**2 * raw_cn0**2)) + (1 / (4 * T * raw_cn0)) )
    return chip_width**2 * (4 * d**2 * bw / raw_cn0 * (2 * (1 - d) + 4 * d / (T * raw_cn0)))


# === prange_rate_residual_var ===
@njit(cache=True, fastmath=True)
def prange_rate_residual_var(cn0: float | np.ndarray, T: float, wavelength: float) -> float | np.ndarray:
    """calculates the pseudorange-rate residual variance based on the FLL of a tracking loop

    Parameters
    ----------
    cn0 : float | np.ndarray
        carrier to noise ratio [dH-Hz]
    T : float
        integration period [s]
    wavelength : float
        wavelength of the carrier [m]

    Returns
    -------
    float | np.ndarray
        pseudorange-rate residual variance [m^2/s^2]
    """

    raw_cn0 = 10 ** (0.1 * cn0)
    bw = 18

    # return (wavelength / (GNSS_PI * T))**2 * ( (2 / (T * raw_cn0)) + (2 / (T**2 * raw_cn0**2)) )
    return (wavelength / (np.pi * T)) ** 2 * (bw / raw_cn0 * (1 + 1 / (raw_cn0 * T)))


# === pll_var ===
@njit(cache=True, fastmath=True)
def pll_var(cn0: float | np.ndarray, T: float) -> float | np.ndarray:
    """generates the variance of the ATAN pll discriminator

    Parameters
    ----------
    cn0 : float | np.ndarray
        carrier to noise ratio [dH-Hz]
    T : float
        integration period [s]

    Returns
    -------
    float | np.ndarray
        pll discriminator variance
    """
    return (1 + 1 / (2 * cn0 * T)) / (8 * cn0 * GNSS_PI**2)  # *T


# === dll_var ===
@njit(cache=True, fastmath=True)
def dll_var(cn0: float | np.ndarray, D: float, T: float) -> float | np.ndarray:
    """generates the variance of the NORMALIZED EARLY-LATE dll discriminator

    Parameters
    ----------
    cn0 : float | np.ndarray
        carrier to noise ratio [dH-Hz]
    D : float
        correlator chip/tap spacing
    T : float
        integration period [s]

    Returns
    -------
    float | np.ndarray
        dll discriminator variance
    """
    # return (D/(4*cn0*T))*(1 + 2/((2-D)*cn0*T))
    return prange_residual_var(cn0, T, 1.0)


# === pll_var ===
@njit(cache=True, fastmath=True)
def fll_var(cn0: float | np.ndarray, T: float) -> float | np.ndarray:
    """generates the variance of the ATAN2 fll discriminator

    Parameters
    ----------
    cn0 : float | np.ndarray
        carrier to noise ratio [dH-Hz]
    T : float
        integration period [s]

    Returns
    -------
    float | np.ndarray
        fll discriminator variance
    """
    # F = 2.0 if cn0 < 35.0 else 1.0
    # return F*(1 + 1/(cn0*T)) / (2*cn0*T*PI_SQ) # *T
    return prange_rate_residual_var(cn0, T, 1.0)
