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

TWO_PI = 2.0 * np.pi
HALF_PI = 0.5 * np.pi
PI_SQUARED = np.pi * np.pi


# * === dll_nceml_normalized ===
def dll_nceml_normalized(IE: float, QE: float, IL: float, QL: float) -> float:
    E = IE * IE + QE * QE
    L = IL * IL + QL * QL
    return 0.5 * (E - L) / (E + L)  # chips


# * === dll_cdp_normalized ===
def dll_cdp_normalized(IE: float, IP: float, IL: float) -> float:
    return 0.25 * (IE - IL) / IP  # chips


# * === fll_atan2_normalized ===
def fll_atan2_normalized(ip1: float, ip2: float, qp1: float, qp2: float, T: float) -> float:
    x = ip1 * qp2 - ip2 * qp1
    d = ip1 * ip2 + qp1 * qp2
    return np.arctan2(x, d) / (TWO_PI * T)  # Hz


# * === fll_ddcp_normalized ===
def fll_ddcp_normalized(ip1: float, ip2: float, qp1: float, qp2: float, T: float) -> float:
    IP = ip1 + ip2
    QP = qp1 + qp2
    x = ip1 * qp2 - ip2 * qp1
    d = ip1 * ip2 + qp1 * qp2
    return x * np.sign(d) / (HALF_PI * (IP * IP + QP * QP) * T)  # Hz


# * === pll_atan_normalized ===
def pll_atan_normalized(IP: float, QP: float) -> float:
    return np.arctan(QP / IP) / TWO_PI  # cycles


# * === pll_ddq_normalized ===
def pll_ddq_normalized(IP: float, QP: float) -> float:
    return QP * np.sign(IP) / (TWO_PI * np.sqrt(IP * IP + QP * QP))


# * === dll_variance ===
def dll_variance(CN0: float, T: float, D: float) -> float:
    tmp = 1.0 / (CN0 * T)
    return 0.25 * D * tmp * (1.0 + tmp)
    # return 0.5 * D * tmp;


# * === fll_variance ===
def fll_variance(CN0: float, T: float) -> float:
    tmp = 1.0 / (CN0 * T)
    return 0.5 * tmp / (PI_SQUARED * T * T) * (1.0 + tmp)


# * === pll_variance ===
def pll_variance(CN0: float, T: float) -> float:
    tmp = 0.5 / (CN0 * T)
    return tmp * (1.0 + tmp)
