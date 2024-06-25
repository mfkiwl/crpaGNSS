"""
|====================================== crpa_gain_pattern.py ======================================|
|                                                                                                  |
|   Property of Daniel Sturdivant. Unauthorized copying of this file via any medium would be       |
|   super sad and unfortunate for me. Proprietary and confidential.                                |
|                                                                                                  |
| ------------------------------------------------------------------------------------------------ |
|                                                                                                  |
|   @file     charlizard/navigators/crpa_gain_pattern.py                                           |
|   @brief    Calculates the expected CRPA gain pattern.                                           |
|   @author   Daniel Sturdivant <sturdivant20@gmail.com>                                           |
|   @date     June 2024                                                                            |
|                                                                                                  |
|==================================================================================================|
"""

import numpy as np
from scipy.linalg import norm

TWO_PI = 2 * np.pi  #! [rad]
LIGHT_SPEED = 299792458.0  #! [m/s]


class CRPAGainPattern:
    def __init__(self, ant_body_pos: np.ndarray, wavelength: np.ndarray = None):
        self.Z = ant_body_pos.T  # 3xN
        self.n_ant = ant_body_pos.shape[0]
        self.scale_factor = np.ones(self.n_ant)
        self.scale_factor[1:] = -1 / (self.n_ant - 1)
        self.wavelength = np.empty(0) if wavelength is None else wavelength

    def set_wavelengths(self, wavelength: np.ndarray):
        self.wavelength = wavelength

    def set_expected_doa(self, az: np.ndarray, el: np.ndarray):
        self.unit_vec = np.array([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)]).T  # Mx3
        self.spatial_phase = TWO_PI / self.wavelength[:, None] * (self.unit_vec @ self.Z)  # MxN
        self.bs_weights = np.exp(-1j * self.spatial_phase)  # MxN
        self.ns_weights = self.bs_weights * self.scale_factor[None, :]  # MxN

    def calc_beamstear_gain(self, az: np.ndarray, el: np.ndarray):
        unit_vec_test = np.array([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)]).T  # Mx3
        return np.abs(
            np.sum(self.bs_weights * np.exp(1j * TWO_PI / self.wavelength[:, None] * (unit_vec_test @ self.Z)), axis=1)
        )

    def calc_nullstear_gain(self, az: np.ndarray, el: np.ndarray):
        unit_vec_test = np.array([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)])  # 3xM
        return np.abs(
            np.sum(self.ns_weights * np.exp(1j * TWO_PI / self.wavelength[:, None] * (unit_vec_test @ self.Z)), axis=1)
        )
