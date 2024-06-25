"""
|======================================= bpsk_correlator.py =======================================|
|                                                                                                  |
|   Property of Daniel Sturdivant. Unauthorized copying of this file via any medium would be       |
|   super sad and unfortunate for me. Proprietary and confidential.                                |
|                                                                                                  |
| ------------------------------------------------------------------------------------------------ |
|                                                                                                  |
|   @file     charlizard/algorithms/bpsk_correlator.py                                             |
|   @brief    GPS BPSK tracking correlator models.                                                 |
|   @author   Daniel Sturdivant <sturdivant20@gmail.com>                                           |
|   @date     June 2024                                                                            |
|                                                                                                  |
|==================================================================================================|
"""

import numpy as np
import navtools as nt
from dataclasses import dataclass

TWO_PI = 2 * np.pi  #! [rad]
HALF_PI = 0.5 * np.pi  #! [rad]
LIGHT_SPEED = 299792458.0  #! [m/s]


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


class CorrelatorSim:
    def __init__(self, wavelength: np.ndarray = None, chip_width: np.ndarray = None, tap_spacing: float = None, dt: float = None):
        # constants
        self.__T = 0.0 if dt is None else dt
        self.__chip_width = np.empty(0) if chip_width is None else chip_width
        self.__wavelength = np.empty(0) if wavelength is None else wavelength
        self.__spacing = np.empty(0) if tap_spacing is None else np.array([tap_spacing, 0.0, -tap_spacing])

        # reset every period
        self.__cn0 = np.empty(0)
        self.__freq_errors = np.empty(0)
        self.__phase_errors = np.empty(0)
        self.__chip_errors = np.empty(0)

        # correlators
        self.__corr = Correlators(
            IE=np.empty(0),
            IP=np.empty(0),
            IL=np.empty(0),
            QE=np.empty(0),
            QP=np.empty(0),
            QL=np.empty(0),
            ip1=np.empty(0),
            ip2=np.empty(0),
            qp1=np.empty(0),
            qp2=np.empty(0),
        )

    def set_integration_period(self, T: np.ndarray):
        self.__T = T

    # def set_cn0(self, cn0: np.ndarray):
    #     self.__cn0 = cn0

    def set_chip_width(self, chip_width: np.ndarray):
        self.__chip_width = chip_width

    def set_wavelength(self, wavelength: np.ndarray):
        self.__wavelength = wavelength

    def set_correlator_spacing(self, tap_spacing: np.ndarray):
        self.__spacing = np.array([tap_spacing, 0.0, -tap_spacing])

    def simulate(
        self,
        true_rng: np.ndarray,
        true_rng_rate: np.ndarray,
        true_clk_bias: np.ndarray,
        true_clk_drift: np.ndarray,
        true_iono_delay: np.ndarray,
        true_trop_delay: np.ndarray,
        true_cn0: np.ndarray,
        meas_rng_nco: np.ndarray,
        meas_rng_rate_nco: np.ndarray,
    ) -> Correlators:
        """Simulates correlators over a single integration period based on the outputs from N
           satellites and M intermediate sampled points.

        Parameters
        ----------
        true_rng : np.ndarray
            NxM true ranges from user to satellite
        true_rng_rate : np.ndarray
            NxM true range-rates from user to satellite
        true_clk_bias : np.ndarray
            NxM true clock biases (satellite + rcvr)
        true_clk_drift : np.ndarray
            NxM true clock drifts (satellite + rcvr)
        true_iono_delay : np.ndarray
            NxM true ionospheric delays
        true_trop_delay : np.ndarray
            NxM true tropospheric delays
        meas_rng_nco : np.ndarray
            NxM estimated range from NCO (probably linear - constant frequency)
        meas_rng_rate_nco : np.ndarray
            NxM estimated freq errors from NCO

        Returns
        -------
        Correlators
            Early, Prompt, and Late correlators
        """

        # assert (
        #     true_rng.shape == true_rng_rate.shape
        #     and true_rng.shape[1] == true_clk_bias.size
        #     and true_rng.shape[1] == true_clk_drift.size
        #     and true_rng.shape == true_iono_delay.shape
        #     and true_rng.shape == true_trop_delay.shape
        #     and true_rng.shape == meas_rng_nco.shape
        #     and true_rng.shape == meas_rng_rate_nco.shape
        # ), "CORRELATORSIM::SIMULATE sizes wrong el mcstupid"

        # calculate true code, phase, and doppler
        true_chips = true_rng + true_clk_bias[None, :, None] + true_iono_delay + true_trop_delay
        true_phase = true_rng + true_clk_bias[None, :, None] - true_iono_delay + true_trop_delay
        true_doppler = true_rng_rate + true_clk_drift[None, :, None]

        # calculate errors
        M, L = meas_rng_nco.shape
        self.__cn0 = true_cn0
        if len(true_rng.shape) == 3:
            self.__chip_errors = (meas_rng_nco[:, :, None] - true_chips) / self.__chip_width[:, None, None]
            self.__phase_errors = (meas_rng_nco[:, :, None] - true_phase) / self.__wavelength[:, None, None]
            self.__freq_errors = -(meas_rng_rate_nco[:, :, None] - true_doppler) / self.__wavelength[:, None, None]
            N = true_rng.shape[2]
            self.__corr.IE = np.zeros((M, N))
            self.__corr.IP = np.zeros((M, N))
            self.__corr.IL = np.zeros((M, N))
            self.__corr.QE = np.zeros((M, N))
            self.__corr.QP = np.zeros((M, N))
            self.__corr.QL = np.zeros((M, N))
            self.__corr.ip1 = np.zeros((M, N))
            self.__corr.ip2 = np.zeros((M, N))
            self.__corr.qp1 = np.zeros((M, N))
            self.__corr.qp2 = np.zeros((M, N))
        else:
            self.__chip_errors = (meas_rng_nco - true_chips) / self.__chip_width[:, None]
            self.__phase_errors = (meas_rng_nco - true_phase) / self.__wavelength[:, None]
            self.__freq_errors = -(meas_rng_rate_nco - true_doppler) / self.__wavelength[:, None]
            self.__corr.IE = np.zeros(M)
            self.__corr.IP = np.zeros(M)
            self.__corr.IL = np.zeros(M)
            self.__corr.QE = np.zeros(M)
            self.__corr.QP = np.zeros(M)
            self.__corr.QL = np.zeros(M)
            self.__corr.ip1 = np.zeros(M)
            self.__corr.ip2 = np.zeros(M)
            self.__corr.qp1 = np.zeros(M)
            self.__corr.qp2 = np.zeros(M)
            N = 1

        # reset correlators
        # self.__phase_errors = nt.wrapToPi(self.__phase_errors)
        # self.__phase_errors[self.__phase_errors > np.pi] -= TWO_PI
        # self.__phase_errors[self.__phase_errors < -np.pi] += TWO_PI
        self.__calc_corr_output(M, L, N)

        return self.__corr

    def __calc_corr_output(self, M: int, L: int, N: int) -> Correlators:
        # assert (
        #     self.__cn0.size == self.__chip_width.size
        #     and self.__cn0.size == self.__wavelength.size
        #     and self.__cn0.size == self.__phase_errors.shape[0]
        #     and self.__cn0.size == self.__chip_errors.shape[0]
        #     and self.__cn0.size == self.__freq_errors.shape[0]
        # ), "CORRELATORSIM::CALC_CORR_OUTPUT sizes wrong el mcstupid"

        L_2 = int(L / 2)

        # average first half (average of subsets is equal to the average of the entire set)
        if len(self.__chip_errors.shape) == 3:
            chip_err_avg_1 = self.__chip_errors[:, :L_2, :].mean(axis=1)
            chip_err_avg_2 = self.__chip_errors[:, L_2:, :].mean(axis=1)
            freq_err_avg_1 = self.__freq_errors[:, :L_2, :].mean(axis=1)
            freq_err_avg_2 = self.__freq_errors[:, L_2:, :].mean(axis=1)
            phase_err_avg_1 = self.__phase_errors[:, :L_2, :].mean(axis=1)
            phase_err_avg_2 = self.__phase_errors[:, L_2:, :].mean(axis=1)
        else:
            chip_err_avg_1 = self.__chip_errors[:, :L_2].mean(axis=1)
            chip_err_avg_2 = self.__chip_errors[:, L_2:].mean(axis=1)
            freq_err_avg_1 = self.__freq_errors[:, :L_2].mean(axis=1)
            freq_err_avg_2 = self.__freq_errors[:, L_2:].mean(axis=1)
            phase_err_avg_1 = self.__phase_errors[:, :L_2].mean(axis=1)
            phase_err_avg_2 = self.__phase_errors[:, L_2:].mean(axis=1)
        chip_err_avg = (chip_err_avg_1 + chip_err_avg_2) / 2.0
        freq_err_avg = (freq_err_avg_1 + freq_err_avg_2) / 2.0
        phase_err_avg = (phase_err_avg_1 + phase_err_avg_2) / 2.0
        cn0_avg_1 = self.__cn0[:, :L_2].mean(axis=1)
        cn0_avg_2 = self.__cn0[:, L_2:].mean(axis=1)
        cn0_avg = (cn0_avg_1 + cn0_avg_2) / 2.0

        # loop through each correlator/satellite
        # equations from Scott Martin's dissertation, appendix A
        for i in range(M):
            # CN0 based amplitude
            A = np.sqrt(2.0 * cn0_avg[i] * self.__T)
            A1 = np.sqrt(cn0_avg_1[i] * self.__T)
            A2 = np.sqrt(cn0_avg_2[i] * self.__T)

            # code based auto-correlation
            spacing = np.tile(self.__spacing, [N, 1]).T
            R = 1.0 - np.abs(chip_err_avg[i] + spacing)
            R1 = 1.0 - np.abs(chip_err_avg_1[i] + spacing)
            R2 = 1.0 - np.abs(chip_err_avg_2[i] + spacing)
            R[R <= 0.0] = 0.0  # remain within 1 chip
            R1[R1 < 0.0] = 0.0
            R2[R2 < 0.0] = 0.0

            # frequency domain error
            F = np.sinc(np.pi * freq_err_avg[i] * self.__T)
            F1 = np.sinc(HALF_PI * freq_err_avg_1[i] * self.__T)
            F2 = np.sinc(HALF_PI * freq_err_avg_2[i] * self.__T)

            # phase based errors
            P = np.exp(1j * TWO_PI * phase_err_avg[i])
            P1 = np.exp(1j * TWO_PI * phase_err_avg_1[i])
            P2 = np.exp(1j * TWO_PI * phase_err_avg_2[i])

            # sub-period correlators
            self.__corr.ip1[i] = A1 * R1[1] * F1 * P1.real + np.random.randn(N)
            self.__corr.ip2[i] = A2 * R2[1] * F2 * P2.real + np.random.randn(N)
            self.__corr.qp1[i] = A1 * R1[1] * F1 * P1.imag + np.random.randn(N)
            self.__corr.qp2[i] = A2 * R2[1] * F2 * P2.imag + np.random.randn(N)

            # full-period correlators
            self.__corr.IE[i] = A * R[0] * F * P.real + np.random.randn(N)
            self.__corr.IP[i] = A * R[1] * F * P.real + np.random.randn(N)
            # self.__corr.IP[i] = self.__corr.ip1[i] + self.__corr.ip2[i]
            self.__corr.IL[i] = A * R[2] * F * P.real + np.random.randn(N)
            self.__corr.QE[i] = A * R[0] * F * P.imag + np.random.randn(N)
            self.__corr.QP[i] = A * R[1] * F * P.imag + np.random.randn(N)
            # self.__corr.QP[i] = self.__corr.qp1[i] + self.__corr.qp2[i]
            self.__corr.QL[i] = A * R[2] * F * P.imag + np.random.randn(N)
