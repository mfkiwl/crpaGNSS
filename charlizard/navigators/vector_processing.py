"""
|====================================== vector_processing.py ======================================|
|                                                                                                  |
|   Property of Daniel Sturdivant. Unauthorized copying of this file via any medium would be       |
|   super sad and unfortunate for me. Proprietary and confidential.                                |
|                                                                                                  |
| ------------------------------------------------------------------------------------------------ |
|                                                                                                  |
|   @file     charlizard/navigators/vector_processing.py                                           |
|   @brief    Vector processing simulation class. Assumes baseband signal.                         |
|   @author   Daniel Sturdivant <sturdivant20@gmail.com>                                           |
|   @date     June 2024                                                                            |
|                                                                                                  |
|==================================================================================================|
"""

import numpy as np
from scipy.linalg import norm, inv

from charlizard.models.bpsk_correlator import *
from charlizard.estimators.discriminator import *
from charlizard.estimators.lock_detectors import *

import navsim as ns
from navsim.error_models.clock import get_clock_allan_variance_values
from navsim.error_models.imu import get_imu_allan_variance_values
from navtools.conversions import ecef2lla, ecef2nedDcm, ecef2enuDcm, euler2dcm, dcm2quat, quat2dcm, dcm2euler, skew
from navtools.measurements import ecefg, geocentricRadius

TWO_PI = 2 * np.pi  #! [rad]
LIGHT_SPEED = 299792458.0  #! [m/s]
OMEGA_EARTH = 7.2921151467e-5
R2D = 180.0 / np.pi

ONE3 = np.ones(3)
I3 = np.eye(3)
Z33 = np.zeros((3, 3))
Z32 = np.zeros((3, 2))
Z23 = np.zeros((2, 3))
OMEGA_IE = np.array([0.0, 0.0, OMEGA_EARTH], dtype=np.double)
OMEGA_IE_E = skew(OMEGA_IE)


@dataclass
class VPConfig:
    T: float  #! integration period [s]
    order: int  #! 1: IMU, 2: constant velocity, 3: constant acceleration
    tap_spacing: float  #! (early, prompt, late) correlator tap/chip spacing
    innovation_stdev: float  #! normalized innovation filter threshold
    process_noise_stdev: float  #! (x, y, z) process noise standard deviation, only used if order==2|3
    cn0_buffer_len: int  #! number of correlator outputs to use in cn0 estimation
    cn0: np.ndarray  #! initial receiver cn0
    chip_width: np.ndarray  #! initial signal chip widths [m]
    wavelength: np.ndarray  #! initial carrier wavelengths [m]
    pos: np.ndarray  #! initial receiver ecef position
    vel: np.ndarray  #! initial receiver ecef velocity
    att: np.ndarray  #! initial receiver RPY attitude
    clock_bias: float  #! initial receiver clock bias
    clock_drift: float  #! initial receiver clock drift
    clock_type: str | ns.error_models.NavigationClock  #! receiver oscillator type
    imu_type: str | ns.error_models.IMU  #! imu specs, only used if order==1
    TOW: float
    meas: np.ndarray
    ecef_ref: np.ndarray


class VectorProcess:
    def __init__(self, config: VPConfig):
        # init
        self.__T = config.T
        self.__chip_freq = LIGHT_SPEED / config.chip_width
        self.__chip_width = config.chip_width
        self.__carrier_freq = LIGHT_SPEED / config.wavelength
        self.__wavelength = config.wavelength
        self.__order = config.order
        self.__tap_spacing = config.tap_spacing

        # state init
        self.__rx_pos = config.pos
        self.__rx_vel = config.vel
        self.__rx_clk_bias = config.clock_bias
        self.__rx_clk_drift = config.clock_bias
        self.__y_hat_old = config.meas
        self.__y_hat_new = np.zeros(config.meas.shape)
        self.__y_nco = np.zeros(config.meas.shape)
        self.__TOW = config.TOW
        self.__ecef0 = config.ecef_ref

        # cn0 estimation init
        self.__num_sv = config.cn0.size
        self.__cn0_dB = config.cn0
        self.__cn0 = 10.0 ** (config.cn0 / 10.0)
        self.__cn0_count = 0
        self.__max_cn0_count = config.cn0_buffer_len
        self.__cn0_I_buf = np.zeros((self.__cn0.size, self.__max_cn0_count))
        self.__cn0_I_buf_prev = None
        self.__cn0_Q_buf = np.zeros((self.__cn0.size, self.__max_cn0_count))

        # dynamic model init
        self.__init_dynamic_model(config)

        # correlator init
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

    #######################################* KALMAN FILTER *########################################

    def propagate(
        self, T: float, w_ib_b: np.ndarray = np.zeros(3), f_ib_b: np.ndarray = np.zeros(3)
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        # generate transition matrices / mechanize imu
        if self.__order == 1:
            fb = f_ib_b - self.__acc_bias
            wb = w_ib_b - self.__gyr_bias
            self.__q_b_e, self.__C_b_e, self.__rx_vel, self.__rx_pos = self.__mechanize(T, wb, fb)
            self.__make_A(T, wb, fb)
        else:
            self.__make_A(T)
        self.__make_Q(T)

        # propagate
        self.__x = self.__A @ self.__x
        self.__P = self.__A @ self.__P @ self.__A.T + self.__Q

        # update state (done inside mechanization for imu)
        if self.__order == 1:
            self.__rx_clk_bias = self.__x[15]
            self.__rx_clk_drift = self.__x[16]
        elif self.__order == 2:  #! constant velocity
            self.__rx_pos = self.__x[0:3]
            self.__rx_vel = self.__x[3:6]
            self.__rx_clk_bias = self.__x[6]
            self.__rx_clk_drift = self.__x[7]
        elif self.__order == 3:  #! constant acceleration
            self.__rx_pos = self.__x[0:3]
            self.__rx_vel = self.__x[3:6]
            self.__rx_clk_bias = self.__x[9]
            self.__rx_clk_drift = self.__x[10]

    def correct(self, y_true=None):
        self.__make_C()
        self.__make_R()

        # try to achieve steady state covariance
        if self.__is_cov_init == False:
            self.__init_cov()
            self.__is_cov_init = True

        # correct
        dy = (
            np.concatenate((self.__chip_width * self.__chip_err, -self.__wavelength * self.__freq_err))
            # + self.__y_nco
            # - self.__y_hat_old
        )
        # dy2 = y_true - self.__y_hat_old
        K = self.__P @ self.__C.T @ inv(self.__C @ self.__P @ self.__C.T + self.__R)
        self.__P = (self.__I - K @ self.__C) @ self.__P
        self.__x += K @ dy

        # apply corrections
        if self.__order == 1:  #! imu
            self.__rx_pos -= self.__x[0:3]
            self.__rx_vel -= self.__x[3:6]
            p0, p1, p2, p3 = 1, *(-self.__x[6:9] / 2.0)
            q0, q1, q2, q3 = self.__q_b_e
            self.__q_b_e = np.array(
                [
                    (p0 * q0 - p1 * q1 - p2 * q2 - p3 * q3),
                    (p0 * q1 + p1 * q0 + p2 * q3 - p3 * q2),
                    (p0 * q2 - p1 * q3 + p2 * q0 + p3 * q1),
                    (p0 * q3 + p1 * q2 - p2 * q1 + p3 * q0),
                ]
            )
            self.__C_b_e = quat2dcm(self.__q_b_e)
            self.__acc_bias = self.__x[9:12]
            self.__gyr_bias = self.__x[12:15]
            self.__rx_clk_bias = self.__x[15]
            self.__rx_clk_drift = self.__x[16]
            self.__x[:9] = np.zeros(9)
        elif self.__order == 2:  #! constant velocity
            self.__rx_pos = self.__x[:3]
            self.__rx_vel = self.__x[3:6]
            self.__rx_clk_bias = self.__x[6]
            self.__rx_clk_drift = self.__x[7]
        elif self.__order == 3:  #! constant acceleration
            self.__rx_pos = self.__x[:3]
            self.__rx_vel = self.__x[3:6]
            self.__rx_clk_bias = self.__x[9]
            self.__rx_clk_drift = self.__x[10]
        # print(f"ECEF pos: [{self.__rx_pos[0]}, {self.__rx_pos[1]}, {self.__rx_pos[2]}]")

    ############################################* NCO *#############################################

    def vdfll_nco_correct(
        self,
        T: float,
        sv_pos: np.ndarray,
        sv_vel: np.ndarray,
        w_ib_b: np.ndarray = np.zeros(3),
        f_ib_b: np.ndarray = np.zeros(3),
    ):
        # propagate
        # self.propagate(T, w_ib_b, f_ib_b)

        # propagate pseudoranges and pseudorange-rates
        dr = self.__rx_pos - sv_pos
        dv = self.__rx_vel - sv_vel
        r = norm(dr, axis=1)
        self.__unit_vec = dr / r[:, None]
        self.__dopp_unit_vec = (dv * r[:, None] ** 2 - dr * np.sum(dv * dr, axis=1)[:, None]) / r[:, None] ** 3
        self.__y_hat_old[: self.__num_sv] = r + self.__rx_clk_bias
        self.__y_hat_old[self.__num_sv :] = np.sum(self.__unit_vec * dv, axis=1) + self.__rx_clk_drift

        #! calculate pseudoranges and pseudorange-rates from nco frequencies
        self.__y_nco[: self.__num_sv] = self.__y_hat_new[: self.__num_sv]
        self.__y_nco[self.__num_sv :] = -self.__carrier_doppler_nco * self.__wavelength

        # correct
        # self.correct()

    def vdfll_nco_predict(
        self,
        T: float,
        sv_pos: np.ndarray,
        sv_vel: np.ndarray,
        w_ib_b: np.ndarray = np.zeros(3),
        f_ib_b: np.ndarray = np.zeros(3),
    ):
        # TODO: how to apply this false propagation to deep-integration?
        if self.__order == 1:
            fb = f_ib_b - self.__acc_bias
            wb = w_ib_b - self.__gyr_bias
            _, _, v, p = self.__mechanize(T, wb, fb)
            tmp_x = np.append(p, v)
            tmp_x = np.append(tmp_x, self.__rx_clk_bias + T * self.__rx_clk_drift)
            tmp_x = np.append(tmp_x, self.__rx_clk_drift)
        else:
            self.__make_A(T)
            tmp_x = self.__A @ self.__x

        # propagate pseudoranges and pseudorange-rates for 'end of NCO'
        dr = tmp_x[0:3] - sv_pos
        dv = tmp_x[3:6] - sv_vel
        r = norm(dr, axis=1)
        u = dr / r[:, None]

        self.__y_hat_new[: self.__num_sv] = r + tmp_x[-2]
        self.__y_hat_new[self.__num_sv :] = np.sum(u * dv, axis=1) + tmp_x[-1]
        self.__carrier_doppler_nco = -self.__y_hat_new[self.__num_sv :] / self.__wavelength

        chips = self.__y_hat_new[: self.__num_sv] / self.__chip_width
        doppler = self.__carrier_doppler_nco
        cycles = self.__y_hat_new[: self.__num_sv] / self.__wavelength

        return chips, doppler, cycles

    ########################################* CORRELATORS *#########################################

    def update_correlators(self, c: Correlators):
        # TODO: add catches for when satellites disappear mid period
        # replace current correlators
        self.__corr = c

        # increment through cn0 estimator
        # self.__cn0_I_buf[:, self.__cn0_count] = self.__corr.IP
        # self.__cn0_Q_buf[:, self.__cn0_count] = self.__corr.QP
        self.__cn0_I_buf[:, self.__cn0_count] = np.sqrt(self.__corr.IP**2 + self.__corr.QP**2)
        self.__cn0_count += 1
        if self.__cn0_count == self.__max_cn0_count:
            # re-estimate cn0
            if self.__cn0_I_buf_prev is None:
                self.__cn0_I_buf_prev = self.__cn0_I_buf.copy()
            else:
                for ii in range(self.__num_sv):
                    self.__cn0_dB[ii], self.__cn0[ii] = cn0_beaulieu_estimate(
                        self.__cn0_I_buf[ii, :], self.__cn0_I_buf_prev[ii, :], self.__T
                    )
                self.__cn0_I_buf_prev = self.__cn0_I_buf.copy()
            # for ii in range(self.__num_sv):
            #     self.__cn0_dB[ii], self.__cn0[ii] = cn0_m2m4_estimate(
            #         self.__cn0_I_buf[ii, :],
            #         self.__cn0_Q_buf[ii, :],
            #         self.__T,
            #     )
            self.__cn0_count = 0
            # print(f"CN0 dB: {self.__cn0_dB.mean()}")

        # calculate discriminators
        self.__phase_err = pll_atan_normalized(self.__corr.IP, self.__corr.QP)
        self.__freq_err = fll_atan2_normalized(
            self.__corr.ip1, self.__corr.ip2, self.__corr.qp1, self.__corr.qp2, self.__T
        )
        self.__chip_err = dll_nceml_normalized(self.__corr.IE, self.__corr.QE, self.__corr.IL, self.__corr.QL)

        # calculate discriminator variances
        self.__phase_var = pll_variance(self.__cn0, self.__T)
        self.__freq_var = fll_variance(self.__cn0, self.__T)
        self.__chip_var = dll_variance(self.__cn0, self.__T, self.__tap_spacing)

    ############################################* INIT *############################################

    def __init_dynamic_model(self, config: VPConfig):  # always add 10% buffer to noise
        # clock model
        if config.clock_type.casefold() is None:
            clock_config = get_clock_allan_variance_values("high_quality_tcxo")
        else:
            clock_config = get_clock_allan_variance_values(config.clock_type)

        self.__Sb = clock_config.h0 / 2
        self.__Sd = clock_config.h2 * 2 * np.pi**2

        # process model
        if config.order == 1:  #! imu
            if (config.imu_type.casefold() is None) or (config.imu_type.casefold() == "perfect"):
                imu_model = get_imu_allan_variance_values("navigation")
            else:
                imu_model = get_imu_allan_variance_values(config.imu_type)

            self.__Srg = (1.1 * imu_model.N_gyr) ** 2  # * self.T
            self.__Sra = (1.1 * imu_model.N_acc) ** 2  # * self.T
            self.__Sbad = (1.1 * imu_model.B_acc) ** 2  # / self.T
            self.__Sbgd = (1.1 * imu_model.B_gyr) ** 2  # / self.T
            self.__acc_bias = np.zeros(3)
            self.__gyr_bias = np.zeros(3)

            # add attitude (generate body to nav rotation)
            att = config.att / R2D  # config.att.copy() / R2D
            C_b_n = euler2dcm(att).T
            self.__C_e_ned = ecef2nedDcm(ecef2lla(config.pos))
            self.__C_b_e = self.__C_e_ned.T @ C_b_n
            self.__q_b_e = dcm2quat(self.__C_b_e)

            # error state
            self.__x = np.block([np.zeros(15), config.clock_bias, config.clock_drift])
            self.__P = np.diag(
                np.block([0.03 * ONE3, 0.301 * ONE3, 5.01 * ONE3, 5e-2 * ONE3, 1e-3 * ONE3, 5.01, 0.301]) ** 2
            )
            self.__I = np.eye(17)

        elif config.order == 2:  #! constant velocity
            self.__Sxyz = (1.1 * config.process_noise_stdev) ** 2
            self.__x = np.block([config.pos, config.vel, config.clock_bias, config.clock_drift])
            self.__P = np.diag(np.block([5.01 * ONE3, 0.301 * ONE3, 5.01, 0.301]) ** 2)
            self.__I = np.eye(8)

        elif config.order == 3:  #! constant acceleration
            self.__Sxyz = config.process_noise_stdev
            self.__x = np.block([config.pos, config.vel, 0.0 * ONE3, config.clock_bias, config.clock_drift])
            self.__P = np.diag(np.block([5.01 * ONE3, 0.301 * ONE3, 5e-2 * ONE3, 5.01, 0.301]) ** 2)
            self.__I = np.eye(11)

        self.__is_cov_init = False

    def __init_cov(self):
        # this allows for quick convergence
        # self.rx_cov = 10*solve_discrete_are(self._A.T, self._C.T, 0.5*(self._Q+self._Q.T), self._R)
        delta_diag_P = np.diag(self.__P)

        while np.any(delta_diag_P > 1e-4):
            previous_P = self.__P

            self.__P = self.__A @ self.__P @ self.__A.T + self.__Q
            PCt = self.__P @ self.__C.T
            K = PCt @ inv(self.__C @ PCt + self.__R)
            self.__P = (self.__I - K @ self.__C) @ self.__P

            delta_diag_P = np.diag(previous_P - self.__P)

    ###################################* KALMAN FILTER MATRICES *###################################

    def __mechanize(self, T: float, w_ib_b: np.ndarray, f_ib_b: np.ndarray):
        # copy old values
        q_old = self.__q_b_e.copy()
        v_old = self.__rx_vel.copy()
        p_old = self.__rx_pos.copy()

        # rotational phase increment
        a_ib_b = w_ib_b * T
        mag_a_ib_b = norm(a_ib_b)

        # (Groves E.39) precision quaternion from old to new attitude
        if mag_a_ib_b > 0:
            q_new_old = np.array([np.cos(mag_a_ib_b / 2), *np.sin(mag_a_ib_b / 2) / mag_a_ib_b * a_ib_b])
        else:
            q_new_old = np.array([1.0, 0.0, 0.0, 0.0])

        p0, p1, p2, p3 = self.__q_b_e
        q0, q1, q2, q3 = q_new_old
        a1, a2, a3 = OMEGA_IE * T / 2.0

        # (Groves E.40) quaternion multiplication attitude update
        q_new = np.array(
            [
                (p0 * q0 - p1 * q1 - p2 * q2 - p3 * q3) - (-a1 * p1 - a2 * p2 - a3 * p3),
                (p0 * q1 + p1 * q0 + p2 * q3 - p3 * q2) - (a1 * p0 + a2 * p3 - a3 * p2),
                (p0 * q2 - p1 * q3 + p2 * q0 + p3 * q1) - (-a1 * p3 + a2 * p0 + a3 * p1),
                (p0 * q3 + p1 * q2 - p2 * q1 + p3 * q0) - (a1 * p2 - a2 * p1 + a3 * p0),
            ]
        )

        # (Groves E.43) quaternion normalization
        q_new /= norm(q_new)

        # convert to DCM and euler angles
        C_new = quat2dcm(q_new)
        C_avg = quat2dcm((q_old + q_new) / 2.0)

        # (Groves 5.85) specific force transformation body-to-ECEF
        f_ib_e = C_avg @ f_ib_b

        # (Groves 5.36) velocity update
        gravity, _ = ecefg(self.__rx_pos)
        v_new = v_old + T * (f_ib_e + gravity - 2.0 * OMEGA_IE_E @ v_old)

        # (Groves 5.38) position update
        p_new = p_old + T * (v_new + v_old) / 2.0

        return q_new, C_new, v_new, p_new

    def __make_A(self, T: float, w_ib_b: np.ndarray = np.zeros(3), f_ib_b: np.ndarray = np.zeros(3)):
        clk_f = np.array([[1.0, T], [0.0, 1.0]])

        if self.__order == 1:  #! imu
            # radii of curvature and gravity
            lla = ecef2lla(self.__rx_pos)
            r_es_e = geocentricRadius(lla[0])
            _, gamma = ecefg(self.__rx_pos)
            C = self.__C_b_e

            # (Groves 14.49/50/87) state transition matrix discretization
            f21 = -skew(self.__C_b_e @ f_ib_b)
            f23 = -np.outer(2 * gamma / r_es_e, self.__rx_pos / norm(self.__rx_pos))
            self.__f21 = f21

            F11 = I3 - OMEGA_IE_E * T
            F15 = C * T - (OMEGA_IE_E @ C * T**2) / 2
            F21 = f21 * T - (f21 @ OMEGA_IE_E * T**2) / 2 - OMEGA_IE_E @ f21 * T**2
            F22 = I3 - 2 * OMEGA_IE_E * T
            F23 = f23 * T
            F24 = C * T - OMEGA_IE_E @ C * T**2
            F25 = (f21 @ C * T**2) / 2 + (f21 @ OMEGA_IE_E @ C * T**3) / 6 - (OMEGA_IE_E @ f21 @ C * T**3) / 3
            F31 = (f21 * T**2) / 2 - (f21 @ OMEGA_IE_E * T**3) / 6 - (OMEGA_IE_E @ f21 * T**3) / 3
            F32 = I3 * T - OMEGA_IE_E * T**2
            F34 = (C * T**2) / 2 - (OMEGA_IE_E @ C * T**3) / 3
            F35 = (f21 @ C * T**3) / 6

            self.__A = np.block(
                [
                    [I3, F32, F31, F34, F35, Z32],
                    [F23, F22, F21, F24, F25, Z32],
                    [Z33, Z33, F11, Z33, F15, Z32],
                    [Z33, Z33, Z33, I3, Z33, Z32],
                    [Z33, Z33, Z33, Z33, I3, Z32],
                    [Z23, Z23, Z23, Z23, Z23, clk_f],
                ]
            )

        elif self.__order == 2:  #! constant velocity
            self.__A = np.block(
                [
                    [I3, T * I3, Z32],
                    [Z33, I3, Z32],
                    [Z23, Z23, clk_f],
                ]
            )

        elif self.__order == 3:  #! constant acceleration
            self.__A = np.block(
                [
                    [I3, T * I3, 0.5 * T**2 * I3, Z32],
                    [Z33, I3, T * I3, Z32],
                    [Z33, Z33, I3, Z32],
                    [Z23, Z23, Z23, clk_f],
                ]
            )

    def __make_Q(self, T: float):
        clk_q = (
            LIGHT_SPEED**2
            * 1.1
            * np.array(
                [
                    [self.__Sb * T + self.__Sd / 3.0 * T**3, self.__Sd / 2.0 * T**2],
                    [self.__Sd / 2.0 * T**2, self.__Sd * T],
                ]
            )
        )

        if self.__order == 1:  #! imu
            Srg, Sbgd, Sra, Sbad = self.__Srg, self.__Sbgd, self.__Sra, self.__Sbad
            F21, C = self.__f21, self.__C_b_e
            F21_F21t = F21 @ F21.T
            F21_C = F21 @ C

            Q11 = (Srg * T + (Sbgd * T**3) / 3) * I3
            Q15 = (Sbgd * T**2) / 2 * C
            Q21 = ((Srg * T**2) / 2 + (Sbgd * T**4) / 4) * F21
            Q22 = (Sra * T + (Sbad * T**3) / 3) * I3 + ((Srg * T**3) / 3 + (Sbgd * T**5) / 5) * F21_F21t
            Q24 = (Sbad * T**2) / 2 * C
            Q25 = (Sbgd * T**3) / 3 * F21_C
            Q31 = ((Srg * T**3) / 3 + (Sbgd * T**5) / 5) * F21
            Q32 = ((Sra * T**2) / 2 + (Sbad * T**4) / 4) * I3 + ((Srg * T**4) / 4 + (Sbgd * T**6) / 6) * F21_F21t
            Q33 = ((Sra * T**3) / 3 + (Sbad * T**5) / 5) * I3 + ((Srg * T**5) / 5 + (Sbgd * T**7) / 7) * F21_F21t
            Q34 = (Sbad * T**3) / 3 * C
            Q35 = (Sbgd * T**4) / 4 * F21_C
            Q44 = Sbad * T * I3
            Q52 = (Sbgd * T**3) / 3 * F21.T @ C
            Q55 = Sbgd * T * I3

            self.__Q = np.block(
                [
                    [Q33, Q32.T, Q31.T, Q34, Q35, Z32],
                    [Q32.T, Q22, Q21, Q24, Q25, Z32],
                    [Q31.T, Q21.T, Q11, Z33, Q15, Z32],
                    [Q34.T, Q24, Z33, Q44, Z33, Z32],
                    [Q32.T, Q52, Q15, Z33, Q55, Z32],
                    [Z23, Z23, Z23, Z23, Z23, clk_q],
                ]
            )

        elif self.__order == 2:  #! constant velocity
            xyz_pp = self.__Sxyz * T**3 * I3 / 3.0
            xyz_vv = self.__Sxyz * T * I3
            xyz_pv = self.__Sxyz * T**2 * I3 / 3.0
            self.__Q = np.block(
                [
                    [xyz_pp, xyz_pv, Z32],
                    [xyz_pv, xyz_vv, Z32],
                    [Z23, Z23, clk_q],
                ]
            )

        elif self.__order == 3:  #! constant acceleration
            xyz_pp = self.__Sxyz * T**5 / 20.0 * I3
            xyz_vv = self.__Sxyz * T**3 / 3.0 * I3
            xyz_aa = self.__Sxyz * T * I3
            xyz_pv = self.__Sxyz * T**4 / 8.0 * I3
            xyz_pa = self.__Sxyz * T**3 / 6.0 * I3
            xyz_va = self.__Sxyz * T**2 / 2.0 * I3
            self.__Q = np.block(
                [
                    [xyz_pp, xyz_pv, xyz_pa, Z32],
                    [xyz_pv, xyz_vv, xyz_va, Z32],
                    [xyz_pa, xyz_va, xyz_aa, Z32],
                    [Z23, Z23, Z23, clk_q],
                ]
            )

    def __make_C(self):
        u = self.__unit_vec
        w = self.__dopp_unit_vec
        z = np.zeros(u.shape)
        z1 = np.zeros((u.shape[0], 1))
        o1 = np.ones((u.shape[0], 1))
        if self.__order == 1:  #! imu
            self.__C = np.block(
                [
                    [-u, z, z, z, z, o1, z1],
                    [-w, -u, z, z, z, z1, o1],
                ]
            )
        elif self.__order == 2:  #! constant velocity
            self.__C = np.block(
                [
                    [u, z, o1, z1],
                    [w, u, z1, o1],
                ]
            )
        elif self.__order == 3:  #! constant acceleration
            self.__C = np.block(
                [
                    [u, z, z, o1, z1],
                    [w, u, z, z1, o1],
                ]
            )

    def __make_R(self):
        o = 1.1 * np.ones(self.__num_sv)
        self.__R = np.diag(
            np.concatenate(
                (self.__chip_var * self.__chip_width**2 * o, self.__freq_var * self.__wavelength**2 * o),
            )
        )

    #####################################* RESULTS EXTRACTION *#####################################

    def extract_lla(self) -> np.ndarray:
        return ecef2lla(self.__rx_pos)

    def extract_ecef_pv(self) -> tuple[np.ndarray, np.ndarray]:
        return self.__rx_pos, self.__rx_vel

    def extract_enu_pva(self, lla0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # lla0 = ecef2lla(self.__rx_pos)
        C_e_ned = ecef2nedDcm(lla0)
        C_e_enu = ecef2enuDcm(lla0)

        if self.__order == 1:
            att = dcm2euler((C_e_ned @ self.__C_b_e).T) * R2D
        else:
            att = np.zeros(3)
        vel = C_e_enu @ self.__rx_vel
        pos = C_e_enu @ (self.__rx_pos - self.__ecef0)

        return pos, vel, att

    def extract_clock(self) -> tuple[float, float]:
        return self.__rx_clk_bias, self.__rx_clk_drift

    def extract_stds(self, lla0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # lla0 = ecef2lla(self.__rx_pos)
        C_e_enu = ecef2enuDcm(lla0)

        # extract ENU standard deviations
        if self.__order == 1:
            att = np.sqrt(np.diag(C_e_enu @ self.__P[0:3, 0:3] @ C_e_enu.T)) * R2D
            vel = np.sqrt(np.diag(C_e_enu @ self.__P[3:6, 3:6] @ C_e_enu.T))
            pos = np.sqrt(np.diag(C_e_enu @ self.__P[6:9, 6:9] @ C_e_enu.T))
        else:
            att = np.zeros(3)
            vel = np.sqrt(np.diag(C_e_enu @ self.__P[3:6, 3:6] @ C_e_enu.T))
            pos = np.sqrt(np.diag(C_e_enu @ self.__P[0:3, 0:3] @ C_e_enu.T))
        clk = np.sqrt(np.diag(self.__P[-2:, -2:]))
        return pos, vel, att, clk[0], clk[1]

    def extract_dops(self) -> tuple[float, float, float, float, float, int]:
        lla0 = ecef2lla(self.__rx_pos)
        C_e_enu = ecef2enuDcm(lla0)

        n = self.__unit_vec.shape[0]
        H = np.column_stack((self.__unit_vec, np.ones(n)))

        dop = inv(H.T @ H)
        dop[:3, :3] = C_e_enu @ dop[:3, :3] @ C_e_enu.T

        gdop = np.sqrt(np.abs(dop.trace()))
        pdop = np.sqrt(np.abs(dop[:3, :3].trace()))
        hdop = np.sqrt(np.abs(dop[:2, :2].trace()))
        vdop = np.sqrt(np.abs(dop[2, 2]))
        tdop = np.sqrt(np.abs(dop[3, 3]))
        return gdop, pdop, hdop, vdop, tdop, n
