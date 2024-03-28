"""
|========================================== gnss_ins.py ===========================================|
|                                                                                                  |
|   Property of Daniel Sturdivant. Unauthorized copying of this file via any medium would be       |
|   super sad and unfortunate for me. Proprietary and confidential.                                |
|                                                                                                  |
| ------------------------------------------------------------------------------------------------ |
|                                                                                                  |
|   @file     charlizard/navigators/soop_ins.py                                                    |
|   @brief    Deeply/Tightly coupled GNSS-INS (optional pseudorange-rates only).                   |
|               - Results in the local tangent frame (ENU) for comparisons!                        |
|   @author   Daniel Sturdivant <sturdivant20@gmail.com>                                           |
|   @date     March 2024                                                                           |
|                                                                                                  |
|==================================================================================================|
"""

import numpy as np
from scipy.linalg import norm, inv, pinv, cholesky

from charlizard.navigators.structures import GNSSINSConfig
from charlizard.models.correlators import Correlators
from charlizard.models.discriminator import prange_rate_residual_var, prange_residual_var, fll_error, dll_error
from charlizard.models.lock_detectors import CN0_m2m4_estimator, CN0_beaulieu_estimator

from navsim.error_models.clock import get_clock_allan_variance_values
from navsim.error_models.imu import get_imu_allan_variance_values

from navtools.conversions import ecef2lla, ecef2enuDcm, ecef2nedDcm, euler2dcm, dcm2quat, quat2dcm, dcm2euler, skew
from navtools.measurements import ned2ecefg, ecefg, geocentricRadius
from navtools.constants import GNSS_OMEGA_EARTH, SPEED_OF_LIGHT

from skyfield.framelib import itrs

R2D = 180 / np.pi
LLA_R2D = np.array([R2D, R2D, 1.0], dtype=np.double)
I3 = np.eye(3, dtype=np.double)
Z33 = np.zeros((3, 3), dtype=np.double)
Z32 = np.zeros((3, 2), dtype=np.double)
Z23 = np.zeros((2, 3), dtype=np.double)
OMEGA_IE = np.array([0.0, 0.0, GNSS_OMEGA_EARTH], dtype=np.double)
OMEGA_IE_E = skew(OMEGA_IE)
# I = np.eye(23)
I = np.eye(17)
# I = np.eye(11)


class SoopIns:
    @property
    def extract_states(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # update ENU state
        att = dcm2euler((self.__C_e_ned @ self.__C_b_e).T) * R2D
        # att = dcm2euler((self.__C_e_n @ self.__C_b_e).T) * R2D
        vel = self.__C_e_n @ self.__v_eb_e
        pos = self.__C_e_n @ (self.__r_eb_e - self.__ecef0)
        lla = ecef2lla(self.__r_eb_e) * LLA_R2D
        clk = np.array([self.__clk_bias, self.__clk_drift])
        return pos, vel, att, lla, clk

    @property
    def extract_stds(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # extract ENU standard deviations
        att = np.sqrt(np.abs(np.diag(self.__C_e_n @ self.rx_cov[0:3, 0:3] @ self.__C_e_n.T))) * R2D
        # att = dcm2euler((self.__C_e_ned @ self.rx_cov[0:3,0:3]).T) * R2D
        vel = np.sqrt(np.abs(np.diag(self.__C_e_n @ self.rx_cov[3:6, 3:6] @ self.__C_e_n.T)))
        pos = np.sqrt(np.abs(np.diag(self.__C_e_n @ self.rx_cov[6:9, 6:9] @ self.__C_e_n.T)))
        clk = np.sqrt(np.abs(np.diag(self.rx_cov[-2:, -2:])))
        return pos, vel, att, clk

    @property
    def extract_dops(self) -> tuple[float, float, float, float, float, int]:
        try:
            n = self.__range_unit_vectors.shape[0]
            H = np.column_stack((-self.__range_unit_vectors, np.ones(n)))
            if n > 3:
                dop = inv(H.T @ H)
            else:
                dop = pinv(H.T @ H)
            dop[:3, :3] = self.__C_e_n @ dop[:3, :3] @ self.__C_e_n.T
            gdop = np.sqrt(np.abs(dop.trace()))
            pdop = np.sqrt(np.abs(dop[:3, :3].trace()))
            hdop = np.sqrt(np.abs(dop[:2, :2].trace()))
            vdop = np.sqrt(np.abs(dop[2, 2]))
            tdop = np.sqrt(np.abs(dop[3, 3]))
        except:
            gdop, pdop, hdop, vdop, tdop, n = 0, 0, 0, 0, 0, 0
        return gdop, pdop, hdop, vdop, tdop, n

    #! === __init__ ===
    def __init__(self, config: GNSSINSConfig) -> None:
        """Initialize GNSS-INS filter

        Parameters
        ----------
        config : GNSSINSConfig
            config -> see charlizard/navigators/structures.py
        """

        # kalman filter config
        self.T = config.T
        self.T_rcvr = config.T_rcvr
        self.cn0 = config.cn0
        self.__innovation_std = config.innovation_stdev
        self.__tap_spacing = config.tap_spacing
        self.__coupling = config.coupling
        self.__chip_width = SPEED_OF_LIGHT / 1.023e6
        self.wavelength = SPEED_OF_LIGHT / 1575.42e6

        # cn0 estimation config
        self.__cn0_counter = 0
        self.__cn0_buffer_len = config.cn0_buffer_len
        self.__cn0_ip_buffer = []
        self.__cn0_qp_buffer = []

        # clock model
        if config.clock_type.casefold() is None:
            clock_config = get_clock_allan_variance_values("high_quality_tcxo")
        else:
            clock_config = get_clock_allan_variance_values(config.clock_type)
        Sb = clock_config.h0 / 2
        Sd = clock_config.h2 * 2 * np.pi**2
        self.__clk_q = (
            1.1
            * SPEED_OF_LIGHT**2
            * np.array(
                [[Sb * self.T + Sd / 3 * self.T**3, Sd / 2 * self.T**2], [Sd / 2 * self.T**2, Sd * self.T]],
                dtype=np.double,
            )
        )
        self.__clk_f = np.array([[1.0, self.T], [0.0, 1.0]], dtype=np.double)

        # imu model
        if config.imu_model.casefold() is None or "perfect":
            imu_model = get_imu_allan_variance_values("industrial")
        else:
            imu_model = get_imu_allan_variance_values(config.imu_model)
        self.__beta_acc = self.T / imu_model.Tc_acc
        self.__beta_gyr = self.T / imu_model.Tc_gyr
        self.__Srg = (1.1 * imu_model.B_gyr) ** 2  # * self.T
        self.__Sra = (1.1 * imu_model.B_acc) ** 2  # * self.T
        self.__Sbad = (1.1 * imu_model.N_acc) ** 2  # / self.T
        self.__Sbgd = (1.1 * imu_model.N_gyr) ** 2  # / self.T

        # initialize user state
        self.__ecef0 = config.pos.copy()
        self.__lla0 = ecef2lla(config.pos)
        self.__r_eb_e = config.pos.copy() + np.random.randn(3) * 5  # 5.75
        self.__v_eb_e = config.vel.copy() + np.random.randn(3) * 0.3  # 0.3
        self.__acc_bias = np.zeros(3)
        self.__gyr_bias = np.zeros(3)
        self.__acc_drift = np.zeros(3)
        self.__gyr_drift = np.zeros(3)
        self.__clk_bias = config.clock_bias
        self.__clk_drift = config.clock_drift

        # generate body to nav rotation
        att = (config.att + np.random.randn(3) * 0.5) / R2D
        C_b_n = euler2dcm(att).T
        self.__C_e_ned = ecef2nedDcm(self.__lla0)
        self.__C_e_n = ecef2enuDcm(self.__lla0)
        self.__C_b_e = self.__C_e_ned.T @ C_b_n
        self.__q_b_e = dcm2quat(self.__C_b_e)

        # kalman state and covariance
        # self.rx_state = np.zeros(23)
        self.rx_state = np.zeros(17)
        # self.rx_state = np.zeros(11)
        self.rx_cov = np.diag(
            np.array(
                [
                    0.03,
                    0.03,
                    0.03,
                    0.05,
                    0.06,
                    0.05,
                    2.0,
                    2.0,
                    2.0,
                    5e-2,
                    5e-2,
                    5e-2,
                    1e-3,
                    1e-3,
                    1e-3,
                    # 5e-2,
                    # 5e-2,
                    # 5e-2,
                    # 1e-3,
                    # 1e-3,
                    # 1e-3,
                    2.0,
                    0.1,
                ]
            )
            ** 2
        )
        self.__is_cov_initialized = False

    # --------------------------------------------------------------------------------------------------#
    #! === Kalman Filter Time Update (prediction) ===
    def time_update(self, w_ib_b: np.ndarray, f_ib_b: np.ndarray):
        # mechanize imu
        wb = w_ib_b + (self.__gyr_bias + self.__gyr_drift)
        fb = f_ib_b + (self.__acc_bias + self.__acc_drift)

        # generate transition matrices
        self.__generate_F(fb)
        self.__generate_Q()

        # time update (assumed constant T)
        self.__clk_bias += self.__clk_drift * self.T
        # half_Q = self.__Q / 2
        # self.rx_cov = self.__F @ (self.rx_cov + half_Q) @ self.__F.T + half_Q
        self.rx_cov = self.__F @ self.rx_cov @ self.__F.T + 0.5 * (self.__Q + self.__Q.T)
        self.mechanize(wb, fb)

    #! === Kalman Filter Measurement Update (correction) ===
    def measurement_update(
        self,
        noise: np.ndarray,
        emitter_states: dict,
        wavelength: np.ndarray,
        rx_pos: np.ndarray,
        rx_vel: np.ndarray,
        rx_c: float,
        emitters,
        times,
    ):

        if noise.size > 0:
            # generate observation and covariance matrix
            self.wavelength = wavelength
            self.__generate_R()

            # compute initial covariance if necessary
            if not self.__is_cov_initialized:
                self.__initialize_covariance(emitter_states)
                self.__is_cov_initialized = True

            #! EKF ---
            pos = self.__r_eb_e
            vel = self.__v_eb_e
            cb = self.__clk_bias / SPEED_OF_LIGHT
            cd = self.__clk_drift / SPEED_OF_LIGHT

            self.__generate_H(emitter_states, pos, vel)
            y = np.zeros(noise.size)
            y_hat = np.zeros(noise.size)

            for j, e in enumerate(emitter_states.values()):
                y[j] = self.__doppler(rx_pos, rx_vel, rx_c[1], e.pos, e.vel) + noise[j]
                y_hat[j] = self.__doppler(pos, vel, cd, e.pos, e.vel)

            # dtR = 0.2
            # ratio = 1 + cd
            # vel_dtr = vel * dtR / ratio
            # clk_dtr = cd * dtR / ratio
            # rx_ratio = 1 / rx_cd
            # rx_vel_dtr = rx_vel * dtR / rx_ratio
            # rx_clk_dtr = rx_cd * dtR / rx_ratio
            # for j, e in enumerate(emitters):
            #     sv_pos, sv_vel = e.at(times).frame_xyz_and_velocity(itrs)
            #     sv_pos = sv_pos.m.T

            #     adr_m2 = self.__adr(pos - 2 * vel_dtr, 0 - 2 * clk_dtr, sv_pos[0, :])
            #     adr_m1 = self.__adr(pos - vel_dtr, 0 - clk_dtr, sv_pos[1, :])
            #     adr_p1 = self.__adr(pos + vel_dtr, 0 + clk_dtr, sv_pos[3, :])
            #     adr_p2 = self.__adr(pos + 2 * vel_dtr, 0 + 2 * clk_dtr, sv_pos[4, :])
            #     y_hat[j] = (adr_m2 - 8 * adr_m1 + 8 * adr_p1 - adr_p2) / (12 * dtR)

            #     adr_m2 = self.__adr(rx_pos - 2 * rx_vel_dtr, 0 - 2 * rx_clk_dtr, sv_pos[0, :])
            #     adr_m1 = self.__adr(rx_pos - rx_vel_dtr, 0 - rx_clk_dtr, sv_pos[1, :])
            #     adr_p1 = self.__adr(rx_pos + rx_vel_dtr, 0 + rx_clk_dtr, sv_pos[3, :])
            #     adr_p2 = self.__adr(rx_pos + 2 * rx_vel_dtr, 0 + 2 * rx_clk_dtr, sv_pos[4, :])
            #     y[j] = (adr_m2 - 8 * adr_m1 + 8 * adr_p1 - adr_p2) / (12 * dtR) + noise[j]

            # innovation filter
            dy = y - y_hat
            # print(dy)
            S = self.__H @ self.rx_cov @ self.__H.T + np.diag(self.__R)
            norm_dy = np.abs(dy / np.sqrt(np.diag(S)))
            mask = norm_dy < 3  # only pass innovations within 3 standard deviations
            dy = dy[mask]
            H = self.__H[mask, :]
            R = np.diag(self.__R[mask])
            # H = self.__H
            # R = np.diag(self.__R)

            # measurement update
            if dy.size > 0:
                PHt = self.rx_cov @ H.T
                K = PHt @ inv(H @ PHt + R)
                I_KH = I - K @ H
                self.rx_state += K @ dy
                self.rx_cov = (I_KH @ self.rx_cov @ I_KH.T) + (K @ R @ K.T)
            #! ---

            # #! LM-IEKF ---
            # x = self.rx_state.copy()
            # delta_X = 1
            # mu = 1e-3
            # k = 0
            # while (delta_X > 1e-6) and (k < 100):
            #     tmp_pos = self.__r_eb_e + x[6:9]
            #     tmp_vel = self.__v_eb_e + x[3:6]
            #     tmp_clk_drift = (self.__clk_drift + x[-1]) / SPEED_OF_LIGHT

            #     self.__generate_H(emitter_states, tmp_pos, tmp_vel)
            #     y_hat = np.zeros(noise.size)
            #     y = np.zeros(noise.size)
            #     for j, e in enumerate(emitter_states.values()):
            #         y[j] = -self.__doppler(rx_pos, rx_vel, rx_c[1], e.pos, e.vel) + noise[j]
            #         y_hat[j] = -self.__doppler(tmp_pos, tmp_vel, tmp_clk_drift, e.pos, e.vel)

            #     dy = y - y_hat
            #     dx = self.rx_state - x
            #     # H = self.__H
            #     # R = np.diag(self.__R)

            #     # innovation filter
            #     S = self.__H @ self.rx_cov @ self.__H.T + np.diag(self.__R)
            #     norm_dy = np.abs(dy / np.sqrt(np.diag(S)))
            #     mask = norm_dy < 2  # only pass innovations within 3 standard deviations
            #     dy = dy[mask]
            #     H = self.__H[mask, :]
            #     R = np.diag(self.__R[mask])

            #     # K = P @ H.T @ inv(H @ P @ H.T + self.__R)
            #     # x = self.rx_state + K @ (meas_prange_rate - y_hat - H @ (self.rx_state - x))
            #     # delta_X = norm(self.rx_state - x) / norm(x)

            #     B = np.diag(np.diag(H.T @ inv(R) @ H + inv(self.rx_cov)))
            #     P = (I - self.rx_cov @ inv(self.rx_cov + inv(B) / mu)) @ self.rx_cov
            #     K = P @ H.T @ inv(H @ P @ H.T + R)
            #     tmp1 = self.rx_state + K @ (dy - H @ dx) - mu * (I - K @ H) @ P @ B @ dx
            #     tmp2 = norm(x - tmp1) / norm(tmp1)

            #     if tmp2 > delta_X:
            #         mu *= 5
            #     else:
            #         mu /= 5
            #         x, delta_X = tmp1, tmp2
            #         K_, H_, R_ = K, H, R
            #     k += 1

            # I_KH = I - K_ @ H_
            # self.rx_cov = (I_KH @ self.rx_cov @ I_KH.T) + (K_ @ R_ @ K_.T)
            # self.rx_state = x
            # #! ---

            # closed loop correction (error state)
            p0, p1, p2, p3 = 1, *(self.rx_state[:3] / 2)
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
            # self.__C_b_e = (I3 - skew(self.rx_state[:3])) @ self.__C_b_e
            self.__v_eb_e += self.rx_state[3:6]
            self.__r_eb_e += self.rx_state[6:9]
            self.__clk_bias += self.rx_state[-2]
            self.__clk_drift += self.rx_state[-1]
            self.__acc_bias += self.rx_state[9:12]
            self.__gyr_bias += self.rx_state[12:15]
            # self.__acc_drift += self.rx_state[15:18]
            # self.__gyr_drift += self.rx_state[18:21]
            # self.rx_state = np.zeros(23)
            self.rx_state = np.zeros(17)
            # self.rx_state = np.zeros(11)

    #! ===IMU Mechanization (ECEF Frame) ===
    def mechanize(self, w_ib_b: np.ndarray, f_ib_b: np.ndarray):
        # copy old values
        q_old = self.__q_b_e.copy()
        v_old = self.__v_eb_e.copy()
        p_old = self.__r_eb_e.copy()

        # rotational phase increment
        a_ib_b = w_ib_b * self.T
        mag_a_ib_b = norm(a_ib_b)

        # (Groves E.39) precision quaternion from old to new attitude
        if mag_a_ib_b > 0:
            q_new_old = np.array(
                [np.cos(mag_a_ib_b / 2), *np.sin(mag_a_ib_b / 2) / mag_a_ib_b * a_ib_b], dtype=np.double
            )
        else:
            q_new_old = np.array([1, 0, 0, 0], dtype=np.double)

        p0, p1, p2, p3 = self.__q_b_e
        q0, q1, q2, q3 = q_new_old
        a1, a2, a3 = OMEGA_IE * self.T / 2

        # (Groves E.40) quaternion multiplication attitude update
        self.__q_b_e = np.array(
            [
                (p0 * q0 - p1 * q1 - p2 * q2 - p3 * q3) - (-a1 * p1 - a2 * p2 - a3 * p3),
                (p0 * q1 + p1 * q0 + p2 * q3 - p3 * q2) - (a1 * p0 + a2 * p3 - a3 * p2),
                (p0 * q2 - p1 * q3 + p2 * q0 + p3 * q1) - (-a1 * p3 + a2 * p0 + a3 * p1),
                (p0 * q3 + p1 * q2 - p2 * q1 + p3 * q0) - (a1 * p2 - a2 * p1 + a3 * p0),
            ]
        )

        # (Groves E.43) quaternion normalization
        self.__q_b_e /= norm(self.__q_b_e)

        # convert to DCM and euler angles
        self.__C_b_e = quat2dcm(self.__q_b_e)
        C_avg = quat2dcm((q_old + self.__q_b_e) / 2)

        # (Groves 5.85) specific force transformation body-to-ECEF
        f_ib_e = C_avg @ f_ib_b

        # (Groves 5.36) velocity update
        gravity, _ = ned2ecefg(self.__r_eb_e)
        # gravity,_ = ecefg(self.__r_eb_e)
        self.__v_eb_e = v_old + self.T * (f_ib_e + gravity - 2 * OMEGA_IE_E @ v_old)

        # (Groves 5.38) position update
        self.__r_eb_e = p_old + self.T * (self.__v_eb_e + v_old) / 2

    #! === Update Correlators ===
    def update_correlators(self, correlators: Correlators):
        self.__correlators = correlators

    # --------------------------------------------------------------------------------------------------#
    #! === State Tranisition Matrix ===
    def __generate_F(self, f_ib_b: np.ndarray):
        # radii of curvature and gravity
        lla = ecef2lla(self.__r_eb_e)
        r_es_e = geocentricRadius(lla[0])
        _, gamma = ned2ecefg(self.__r_eb_e)
        # _,gamma = ecefg(self.__r_eb_e)

        # (Groves 14.49/50/87) state transition matrix discretization
        f21 = -skew(self.__C_b_e @ f_ib_b)
        f23 = -np.outer(2 * gamma / r_es_e, self.__r_eb_e / norm(self.__r_eb_e))
        self.__f21 = f21

        T, C = self.T, self.__C_b_e
        self.__clk_f = np.array([[1, T / (1 + self.__clk_drift / SPEED_OF_LIGHT)], [0, 1]])

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
        F35 = (F21 @ C * T**3) / 6
        F66 = np.exp(-self.__beta_acc) * I3
        F77 = np.exp(-self.__beta_gyr) * I3
        self.__F = np.block(
            [
                [F11, Z33, Z33, Z33, F15, Z32],
                [F21, F22, F23, F24, F25, Z32],
                [F31, F32, I3, F34, F35, Z32],
                [Z33, Z33, Z33, I3, Z33, Z32],
                [Z33, Z33, Z33, Z33, I3, Z32],
                [Z23, Z23, Z23, Z23, Z23, self.__clk_f],
            ]
        )
        # self.__F = np.block(
        #     [
        #         [F11, Z33, Z33, Z33, F15, Z33, F15, Z32],
        #         [F21, F22, F23, F24, F25, F24, F25, Z32],
        #         [F31, F32, I3, F34, F35, F34, F35, Z32],
        #         [Z33, Z33, Z33, I3, Z33, Z33, Z33, Z32],
        #         [Z33, Z33, Z33, Z33, I3, Z33, Z33, Z32],
        #         [Z33, Z33, Z33, Z33, Z33, F66, Z33, Z32],
        #         [Z33, Z33, Z33, Z33, Z33, Z33, F77, Z32],
        #         [Z23, Z23, Z23, Z23, Z23, Z23, Z23, self.__clk_f],
        #     ]
        # )

        # idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16]
        # self.__F = (self.__F[idx, :])[:, idx]

        # self.__F = np.block(
        #     [
        #         [F11, Z33, Z33, Z33, C * T, Z32],
        #         [F21 * T, F22, F23 * T, C * T, Z33, Z32],
        #         [Z33, I3 * T, I3, Z33, Z33, Z32],
        #         [Z33, Z33, Z33, I3, Z33, Z32],
        #         [Z33, Z33, Z33, Z33, I3, Z32],
        #         [Z23, Z23, Z23, Z23, Z23, self.__clk_f],
        #     ]
        # )

    #! === Process Noise Covariance ===
    def __generate_Q(self):
        Srg, Sbgd, Sra, Sbad = self.__Srg, self.__Sbgd, self.__Sra, self.__Sbad
        T, F21, C = self.T, self.__f21, self.__C_b_e
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
        Q46 = Sra * T**2 / 2 * I3
        Q52 = (Sbgd * T**3) / 3 * F21.T @ C
        Q55 = Sbgd * T * I3
        Q57 = Srg * T**2 / 2 * I3
        Q66 = (1 - np.exp(-2 * self.__beta_acc)) * Sbad * I3
        Q77 = (1 - np.exp(-2 * self.__beta_gyr)) * Sbgd * I3
        self.__Q = np.block(
            [
                [Q11, Q21.T, Q31.T, Z33, Q15, Z32],
                [Q21, Q22, Q32.T, Q24, Q25, Z32],
                [Q31, Q32, Q33, Q34, Q35, Z32],
                [Z33, Q24, Q34.T, Q44, Z33, Z32],
                [Q15, Q52, Q32.T, Z33, Q55, Z32],
                [Z23, Z23, Z23, Z23, Z23, self.__clk_q],
            ]
        )
        # self.__Q = np.block(
        #     [
        #         [Q11, Q21.T, Q31.T, Z33, Q15, Z33, Q15, Z32],
        #         [Q21, Q22, Q32.T, Q24, Q25, Q24, Q25, Z32],
        #         [Q31, Q32, Q33, Q34, Q35, Q34, Q35, Z32],
        #         [Z33, Q24, Q34.T, Q44, Z33, Q46, Z33, Z32],
        #         [Q15, Q52, Q32.T, Z33, Q55, Z33, Q57, Z32],
        #         [Z33, Q24, Q34.T, Q46, Z33, Q66, Z33, Z32],
        #         [Q15, Q52, Q32.T, Z33, Q57, Z33, Q77, Z32],
        #         [Z23, Z23, Z23, Z23, Z23, Z23, Z23, self.__clk_q],
        #     ]
        # )

        # idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16]
        # self.__Q = (self.__Q[idx, :])[:, idx]

        # self.__Q = np.block(
        #     [
        #         [Srg * I3, Z33, Z33, Z33, Z33, Z32],
        #         [Z33, Sra * I3, Z33, Z33, Z33, Z32],
        #         [Z33, Z33, Z33, Z33, Z33, Z32],
        #         [Z33, Z33, Z33, Sbad * I3, Z33, Z32],
        #         [Z33, Z33, Z33, Z33, Sbgd * I3, Z32],
        #         [Z23, Z23, Z23, Z23, Z23, self.__clk_q],
        #     ]
        # )

    #! === Observation Matrix ===
    def __generate_H(self, emitter_states: dict, pos: np.ndarray, vel: np.ndarray):
        N = len(emitter_states)

        u = np.zeros((N, 3))
        w = np.zeros((N, 3))
        Z = np.zeros((N, 3))
        z1 = np.zeros(N)
        i1 = np.ones(N)
        for j, emitter in enumerate(emitter_states.values()):
            dr = pos - emitter.pos
            dv = vel - emitter.vel
            r = norm(dr)
            u[j, :] = dr / r
            w[j, :] = -np.cross(u[j, :], np.cross(u[j, :], dv / r))
            # w[j, :] = (dv * r**2 - dr * (dv @ dr)) / r**3
        # self.__H = np.column_stack((Z, u, w, Z, Z, Z, Z, z1, i1))
        self.__H = np.column_stack((Z, u, w, Z, Z, z1, i1))
        # self.__H = np.column_stack((Z, u, w, z1, i1))
        self.__range_unit_vectors = u
        self.__rate_unit_vectors = w

    #! === Measurement Noise Covariance ===
    def __generate_R(self):
        self.__R = 10.1 * prange_rate_residual_var(self.cn0, 0.02, self.wavelength)

    #! === doppler measurement ===
    def __doppler(self, pos, vel, clk_drift, sv_pos, sv_vel):
        dr = pos - sv_pos
        dv = vel - sv_vel
        u = dr / np.linalg.norm(dr)
        return (dv @ u) + clk_drift * SPEED_OF_LIGHT  # / (1 + clk_drift)

    #! === acumulated delta range measurement ===
    def __adr(self, pos, clk_bias, sv_pos):
        return norm(sv_pos - pos) + SPEED_OF_LIGHT * clk_bias

    # --------------------------------------------------------------------------------------------------#
    #! === Estimate CN0 ===
    def __estimate_cn0(self):
        self.__cn0_ip_buffer.append(self.__correlators.IP)
        self.__cn0_qp_buffer.append(self.__correlators.QP)
        self.__cn0_counter += 1
        if self.__cn0_counter == self.__cn0_buffer_len:
            ip_tmp = np.array(self.__cn0_ip_buffer, dtype=np.double)
            qp_tmp = np.array(self.__cn0_qp_buffer, dtype=np.double)
            for i in range(self.cn0.size):
                self.cn0[i] = CN0_m2m4_estimator(ip_tmp[:, i], qp_tmp[:, i], self.cn0[i], self.T_rcvr)
                # self.cn0[i] = CN0_beaulieu_estimator(
                #     ip_tmp[:,i],
                #     qp_tmp[:,i],
                #     self.cn0[i],
                #     self.T_rcvr,
                #     False,
                #   )
            self.__cn0_counter = 0
            self.__cn0_ip_buffer = []
            self.__cn0_qp_buffer = []

    #! === Initialize Receiver Covariance Matrix ===
    def __initialize_covariance(self, emitter_states: dict):
        delta_diag_P = np.diag(self.rx_cov)

        self.__generate_H(emitter_states, self.__r_eb_e, self.__v_eb_e)
        H = self.__H
        R = np.diag(self.__R)
        k = 0
        while np.any(delta_diag_P > 1e-4) and (k < 500):
            previous_P = self.rx_cov.copy()

            self.rx_cov = self.__F @ self.rx_cov @ self.__F.T + self.__Q
            K = self.rx_cov @ H.T @ inv(H @ self.rx_cov @ H.T + R)
            self.rx_cov = (I - K @ H) @ self.rx_cov @ (I - K @ H).T + K @ R @ K.T

            delta_diag_P = np.abs(np.diag(previous_P - self.rx_cov))
            k += 1

        # self.rx_cov *= 10


# # * schmidt-kalman filter matrices
# # time-correlated system and measurement noise (consider states)
# self.__W = np.zeros((12, 12))
# self.__Qw = np.diag(
#     (self.__Sbad * self.T).tolist()
#     + (self.__Sbgd * self.T).tolist()
#     + ((1 - np.exp(-2 * self.__beta_acc)) * self.__Sra).tolist()
#     + ((1 - np.exp(-2 * self.__beta_gyr)) * self.__Sra).tolist()
# )
# self.__Phiw = np.diag([1] * 6 + np.exp(-self.__beta_acc).tolist() + np.exp(-self.__beta_gyr).tolist())

# # correlation between unestimated parameters and states
# self.__U = np.zeros((23, 12))
# I3, Z3 = np.eye(3), np.zeros((3, 3))
# self.__Psi = np.block(
#     [
#         [Z3, I3, Z3, I3],
#         [I3, Z3, I3, Z3],
#         [I3 * self.T, Z3, I3 * self.T, Z3],
#         [Z3, Z3, Z3, Z3],
#         [Z3, Z3, Z3, Z3],
#         [Z3, Z3, Z3, Z3],
#         [Z3, Z3, Z3, Z3],
#         [Z23, Z23, Z23, Z23],
#     ]
# )

# # * schmidt time update
# self.__clk_bias += self.__clk_drift * self.T
# P, Q, Phi = self.rx_cov, self.__Q, self.__F
# W, U, Qu, Phi_w, Psi = self.__W, self.__U, self.__Phiw, self.__Qw, self.__Psi
# self.rx_cov = Phi @ P @ Phi.T + Phi @ U @ Psi.T + Psi @ U.T @ Phi.T + Psi @ W @ Psi.T + Q
# self.__W = Phi_w @ W @ Phi_w.T + Qu
# self.__U = Phi @ U @ Phi_w.T + Psi @ W @ Phi_w.T

# # * schmidt measurement update
# J = np.ones((dy.size, 12))
# P, H, R = self.rx_cov, self.__H, self.__R
# W, U = self.__W, self.__U
# K = (P @ H.T + U @ J.T) @ inv(H @ P @ H.T + H @ U @ J.T + J @ U.T @ H.T + J @ W @ J.T + R)
# self.rx_cov = (I - K @ H) @ P - K @ J @ U.T
# self.__U = (I - K @ H) @ U - K @ J @ W
