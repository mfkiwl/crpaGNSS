"""
|====================================== crpa_imu_gnss_ecef.py =====================================|
|                                                                                                  |
|   Property of Daniel Sturdivant. Unauthorized copying of this file via any medium would be       |
|   super sad and unfortunate for me. Proprietary and confidential.                                |
|                                                                                                  |
| ------------------------------------------------------------------------------------------------ |
|                                                                                                  |
|   @file     charlizard/navigators/crpa_imu_gnss_ecef.py                                          |
|   @brief    Deep integration simulation class with CRPA to simulate attitude updates.            |
|   @author   Daniel Sturdivant <sturdivant20@gmail.com>                                           |
|   @date     June 2024                                                                            |
|                                                                                                  |
|==================================================================================================|
"""

import numpy as np
import navtools as nt
import navsim as ns
from scipy.linalg import norm, inv, block_diag
from dataclasses import dataclass

from charlizard.models.bpsk_correlator import *
from charlizard.estimators.discriminator import *
from charlizard.estimators.lock_detectors import *

TWO_PI = 2 * np.pi  #! [rad]
LIGHT_SPEED = 299792458.0  #! [m/s]
OMEGA_EARTH = 7.2921151467e-5
OMEGA_IE = np.array([0.0, 0.0, OMEGA_EARTH], dtype=np.double)
OMEGA_IE_E = nt.skew(OMEGA_IE)
R2D = 180.0 / np.pi
LLA_R2D = np.array([R2D, R2D, 1.0])

I3 = np.eye(3)
Z3 = np.zeros((3, 3))
Z32 = np.zeros((3, 2))
Z23 = Z32.T
O3 = np.ones(3)


# deep integration config
@dataclass
class DIConfig:
    T: float  #! integration period [s]
    innovation_stdev: float  #! normalized innovation filter threshold
    cn0_buffer_len: int  #! number of correlator outputs to use in cn0 estimation
    cn0: np.ndarray  #! initial receiver cn0
    tap_spacing: np.ndarray | float  #! (early, prompt, late) correlator tap/chip spacing
    chip_width: np.ndarray | float  #! initial signal chip widths [m]
    wavelength: np.ndarray | float  #! initial carrier wavelengths [m]
    pos: np.ndarray  #! initial receiver LLA position
    vel: np.ndarray  #! initial receiver ENU velocity
    att: np.ndarray  #! initial receiver RPY attitude
    clock_bias: float  #! initial receiver clock bias
    clock_drift: float  #! initial receiver clock drift
    clock_type: str | ns.error_models.NavigationClock  #! receiver oscillator type
    imu_type: str | ns.error_models.IMU  #! imu specs, only used if order==1
    TOW: float  #! initial GPS time of week
    ant_body_pos: np.ndarray
    frame: str
    mode: int  #! 0: DI, 1: DI + CRPA, 2: VT, 3: VT + CRPA


class CRPA_IMU_GNSS:
    def __init__(self, conf: DIConfig):
        # init
        self.dt = conf.T
        self.chip_freq = LIGHT_SPEED / conf.chip_width
        self.chip_width = conf.chip_width
        self.carrier_freq = LIGHT_SPEED / conf.wavelength
        self.wavelength = conf.wavelength
        self.tap_spacing = conf.tap_spacing
        self.TOW = conf.TOW
        self.frame = conf.frame
        self.innovation_stdev = conf.innovation_stdev

        # cn0 estimation init
        self.num_sv = conf.cn0.size
        self.cn0_dB = conf.cn0
        self.cn0 = 10.0 ** (conf.cn0 / 10.0)
        self.cn0_count = 0
        self.max_cn0_count = conf.cn0_buffer_len
        self.cn0_I_buf = np.zeros((self.cn0.size, self.max_cn0_count))
        self.cn0_I_buf_prev = None
        self.cn0_Q_buf = np.zeros((self.cn0.size, self.max_cn0_count))

        # antenna init
        self.Z = conf.ant_body_pos
        self.num_ant = self.Z.shape[0]
        self.mode = conf.mode
        if self.mode == 0:
            self.__make_dy = self.__make_vp_dy
            self.__make_H = self.__make_vp_H
            self.__make_R = self.__make_vp_R
        elif self.mode == 1:
            self.__make_dy = self.__make_crpa_dy
            self.__make_H = self.__make_crpa_H
            self.__make_R = self.__make_crpa_R

        # state init
        self.P = np.diag(np.block([5.01 * O3, 0.301 * O3, 0.03 * O3, 5e-2 * O3, 1e-3 * O3, 5.01, 0.301]) ** 2)
        self.x = np.block([np.zeros(15), conf.clock_bias, conf.clock_drift])
        self.I = np.eye(17)
        self.lla_pos = conf.pos / LLA_R2D
        self.r_eb_e = nt.lla2ecef(self.lla_pos)
        self.v_eb_e = conf.vel
        self.acc_bias = np.zeros(3)
        self.gyr_bias = np.zeros(3)
        self.clk_bias = conf.clock_bias
        self.clk_drift = conf.clock_drift
        self.is_cov_init = False

        # frame init (enu/ned)
        C_b_n = nt.euler2dcm(conf.att / R2D, "enu")
        self.C_b_e = nt.enu2ecefDcm(self.lla_pos) @ C_b_n
        # C_b_n = nt.euler2dcm(conf.att / R2D, "ned")
        # self.C_b_e = nt.ned2ecefDcm(self.lla_pos) @ C_b_n
        self.q_b_e = nt.dcm2quat(self.C_b_e)

        # dynamic model init
        if (conf.imu_type.casefold() is None) or (conf.imu_type.casefold() == "perfect"):
            imu_model = ns.error_models.get_imu_allan_variance_values("navigation")
        else:
            imu_model = ns.error_models.get_imu_allan_variance_values(conf.imu_type)
        if conf.clock_type.casefold() is None:
            clock_config = ns.error_models.get_clock_allan_variance_values("high_quality_tcxo")
        else:
            clock_config = ns.error_models.get_clock_allan_variance_values(conf.clock_type)
        self.S_rg = 1.1 * imu_model.N_gyr**2  # * self.T
        self.S_ra = 1.1 * imu_model.N_acc**2  # * self.T
        self.S_bad = 1.1 * imu_model.B_acc**2  # / self.T
        self.S_bgd = 1.1 * imu_model.B_gyr**2  # / self.T
        self.S_b = 1.1 * clock_config.h0 / 2
        self.S_d = 1.1 * clock_config.h2 * 2 * np.pi**2

    ##### *KALMAN FILTER* ##########################################################################

    #! --- propagate ---
    def propagate(
        self,
        sv_pos: np.ndarray,
        sv_vel: np.ndarray,
        f_ib_b: np.ndarray,
        w_ib_b: np.ndarray,
        dt: float,
        save: bool = False,
    ):
        # always mechanize
        fb = f_ib_b - self.acc_bias
        wb = w_ib_b - self.gyr_bias
        q, C, v, p = self.mechanize(wb, fb, dt)
        b = self.clk_bias + dt * self.clk_drift
        d = self.clk_drift
        # always calculate range and range rates for NCO
        dr = p[None, :] - sv_pos
        dv = v[None, :] - sv_vel
        az, el, r = nt.ecef2aer2d(sv_pos, p)
        u = dr / r[:, None]
        psr = r + b
        psr_dot = np.sum(u * dv, axis=1) + d

        # integration period complete
        if save:
            F21 = self.__make_A(fb, dt)
            self.__make_Q(F21, dt)
            self.P = self.A @ self.P @ self.A.T + self.Q
            self.x = self.A @ self.x

            self.lla_pos = nt.ecef2lla(p)
            self.r_eb_e = p
            self.v_eb_e = v
            self.C_b_e = C
            self.q_b_e = q
            self.clk_bias = b
            self.clk_drift = d
            self.psr = psr
            self.psr_dot = psr_dot
            # self.__make_unit_vec(az, el)
            self.dopp_unit_vec = -((dv * r[:, None] ** 2 - dr * np.sum(dv * dr, axis=1)[:, None]) / r[:, None] ** 3)
            self.unit_vec = -u

        return psr, psr_dot, az, el

    #! --- update ---
    def update(self):
        self.__make_H()
        self.__make_R()
        self.__make_dy()

        # try to achieve steady state covariance
        if self.is_cov_init == False:
            self.init_cov()
            self.is_cov_init = True

        # innovation filter
        S = self.H @ self.P @ self.H.T + self.R
        norm_dy = np.abs(self.dy / np.sqrt(S.diagonal()))
        mask = norm_dy < self.innovation_stdev
        self.dy = self.dy[mask]
        self.H = self.H[mask, :]
        self.R = np.diag(self.R.diagonal()[mask])

        # update
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ inv(S)
        # self.P -= K @ self.H @ self.P
        self.P -= K @ S @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        self.x += K @ self.dy

        # error state feedback
        self.r_eb_e -= self.x[0:3]
        self.lla_pos = nt.ecef2lla(self.r_eb_e)
        self.v_eb_e -= self.x[3:6]

        p0, p1, p2, p3 = 1.0, *(-self.x[6:9] / 2.0)
        q0, q1, q2, q3 = self.q_b_e
        self.q_b_e = np.array(
            [
                (p0 * q0 - p1 * q1 - p2 * q2 - p3 * q3),
                (p0 * q1 + p1 * q0 + p2 * q3 - p3 * q2),
                (p0 * q2 - p1 * q3 + p2 * q0 + p3 * q1),
                (p0 * q3 + p1 * q2 - p2 * q1 + p3 * q0),
            ]
        )
        self.q_b_e /= norm(self.q_b_e)
        self.C_b_e = nt.quat2dcm(self.q_b_e)

        self.acc_bias = self.x[9:12]
        self.gyr_bias = self.x[12:15]
        self.clk_bias = self.x[15]
        self.clk_drift = self.x[16]
        self.x[0:9] = np.zeros(9)

    #! --- update_correlators ---
    def update_correlators(self, c: Correlators):
        # TODO: add catches for when satellites disappear mid period
        # increment through cn0 estimator
        self.cn0_I_buf[:, self.cn0_count] = c.IP[:, 0]
        self.cn0_Q_buf[:, self.cn0_count] = c.QP[:, 0]
        self.cn0_count += 1
        if self.cn0_count == self.max_cn0_count:
            # re-estimate cn0 (moving average filter)
            cn0_pow = cn0_m2m4_estimate2d(self.cn0_I_buf, self.cn0_Q_buf, self.dt)
            o_15dB = 31.6 * np.ones(self.num_sv)  # dont use info less than 15 dB = 31.6 W
            cn0_pow = np.where(np.isnan(cn0_pow), self.cn0, cn0_pow)
            cn0_pow = np.where(cn0_pow < 31.6, o_15dB, cn0_pow)
            self.cn0 = 0.95 * self.cn0 + 0.05 * cn0_pow
            self.cn0_dB = 10.0 * np.log10(self.cn0)
            self.cn0_count = 0

        # calculate discriminators
        self.freq_err = fll_atan2_normalized(c.ip1[:, 0], c.ip2[:, 0], c.qp1[:, 0], c.qp2[:, 0], self.dt)
        self.chip_err = dll_nceml_normalized(c.IE[:, 0], c.QE[:, 0], c.IL[:, 0], c.QL[:, 0])

        # calculate discriminator variances
        self.freq_var = fll_variance(self.cn0, self.dt)
        self.chip_var = dll_variance(self.cn0, self.dt, self.tap_spacing)

        # only do phase if necessary
        if self.mode == 1:
            self.phase_err = pll_atan_normalized(c.IP, c.QP)
            self.phase_var = pll_variance(self.cn0, self.dt)

    #! --- init_cov ---
    def init_cov(self):
        # this allows for quick convergence
        # self.rx_cov = 10*solve_discrete_are(self.A.T, self.C.T, 0.5*(self.Q+self.Q.T), self.R)
        delta_diag_P = np.diag(self.P)

        while np.any(delta_diag_P > 1e-4):
            previous_P = self.P

            self.P = self.A @ self.P @ self.A.T + self.Q
            PCt = self.P @ self.H.T
            K = PCt @ inv(self.H @ PCt + self.R)
            self.P -= K @ self.H @ self.P

            delta_diag_P = np.diag(previous_P - self.P)

    ##### *INS SOLUTION* ###########################################################################

    #! ===IMU Mechanization (ECEF Frame) ===
    def mechanize(self, w_ib_b: np.ndarray, f_ib_b: np.ndarray, T: float):
        # copy old values
        q_old = self.q_b_e.copy()
        v_old = self.v_eb_e.copy()
        p_old = self.r_eb_e.copy()

        # rotational phase increment
        a_ib_b = w_ib_b * T
        mag_a_ib_b = norm(a_ib_b)

        # (Groves E.39) precision quaternion from old to new attitude
        if mag_a_ib_b > 0:
            q_new_old = np.array([np.cos(mag_a_ib_b / 2), *np.sin(mag_a_ib_b / 2) / mag_a_ib_b * a_ib_b])
        else:
            q_new_old = np.array([1, 0, 0, 0], dtype=np.double)

        p0, p1, p2, p3 = q_old
        q0, q1, q2, q3 = q_new_old
        a1, a2, a3 = OMEGA_IE * T / 2

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
        C_new = nt.quat2dcm(q_new)
        C_avg = nt.quat2dcm((q_old + q_new) / 2)

        # (Groves 5.85) specific force transformation body-to-ECEF
        f_ib_e = C_avg @ f_ib_b

        # (Groves 5.36) velocity update
        gravity, _ = nt.ecefg(self.r_eb_e)
        v_new = v_old + T * (f_ib_e + gravity - 2 * OMEGA_IE_E @ v_old)

        # (Groves 5.38) position update
        p_new = p_old + T * (v_new + v_old) / 2

        return q_new, C_new, v_new, p_new

    ##### *KALMAN MATRICES* ########################################################################

    #! --- make_A ---
    def __make_A(self, f_ib_b: np.ndarray, T: float):
        # radii of curvature and gravity
        r_es_e = nt.geocentricRadius(self.lla_pos)
        _, gamma = nt.ecefg(self.r_eb_e)

        C = self.C_b_e
        clk_f = np.array([[1, T], [0, 1]])

        # # (Groves 14.49/50/87) state transition matrix discretization
        f21 = -nt.skew(self.C_b_e @ f_ib_b)
        f23 = -np.outer(2 * gamma / r_es_e, self.r_eb_e / norm(self.r_eb_e))

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

        # fmt: off
        # self.A = np.block(
        #     [
        #         [F11, Z3, Z3, Z3, F15, Z32],
        #         [F21, F22, F23, F24, F25, Z32],
        #         [F31, F32, I3, F34, F35, Z32],
        #         [Z3, Z3, Z3, I3, Z3, Z32],
        #         [Z3, Z3, Z3, Z3, I3, Z32],
        #         [Z23, Z23, Z23, Z23, Z23, clk_f],
        #     ]
        # )
        self.A = np.block(
            [
                [ I3, F32, F31, F34, F35, Z32],
                [F23, F22, F21, F24, F25, Z32],
                [ Z3,  Z3, F11,  Z3, F15, Z32],
                [ Z3,  Z3,  Z3,  I3,  Z3, Z32],
                [ Z3,  Z3,  Z3,  Z3,  I3, Z32],
                [Z23, Z23, Z23, Z23, Z23, clk_f],
            ]
        )
        # fmt: on
        return f21

    #! --- make_Q ---
    def __make_Q(self, F21: np.ndarray, T: float):
        clk_q = (
            LIGHT_SPEED**2
            * 1.1
            * np.array(
                [
                    [self.S_b * T + self.S_d / 3.0 * T**3, self.S_d / 2.0 * T**2],
                    [self.S_d / 2.0 * T**2, self.S_d * T],
                ]
            )
        )

        Srg, Sbgd, Sra, Sbad = self.S_rg, self.S_bgd, self.S_ra, self.S_bad
        C = self.C_b_e
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

        # fmt: off
        # self.Q = np.block(
        #     [
        #         [Q33, Q32.T, Q31.T, Q34, Q35, Z32],
        #         [Q32.T, Q22, Q21, Q24, Q25, Z32],
        #         [Q31.T, Q21.T, Q11, Z3, Q15, Z32],
        #         [Q34.T, Q24, Z3, Q44, Z3, Z32],
        #         [Q32.T, Q52, Q15, Z3, Q55, Z32],
        #         [Z23, Z23, Z23, Z23, Z23, clk_q],
        #     ]
        # )
        self.Q = np.block(
            [
                [  Q33, Q32.T, Q31.T, Q34, Q35, Z32],
                [Q32.T,   Q22,   Q21, Q24, Q25, Z32],
                [Q31.T, Q21.T,   Q11,  Z3, Q15, Z32],
                [Q34.T,   Q24,    Z3, Q44,  Z3, Z32],
                [Q32.T,   Q52,   Q15,  Z3, Q55, Z32],
                [  Z23,   Z23,   Z23, Z23, Z23, clk_q],
            ]
        )
        # fmt: off

    #! --- make_vp_H ---
    def __make_vp_H(self):
        # position/velocity unit vectors (Groves 14.127/128)
        z = np.zeros((self.num_sv, 3))
        z1 = np.zeros((self.num_sv, 1))
        o1 = np.ones((self.num_sv, 1))
        self.H = np.block(
            [
                [self.unit_vec, z, z, z, z, o1, z1],
                [self.dopp_unit_vec, self.unit_vec, z, z, z, z1, o1],
            ]
        )

    #! --- make_crpa_H ---
    def __make_crpa_H(self):
        # position/velocity unit vectors (Groves 14.127/128)
        C_n_e = nt.enu2ecefDcm(self.lla_pos)
        hp = self.unit_vec @ C_n_e
        z = np.zeros((self.num_sv, 3))
        z1 = np.zeros((self.num_sv, 1))
        o1 = np.ones((self.num_sv, 1))

        # estimate spatial phase offsets and jacobian
        nm = self.num_sv * (self.num_ant - 1)  # reference element provides no benefit bc its position is [0, 0, 0]
        ant_xyz = (C_n_e.T @ self.C_b_e) @ self.Z[1:, :].T  # this means the jacobian and spatial phase are 0

        ha = np.zeros((nm, 3))
        self.spatial_phase = np.zeros((self.num_sv, self.num_ant - 1))
        for i in range(self.num_sv):
            for j in range(self.num_ant - 1):
                self.spatial_phase[i, j] = ant_xyz[:, j] @ hp[i, :] / self.wavelength[i]
                ha[i * (self.num_ant - 1) + j, :] = -TWO_PI / self.wavelength[i] * nt.skew(ant_xyz[:, j]) @ hp[i, :]

        # update phase errors, flatten, wrap
        tmp = self.phase_err[:, 1:] - self.spatial_phase - self.phase_err[:, 0][:, None]
        tmp[tmp >= 0.5] -= 1.0
        tmp[tmp <= -0.5] += 1.0
        self.phase_err = tmp.ravel()

        self.H = np.block(
            [
                [self.unit_vec, z, z, z, z, o1, z1],
                [self.dopp_unit_vec, self.unit_vec, z, z, z, z1, o1],
                [np.zeros((nm, 6)), ha @ C_n_e.T, np.zeros((nm, 8))],
            ]
        )

    #! --- make_vp_R ---
    def __make_vp_R(self):
        self.R = np.diag(
            1.1
            * np.block(
                [
                    self.chip_var * self.chip_width**2,
                    self.freq_var * self.wavelength**2,
                ],
            )
        )

    #! --- make_crpa_R ---
    def __make_crpa_R(self):
        self.R = np.diag(
            1.1
            * np.block(
                [
                    self.chip_var * self.chip_width**2,
                    self.freq_var * self.wavelength**2,
                    np.repeat(self.phase_var, self.num_ant - 1) * TWO_PI**2,
                ],
            )
        )

    #! --- make_vp_dy ---
    def __make_vp_dy(self):
        self.dy = np.block(
            [
                self.chip_width * self.chip_err,
                -self.wavelength * self.freq_err,
            ]
        )

    #! --- make_crpa_dy ---
    def __make_crpa_dy(self):
        self.dy = np.block(
            [
                self.chip_width * self.chip_err,
                -self.wavelength * self.freq_err,
                TWO_PI * self.phase_err,
            ]
        )

    ##### *RESULTS EXTRACTION* #####################################################################

    #! --- extract_states ---
    def extract_states(self):
        C_e_n = nt.ecef2enuDcm(self.lla_pos)
        vel = C_e_n @ self.v_eb_e
        rpy = nt.dcm2euler(C_e_n @ self.C_b_e, "enu") * R2D
        # rpy = nt.dcm2euler(nt.ecef2nedDcm(self.lla_pos) @ self.C_b_e, "ned") * R2D
        return self.lla_pos * LLA_R2D, vel, rpy, self.clk_bias, self.clk_drift

    #! --- extract_stds ---
    def extract_stds(self):
        C_e_n = nt.ecef2enuDcm(self.lla_pos)
        pos = np.sqrt(np.diag(C_e_n @ self.P[0:3, 0:3] @ C_e_n.T))
        vel = np.sqrt(np.diag(C_e_n @ self.P[3:6, 3:6] @ C_e_n.T))
        att = np.sqrt(np.diag(C_e_n @ self.P[6:9, 6:9] @ C_e_n.T)) * R2D
        cb = np.sqrt(self.P[15, 15])
        cd = np.sqrt(self.P[16, 16])
        return pos, vel, att, cb, cd

    #! --- extract_dops ---
    def extract_dops(self):
        C_e_n = nt.ecef2enuDcm(self.lla_pos)
        G = np.column_stack((self.unit_vec, np.ones(self.num_sv)))
        dop = inv(G.T @ G)
        dop[:3, :3] = C_e_n @ dop[:3, :3] @ C_e_n.T

        gdop = np.sqrt(np.abs(dop.trace()))
        pdop = np.sqrt(np.abs(dop[:3, :3].trace()))
        hdop = np.sqrt(np.abs(dop[:2, :2].trace()))
        vdop = np.sqrt(np.abs(dop[2, 2]))
        tdop = np.sqrt(np.abs(dop[3, 3]))
        return gdop, pdop, hdop, vdop, tdop, self.num_sv

    #! --- extract_cn0 ---
    def extract_cn0(self):
        return self.cn0_dB
