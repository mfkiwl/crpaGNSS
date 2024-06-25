"""
|======================================== crpa_imu_gnss.py ========================================|
|                                                                                                  |
|   Property of Daniel Sturdivant. Unauthorized copying of this file via any medium would be       |
|   super sad and unfortunate for me. Proprietary and confidential.                                |
|                                                                                                  |
| ------------------------------------------------------------------------------------------------ |
|                                                                                                  |
|   @file     charlizard/navigators/crpa_imu_gpss.py                                               |
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

        # state init
        self.P = np.diag(np.block([5.01 * O3, 0.301 * O3, 0.03 * O3, 5e-2 * O3, 1e-3 * O3, 5.01, 0.301]) ** 2)
        self.x = np.block([np.zeros(15), conf.clock_bias, conf.clock_drift])
        self.I = np.eye(17)
        self.lla_pos = conf.pos / LLA_R2D
        self.v_b_n = conf.vel
        self.acc_bias = np.zeros(3)
        self.gyr_bias = np.zeros(3)
        self.clk_bias = conf.clock_bias
        self.clk_drift = conf.clock_drift
        self.is_cov_init = False

        # frame init (enu/ned)
        self.C_b_n = nt.euler2dcm(conf.att / R2D, self.frame)
        self.q_b_n = nt.euler2quat(conf.att / R2D, self.frame)
        if self.frame.casefold() == "enu":
            self.__nav2ecefDcm = nt.enu2ecefDcm
            self.__gravity = nt.enug
            self.__h_sign = 1.0
            self.__nav_rotations = self.__enu_nav_rotations
            self.__get_T_r_p = self.__get_enu_T_r_p
            self.__make_sub_F = self.__make_enu_sub_F
            self.__get_vel_components = self.__get_enu_vel_components
            self.__make_unit_vec = self.__make_enu_unit_vec
        elif self.frame.casefold() == "ned":
            self.__nav2ecefDcm = nt.ned2ecefDcm
            self.__gravity = nt.nedg
            self.__h_sign = -1.0
            self.__nav_rotations = self.__ned_nav_rotations
            self.__get_T_r_p = self.__get_ned_T_r_p
            self.__make_sub_F = self.__make_ned_sub_F
            self.__get_vel_components = self.__get_ned_vel_components
            self.__make_unit_vec = self.__make_ned_unit_vec

        # measurement model init
        # self.use_multi_ant = conf.use_multi_ant
        self.mode = conf.mode
        if self.mode == 0:
            self.__make_dy = self.__make_vp_dy
            self.__make_H = self.__make_vp_H
            self.__make_R = self.__make_vp_R
        elif self.mode == 1:
            self.__make_dy = self.__make_crpa_dy
            self.__make_H = self.__make_crpa_H
            self.__make_R = self.__make_crpa_R

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
        q, C, v, p = self.mechanize(fb, wb, dt)
        b = self.clk_bias + dt * self.clk_drift
        d = self.clk_drift

        # conversion from geodetic/enu-ned to ecef (Groves 14.122)
        C_n_e = self.__nav2ecefDcm(p)
        ecef_p = nt.lla2ecef(p)
        ecef_v = C_n_e @ v

        # always calculate range and range rates for NCO
        dr = ecef_p[None, :] - sv_pos
        dv = ecef_v[None, :] - sv_vel
        az, el, r = nt.ecef2aer2d(sv_pos, ecef_p)
        u = dr / r[:, None]
        psr = r + b
        psr_dot = np.sum(u * dv, axis=1) + d

        # integration period complete
        if save:
            F21, T_r_p = self.__make_A(fb, dt)
            self.__make_Q(F21, T_r_p, dt)
            self.P = self.A @ self.P @ self.A.T + self.Q
            self.x = self.A @ self.x
            self.lla_pos = p
            self.v_b_n = v
            self.C_b_n = C
            self.q_b_n = q
            self.clk_bias = b
            self.clk_drift = d
            self.psr = psr
            self.psr_dot = psr_dot
            # self.__make_unit_vec(az, el)
            self.dopp_unit_vec = (-(dv * r[:, None] ** 2 - dr * np.sum(dv * dr, axis=1)[:, None]) / r[:, None] ** 3) @ C_n_e
            self.unit_vec = -u @ C_n_e

        return psr, psr_dot, az, el
        # return 0, 0

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
        lat, _, h = self.lla_pos
        Re, Rn, r_es_e = nt.radiiOfCurvature(lat)
        T_r_p = self.__get_T_r_p(h, lat, Re, Rn)
        self.lla_pos -= T_r_p @ self.x[0:3]
        self.v_b_n -= self.x[3:6]

        p0, p1, p2, p3 = 1.0, *(-self.x[6:9] / 2.0)
        q0, q1, q2, q3 = self.q_b_n
        self.q_b_n = np.array(
            [
                (p0 * q0 - p1 * q1 - p2 * q2 - p3 * q3),
                (p0 * q1 + p1 * q0 + p2 * q3 - p3 * q2),
                (p0 * q2 - p1 * q3 + p2 * q0 + p3 * q1),
                (p0 * q3 + p1 * q2 - p2 * q1 + p3 * q0),
            ]
        )
        self.q_b_n /= norm(self.q_b_n)
        self.C_b_n = nt.quat2dcm(self.q_b_n)

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
            # ip = np.ma.array(self.cn0_I_buf, mask=np.isnan(self.cn0_I_buf))
            # qp = np.ma.array(self.cn0_Q_buf, mask=np.isnan(self.cn0_Q_buf))
            # cn0_pow = cn0_m2m4_estimate2d(ip, qp, self.dt)
            cn0_pow = cn0_m2m4_estimate2d(self.cn0_I_buf, self.cn0_Q_buf, self.dt)
            o_15dB = 31.6 * np.ones(self.num_sv)  # dont use info less than 15 dB = 31.6 W
            cn0_pow = np.where(np.isnan(cn0_pow), self.cn0, cn0_pow)
            cn0_pow = np.where(cn0_pow < 31.6, o_15dB, cn0_pow)
            self.cn0 = 0.95 * self.cn0 + 0.05 * cn0_pow
            self.cn0_dB = 10.0 * np.log10(self.cn0)
            self.cn0_count = 0
        # if self.cn0_count == self.max_cn0_count:
        #     prompt = np.sqrt(self.cn0_I_buf**2 + self.cn0_Q_buf**2)  # account for non-phase-lock
        #     if self.cn0_I_buf_prev is not None:
        #         cn0_pow = cn0_beaulieu_estimate2d(prompt, self.cn0_I_buf_prev, self.dt)
        #         self.cn0 = 0.95 * self.cn0 + 0.05 * cn0_pow
        #         self.cn0_dB = 10.0 * np.log10(self.cn0)
        #     self.cn0_count = 0
        #     self.cn0_I_buf_prev = prompt
        # print(f"CN0 dB: {self.__cn0_dB.mean()}")

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

        # return self.phase_err, self.freq_err, self.chip_err

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

    #! --- mechanize ---
    def mechanize(self, f_ib_b: np.ndarray, w_ib_b: np.ndarray, dt: float):
        # copy old values
        q_old = self.q_b_n.copy()
        v_old = self.v_b_n.copy()
        p_old = self.lla_pos.copy()

        lat, lon, h = p_old
        Re, Rn, _ = nt.radiiOfCurvature(lat)
        ve, vn, vh = self.__get_vel_components(v_old)
        w_en_n, w_ie_n = self.__nav_rotations(lat, h, ve, vn, Re, Rn)

        # rotational phase increment
        a_ib_b = w_ib_b * dt
        mag_a_ib_b = norm(a_ib_b)

        # (Groves E.39) precision quaternion from old to new attitude
        if mag_a_ib_b > 0:
            q_new_old = np.array([np.cos(mag_a_ib_b / 2), *np.sin(mag_a_ib_b / 2) / mag_a_ib_b * a_ib_b])
        else:
            q_new_old = np.array([1.0, 0.0, 0.0, 0.0])

        # (Groves E.40) quaternion multiplication attitude update
        p0, p1, p2, p3 = self.q_b_n
        q0, q1, q2, q3 = q_new_old
        a1, a2, a3 = (w_ie_n + w_en_n) * dt / 2.0
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
        C_new = nt.quat2dcm(q_new)
        C_avg = nt.quat2dcm((q_old + q_new) / 2.0)

        # (Groves 5.86) specific force transformation body-to-ECEF
        f_ib_n = C_avg @ f_ib_b

        # (Groves 5.54) velocity update
        g = self.__gravity(p_old)
        v_new = v_old + (f_ib_n + g - (nt.skew(w_en_n) + 2.0 * nt.skew(w_ie_n)) @ v_old) * dt
        ve_plus, vn_plus, vh_plus = self.__get_vel_components(v_new)

        # (Groves 5.56) position update
        h_plus = h + self.__h_sign * (vh + vh_plus) * dt / 2.0
        lat_plus = lat + (vn / (Rn + h) + vn_plus / (Rn + h_plus)) * dt / 2.0
        Re_plus = nt.transverseRadius(lat_plus)
        lon_plus = lon + ((ve / ((Re + h) * np.cos(lat))) + (ve_plus / ((Re_plus + h_plus) * np.cos(lat_plus)))) * dt / 2.0
        p_new = np.array([lat_plus, lon_plus, h_plus])

        return q_new, C_new, v_new, p_new

    #! --- enu_nav_rotations ---
    def __enu_nav_rotations(self, lat, h, ve, vn, Re, Rn):
        w_en_n = np.array([-vn / (Rn + h), ve / (Re + h), ve * np.tan(lat) / (Re + h)])  # Groves 5.44
        w_ie_n = np.array([0.0, OMEGA_EARTH * np.cos(lat), OMEGA_EARTH * np.sin(lat)])  # Groves 2.123
        return w_en_n, w_ie_n

    #! --- ned_nav_rotations ---
    def __ned_nav_rotations(self, lat, h, ve, vn, Re, Rn):
        w_en_n = np.array([ve / (Re + h), -vn / (Rn + h), ve * np.tan(lat) / (Re + h)])  # Groves 5.44
        w_ie_n = np.array([OMEGA_EARTH * np.cos(lat), 0.0, -OMEGA_EARTH * np.sin(lat)])  # Groves 2.123
        return w_en_n, w_ie_n

    #! --- get_enu_T_r_p ---
    def __get_enu_T_r_p(self, h, lat, Re, Rn):
        T_r_p = np.array([[0.0, 1.0 / ((Re + h) * np.cos(lat)), 0.0], [1.0 / (Rn + h), 0.0, 0.0], [0.0, 0.0, 1.0]])
        return T_r_p

    #! --- get_ned_T_r_p ---
    def __get_ned_T_r_p(self, h, lat, Re, Rn):
        T_r_p = np.diag([1.0 / (Rn + h), 1.0 / ((Re + h) * np.cos(lat)), -1.0])
        return T_r_p

    #! --- get_enu_vel_components ---
    def __get_enu_vel_components(self, v: np.ndarray):
        return v[0], v[1], v[2]

    #! --- get_ned_vel_components ---
    def __get_ned_vel_components(self, v: np.ndarray):
        return v[1], v[0], v[2]

    ##### *KALMAN MATRICES* ########################################################################

    #! --- make_enu_unit_vec ---
    def __make_enu_unit_vec(self, az, el):
        self.unit_vec = np.array([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)]).T

    #! --- make_ned_unit_vec ---
    def __make_ned_unit_vec(self, az, el):
        self.unit_vec = np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), -np.sin(el)]).T

    #! --- make_enu_sub_F ---
    def __make_enu_sub_F(self, f_ib_b: np.ndarray):
        lat, lon, h = self.lla_pos
        ve, vn, vu = self.v_b_n
        Re, Rn, r_es_e = nt.radiiOfCurvature(lat)
        g0 = nt.somigliana(lat)
        w_ie = OMEGA_EARTH

        sL = np.sin(lat)
        cL = np.cos(lat)
        tL = sL / cL
        Re_h = Re + h
        Rn_h = Rn + h
        Re_h2 = Re_h * Re_h
        Rn_h2 = Rn_h * Rn_h

        # Groves Ch. 14.2.4 and Appendix I.2.2
        # fmt: off
        w_en_n, w_ie_n = self.__enu_nav_rotations(lat, h, ve, vn, Re, Rn)
        F11 = -nt.skew(w_en_n + w_ie_n)
        # F11 = np.array(
        #     [
        #         [w_ie * sL - ve * tL / Re_h,                         0.0, w_ie * cL + ve / Re_h],
        #         [                       0.0, -w_ie * sL + ve * tL / Re_h,             vn / Rn_h],
        #         [                 vn / Rn_h,       w_ie * cL + ve / Rn_h,                   0.0],
        #     ]
        # )
        F12 = np.array([[1.0 / Rn_h, 0.0, 0.0], [0.0, -1.0 / Re_h, 0.0], [0.0, -tL / Re_h, 0.0]])
        F13 = np.array(
            [
                [                             0.0, 0.0,      -vn / Rn_h],
                [                       w_ie * sL, 0.0,      ve / Re_h2],
                [-w_ie * cL - ve / (Re_h * cL**2), 0.0, ve * tL / Re_h2],
            ]
        )
        F21 = -nt.skew(self.C_b_n @ f_ib_b)
        F22 = np.array(
            [
                [2.0 * w_ie * sL + ve * tL / Re_h,               (vu + vn * tL) / Re_h, 2.0 * w_ie * cL + ve / Re_h],
                [                       vu / Re_h, -2.0 * (w_ie * sL + ve * tL / Re_h),                   vn / Rn_h],
                [                 2.0 * vn / Rn_h,       2.0 * (w_ie * cL + ve / Re_h),                         0.0],
            ]
        )
        F23 = np.array(
            [
                [2.0 * w_ie * (vn * cL - vu * sL) + ve * vn / (Re_h * cL**2), 0.0,                         -ve * (vn * tL + vu) / Re_h2],
                [             -2.0 * w_ie * ve * cL - ve**2 / (Re_h * cL**2), 0.0,                 ve**2 * tL / Re_h2 - ve * vu / Rn_h2],
                [                                       2.0 * w_ie * ve * sL, 0.0, -(ve**2) / Re_h2 - vn**2 / Rn_h2 + 2.0 * g0 / r_es_e],
            ]
        )
        F32 = np.array([[0.0, 1.0 / (Re_h * cL), 0.0], [1.0 / Rn_h, 0.0, 0.0], [0.0, 0.0, 1.0]])
        F33 = np.array([[ve * sL / (Re_h * cL**2), 0.0, -ve / (Re_h2 * cL)], [0.0, 0.0, -vn / Rn_h2], [0.0, 0.0, 0.0]])
        T_r_p = F32
        # fmt: on

        return F11, F12, F13, F21, F22, F23, F32, F33, T_r_p

    #! --- make_ned_sub_F ---
    def __make_ned_sub_F(self, f_ib_b: np.ndarray):
        lat, lon, h = self.lla_pos
        vn, ve, vd = self.v_b_n
        Re, Rn, r_es_e = nt.radiiOfCurvature(lat)
        g0 = nt.somigliana(lat)
        w_ie = OMEGA_EARTH

        sL = np.sin(lat)
        cL = np.cos(lat)
        tL = sL / cL
        Re_h = Re + h
        Rn_h = Rn + h
        Re_h2 = Re_h * Re_h
        Rn_h2 = Rn_h * Rn_h

        # Groves Ch. 14.2.4 and Appendix I.2.2
        # fmt: off
        w_en_n, w_ie_n = self.__ned_nav_rotations(lat, h, ve, vn, Re, Rn)
        F11 = -nt.skew(w_en_n + w_ie_n)
        F12 = np.array([[0.0, -1.0 / Re_h, 0.0], [1.0 / Rn_h, 0.0, 0.0], [0.0, tL / Re_h, 0.0]])
        F13 = np.array(
            [
                [                      w_ie * sL, 0.0,       ve / Re_h2],
                [                            0.0, 0.0,      -vn / Rn_h2],
                [w_ie * cL + ve / (Re_h * cL**2), 0.0, -ve * tL / Re_h2],
            ]
        )
        F21 = -nt.skew(self.C_b_n @ f_ib_b)
        F22 = np.array(
            [
                [                       vd / Rn_h, -2.0 * (w_ie * sL + ve * tL / Re_h),                   vn / Rn_h],
                [2.0 * w_ie * sL + ve * tL / Re_h,               (vn * tL + vd) / Re_h, 2.0 * w_ie * cL + ve / Re_h],
                [                -2.0 * vn / Rn_h,      -2.0 * (w_ie * cL + ve / Re_h),                         0.0],
            ]
        )
        F23 = np.array(
            [
                [             -(ve**2) / (Re_h * cL**2) - 2 * ve * w_ie * cL, 0.0,             -vn * vd / Rn_h2 + ve**2 * tL / Re_h2],
                [vn * ve / (Re_h * cL**2) + 2.0 * w_ie * (vn * cL - vd * sL), 0.0,                      -ve * (vn * tL + vd) / Re_h2],
                [                                       2.0 * ve * w_ie * sL, 0.0, ve**2 / Re_h2 + vn**2 / Rn_h2 - 2.0 * g0 / r_es_e],
            ]
        )
        F32 = np.array([[1.0 / Rn_h, 0.0, 0.0], [0.0, 1.0 / (Re_h * cL), 0.0], [0.0, 0.0, -1.0]])
        F33 = np.array([[0.0, 0.0, -vn / Rn_h2], [ve * sL / (Re_h * cL**2), 0.0, -ve / (Re_h2 * cL)], [0.0, 0.0, 0.0]])
        T_r_p = F32
        # fmt: on

        return F11, F12, F13, F21, F22, F23, F32, F33, T_r_p

    #! --- make_A ---
    def __make_A(self, f_ib_b: np.ndarray, dt: float):
        C = self.C_b_n
        clk_f = np.array([[1.0, dt], [0.0, 1.0]])

        F11, F12, F13, F21, F22, F23, F32, F33, T_r_p = self.__make_sub_F(f_ib_b)
        F_11_11 = F11 @ F11
        F_11_13 = F11 @ F13
        F_13_33 = F13 @ F33
        F_21_11 = F21 @ F11
        F_22_21 = F22 @ F21
        F_32_21 = F32 @ F21
        F_32_22 = F32 @ F22
        F_33_32 = F33 @ F32
        F_12_22_23 = F11 @ F12 + F12 @ F22 + F13 @ F32
        F_3222_3332 = F_32_22 + F_33_32

        A11 = I3 + F11 * dt + (F_11_11 * dt**2) / 2 + (F_11_11 @ F11 * dt**3) / 6
        A12 = F12 * dt + (F_12_22_23 * dt**2) / 2 + ((F11 @ F_12_22_23 + F13 @ (F32 @ F22 + F33 @ F32)) * dt**3) / 6
        A13 = F13 * dt + ((F_11_13 + F12 @ F23 + F_13_33) * dt**2) / 2 + ((F_11_11 @ F13 + F_11_13 @ F33) * dt**3) / 6
        A14 = (F12 @ C * dt**2) / 2 + (F_12_22_23 @ C * dt**3) / 6
        A15 = C * dt + (F11 @ C * dt**2) / 2 + (F_11_11 @ C * dt**3) / 6
        A21 = F21 * dt + ((F_21_11 + F_22_21) * dt**2) / 2 + ((F21 @ F_11_11 + F22 @ F_21_11) * dt**3) / 6
        A22 = I3 + F22 * dt
        A23 = (
            F23 * dt
            + ((F21 @ F13 + F22 @ F23 + F23 @ F33) * dt**2) / 2
            + ((F21 @ F_11_13 + F21 @ F_13_33 + F_22_21 @ F13) * dt**3) / 6
        )
        A24 = C * dt + (F22 @ C * dt**2) / 2
        A25 = (F21 @ C * dt**2) / 2 + ((F_21_11 + F22 @ F11) @ C * dt**3) / 6
        A31 = (F_32_21 * dt**2) / 2 + ((F32 @ F_21_11 + F32 @ F_22_21) * dt**3) / 6
        A32 = F32 * dt + (F_3222_3332 * dt**2) / 2 + (F33 @ F_32_22 * dt**3) / 6
        A33 = I3 + F33 * dt
        A34 = (F32 @ C * dt**2) / 2 + (F_3222_3332 @ C * dt**3) / 6
        A35 = (F_32_21 @ C * dt**3) / 6

        # fmt: off
        self.A = np.block(
            [
                [A33, A32, A31, A34, A35,   Z32],
                [A23, A22, A21, A24, A25,   Z32],
                [A13, A12, A11, A14, A15,   Z32],
                [ Z3,  Z3,  Z3,  I3,  Z3,   Z32],
                [ Z3,  Z3,  Z3,  Z3,  I3,   Z32],
                [Z23, Z23, Z23, Z23, Z23, clk_f],
            ]
        )
        # self.A = np.block(
        #     [
        #         [I3 + F33 * dt,      F32 * dt,            Z3,     Z3,     Z3,   Z32],
        #         [     F23 * dt, I3 + F22 * dt,      F21 * dt, C * dt,     Z3,   Z32],
        #         [     F13 * dt,      F12 * dt, I3 + F11 * dt,     Z3, C * dt,   Z32],
        #         [           Z3,            Z3,            Z3,     I3,     Z3,   Z32],
        #         [           Z3,            Z3,            Z3,     Z3,     I3,   Z32],
        #         [          Z23,           Z23,           Z23,    Z23,    Z23, clk_f],
        #     ]
        # )
        # fmt: on
        return F21, T_r_p

    #! --- make_Q ---
    def __make_Q(self, F21: np.ndarray, T_r_p: np.ndarray, dt: float):
        C, S_rg, S_ra, S_bgd, S_bad, S_b, S_d = (
            self.C_b_n,
            self.S_rg,
            self.S_ra,
            self.S_bgd,
            self.S_bad,
            self.S_b,
            self.S_d,
        )
        F_21_21t = F21 @ F21.T
        F_21_C = F21 @ C

        clk_q = LIGHT_SPEED**2 * np.array([[S_b * dt + S_d / 3.0 * dt**3, S_d / 2.0 * dt**2], [S_d / 2.0 * dt**2, S_d * dt]])

        Q11 = (S_rg * dt + (S_bgd * dt**3) / 3) * I3
        Q15 = (S_bgd / 2 * dt**2) * C
        Q21 = ((S_rg * dt**2) / 2 + (S_bgd * dt**4) / 4) * F21
        Q22 = (S_ra * dt + (S_bad * dt**3) / 3) * I3 + ((S_rg * dt**3) / 3 + (S_bgd * dt**5) / 5) * F_21_21t
        Q24 = (S_bad / 2 * dt**2) * C
        Q25 = (S_bad / 3 * dt**3) * F_21_C
        Q31 = ((S_rg * dt**3) / 3 + (S_bgd * dt**5) / 5) * (T_r_p @ F21)
        Q32 = ((S_ra * dt**2) / 2 + (S_bad * dt**4) / 4) * T_r_p + ((S_rg * dt**4) / 4 + (S_bgd * dt**6) / 6) * (T_r_p @ F_21_21t)
        Q33 = ((S_ra * dt**3) / 3 + (S_bad * dt**5) / 5) * (T_r_p @ T_r_p) + (
            (S_rg * dt**5) / 5 + (S_bgd * dt**7) / 7
        ) * (T_r_p @ F_21_21t @ T_r_p)
        Q34 = (S_bad / 3 * dt**3) * (T_r_p @ C)
        Q35 = (S_bgd / 4 * dt**4) * (T_r_p @ F_21_C)
        Q44 = S_bad * dt * I3
        Q52 = (S_bgd / 3 * dt**3) * (F21.T @ C)
        Q55 = S_bgd * dt * I3

        # fmt: off
        self.Q = np.block(
            [
                [Q33  , Q32  , Q31, Q34, Q35, Z32  ],
                [Q32.T, Q22  , Q21, Q24, Q25, Z32  ],
                [Q31.T, Q21.T, Q11, Z3 , Q15, Z32  ],
                [Q34.T, Q24  , Z3 , Q44, Z3 , Z32  ],
                [Q35.T, Q52  , Q15, Z3 , Q55, Z32  ],
                [Z23  , Z23  , Z23, Z23, Z23, clk_q],
            ]
        )
        # fmt: on

    #! --- make_vp_H ---
    def __make_vp_H(self):
        # position/velocity unit vectors (Groves 14.127/128)
        hp = self.unit_vec
        hv = self.dopp_unit_vec
        z = np.zeros((self.num_sv, 3))
        z1 = np.zeros((self.num_sv, 1))
        o1 = np.ones((self.num_sv, 1))
        self.H = np.block(
            [
                [hp, z, z, z, z, o1, z1],
                [z, hp, z, z, z, z1, o1],
            ]
        )

    #! --- make_crpa_H ---
    def __make_crpa_H(self):
        # position/velocity unit vectors (Groves 14.127/128)
        hp = self.unit_vec
        hv = self.dopp_unit_vec
        z = np.zeros((self.num_sv, 3))
        z1 = np.zeros((self.num_sv, 1))
        o1 = np.ones((self.num_sv, 1))

        # estimate spatial phase offsets and jacobian
        nm = self.num_sv * (self.num_ant - 1)  # reference element provides no benefit bc its position is [0, 0, 0]
        ant_xyz = self.C_b_n @ self.Z[1:, :].T  # this means the jacobian and spatial phase are 0

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
                [hp, z, z, z, z, o1, z1],
                [hv, hp, z, z, z, z1, o1],
                [np.zeros((nm, 6)), ha, np.zeros((nm, 8))],
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
        vel = np.array(self.__get_vel_components(self.v_b_n))
        rpy = nt.dcm2euler(self.C_b_n, self.frame) * R2D
        return self.lla_pos * LLA_R2D, vel, rpy, self.clk_bias, self.clk_drift

    #! --- extract_stds ---
    def extract_stds(self):
        pos = np.sqrt(np.diag(self.P[0:3, 0:3]))
        vel = np.array(self.__get_vel_components(np.sqrt(np.diag(self.P[3:6, 3:6]))))
        att = np.sqrt(np.diag(self.P[6:9, 6:9])) * R2D
        cb = np.sqrt(self.P[15, 15])
        cd = np.sqrt(self.P[16, 16])
        return pos, vel, att, cb, cd

    #! --- extract_dops ---
    def extract_dops(self):
        G = np.column_stack((self.unit_vec, np.ones(self.num_sv)))
        dop = inv(G.T @ G)
        dop[:3, :3] = dop[:3, :3]

        gdop = np.sqrt(np.abs(dop.trace()))
        pdop = np.sqrt(np.abs(dop[:3, :3].trace()))
        hdop = np.sqrt(np.abs(dop[:2, :2].trace()))
        vdop = np.sqrt(np.abs(dop[2, 2]))
        tdop = np.sqrt(np.abs(dop[3, 3]))
        return gdop, pdop, hdop, vdop, tdop, self.num_sv

    #! --- extract_cn0 ---
    def extract_cn0(self):
        return self.cn0_dB
