'''
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
'''

import numpy as np
from scipy.linalg import norm, inv

from charlizard.navigators.structures import GNSSINSConfig
from charlizard.models.correlators import Correlators
from charlizard.models.discriminator import prange_rate_residual_var, prange_residual_var, fll_error, dll_error
from charlizard.models.lock_detectors import CN0_m2m4_estimator, CN0_beaulieu_estimator

from navsim.error_models import get_clock_allan_variance_values, get_imu_allan_variance_values

from navtools.conversions import ecef2lla, ecef2enuDcm, ecef2nedDcm, euler2dcm, dcm2quat, quat2dcm, dcm2euler, skew
from navtools.measurements import ned2ecefg, ecefg, geocentricRadius
from navtools.constants import GNSS_OMEGA_EARTH, SPEED_OF_LIGHT

R2D = 180 / np.pi
LLA_R2D = np.array([R2D, R2D, 1.0], dtype=np.double)
I3 = np.eye(3, dtype=np.double)
Z33 = np.zeros((3,3), dtype=np.double)
Z32 = np.zeros((3,2), dtype=np.double)
Z23 = np.zeros((2,3), dtype=np.double)
OMEGA_IE = np.array([0.0, 0.0, GNSS_OMEGA_EARTH], dtype=np.double)
OMEGA_IE_E = skew(OMEGA_IE)

class GnssIns:
  @property
  def extract_states(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # update ENU state
    att = dcm2euler((self.__C_e_ned @ self.__C_b_e).T) * R2D
    # att = dcm2euler((self.__C_e_n @ self.__C_b_e).T) * R2D
    vel = self.__C_e_n @ self.__v_eb_e
    pos = self.__C_e_n @ (self.__r_eb_e - self.__ecef0)
    lla = ecef2lla(self.__r_eb_e) * LLA_R2D
    clk = self.rx_state[-2:]
    return pos, vel, att, lla, clk
  
  @property
  def extract_stds(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # extract ENU standard deviations
    att = np.sqrt(np.diag(self.__C_e_n @ self.rx_cov[0:3,0:3] @ self.__C_e_n.T)) * R2D
    # att = dcm2euler((self.__C_e_ned @ self.rx_cov[0:3,0:3]).T) * R2D
    vel = np.sqrt(np.diag(self.__C_e_n @ self.rx_cov[3:6,3:6] @ self.__C_e_n.T))
    pos = np.sqrt(np.diag(self.__C_e_n @ self.rx_cov[6:9,6:9] @ self.__C_e_n.T))
    clk = np.sqrt(np.diag(self.rx_cov[-2:,-2:]))
    return pos, vel, att, clk
  
  @property
  def extract_dops(self) -> tuple[float, float, float, float, float, int]:
    n = self.__range_unit_vectors.shape[0]
    i1 = np.ones(n)
    H = np.block([self.__range_unit_vectors, i1[:,None]])
    dop = inv(H.T @ H)
    dop[:3,:3] = self.__C_e_n @ dop[:3,:3] @ self.__C_e_n.T
    gdop = np.sqrt(dop.trace())
    pdop = np.sqrt(dop[:3,:3].trace())
    hdop = np.sqrt(dop[:2,:2].trace())
    vdop = np.sqrt(dop[2,2])
    tdop = np.sqrt(dop[3,3])
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
    self.cn0 = config.cn0
    self.__innovation_std = config.innovation_stdev
    self.__tap_spacing = config.tap_spacing
    self.__coupling = config.coupling
    self.__chip_width = SPEED_OF_LIGHT / 1.023e6
    self.__wavelength = SPEED_OF_LIGHT / 1575.42e6
    
    # cn0 estimation config
    self.__cn0_counter = 0
    self.__cn0_buffer_len = config.cn0_buffer_len
    self.__cn0_ip_buffer = []
    self.__cn0_qp_buffer = []
    
    # clock model
    if config.clock_type is None:
      self.__clk_q = np.zeros((2,2), dtype=np.double)
    else:
      clock_config = get_clock_allan_variance_values(config.clock_type)
      Sb = clock_config.h0 / 2
      Sd = clock_config.h2 * 2 * np.pi**2
      self.__clk_q = 1.1 * SPEED_OF_LIGHT**2 * np.array(
                                                  [[Sb*self.T + Sd/3*self.T**3, Sd/2*self.T**2],
                                                   [            Sd/2*self.T**2,      Sd*self.T]], \
                                                dtype=np.double)
    self.__clk_f = np.array(
                      [[1.0, self.T], \
                       [0.0,    1.0]], \
                    dtype=np.double)
      
    # imu model
    if config.imu_model is None:
      self.__Srg, self.__Sra, self.__Sbad, self.__Sbgd = np.zeros(4, dtype=np.double)
    else:
      imu_model = get_imu_allan_variance_values(config.imu_model)
      self.__Srg = (1.1*imu_model.B_gyr)**2
      self.__Sra = (1.1*imu_model.B_acc)**2
      self.__Sbad = (1.1*imu_model.N_acc)**2
      self.__Sbgd = (1.1*imu_model.N_gyr)**2
    
    # initialize user state
    self.__ecef0 = config.pos.copy()
    self.__lla0 = ecef2lla(config.pos)
    self.__r_eb_e = config.pos.copy()
    self.__v_eb_e = config.vel.copy()
    self.__acc_err = np.zeros(3)
    self.__gyr_err = np.zeros(3)
    self.__clk_bias = config.clock_bias
    self.__clk_drift = config.clock_drift
    
    # generate body to nav rotation
    # sr, sp, sy = np.sin(config.att/R2D)
    # cr, cp, cy = np.cos(config.att/R2D)
    # C_b_n = np.array([[cr*cy - sr*sp*sy, cr*sy + sr*sp*cy, -sr*cp],
    #                   [          -cp*sy,            cp*cy,     sp],
    #                   [sr*cy + cr*sp*sy, sr*sy - cr*sp*cy,  cr*cp]], dtype=np.double).T
    # C_b_n = np.array([[sy*cp,  cr*cy + sr*sy*sp, -sr*cy + cr*sy*sp], 
    #                   [cy*cp, -cr*sy + sr*cy*sp,  sr*sy + cr*cy*sp], 
    #                   [   sp,            -sr*cp,            -cr*cp]], dtype=np.double).T
    C_b_n = euler2dcm(config.att/R2D).T
    self.__C_e_ned = ecef2nedDcm(self.__lla0)
    self.__C_e_n = ecef2enuDcm(self.__lla0)
    self.__C_b_e = self.__C_e_ned.T @ C_b_n
    # self.__C_b_e = self.__C_e_n.T @ C_b_n
    self.__q_b_e = dcm2quat(self.__C_b_e)
    
    # kalman state and covariance
    self.rx_state = np.zeros(17, dtype=np.double)
    self.rx_state[-2] = config.clock_bias
    self.rx_state[-1] = config.clock_drift
    self.rx_cov = np.diag(np.array([0.03,0.03,0.03,0.05,0.06,0.05,2.0,2.0,2.0,0.01,0.01,0.01,1e-4,1e-4,1e-4,2.0,0.1], dtype=np.double)**2)
    self.__is_cov_initialized = False


#--------------------------------------------------------------------------------------------------#
  #! === Kalman Filter Time Update (prediction) ===
  def time_update(self, w_ib_b: np.ndarray, f_ib_b: np.ndarray):
    # mechanize imu
    wb = w_ib_b - self.__gyr_err
    fb = f_ib_b - self.__acc_err
    self.mechanize(wb, fb)
    
    # generate transition matrices
    self.__generate_F(fb)
    self.__generate_Q()
    
    # time update (assumed constant T)
    self.__clk_bias += self.__clk_drift * self.T
    self.rx_state[-2] = self.__clk_bias
    half_Q = self.__Q/2
    self.rx_cov = self.__F @ (self.rx_cov + half_Q) @ self.__F.T + half_Q
    
  #! === Kalman Filter Measurement Update (correction) ===
  def measurement_update(self, meas_prange: np.ndarray, meas_prange_rate: np.ndarray):
    # generate observation and covariance matrix
    self.__estimate_cn0()
    self.__generate_H()
    self.__generate_R()
    
    # compute initial covariance if necessary
    if not self.__is_cov_initialized:
      self.__initialize_covariance()
      self.__is_cov_initialized = True
      
    # measurement residuals (half chip spacing to minimize correlation between early and late)
    # chip_error = dll_error(
    #     self.__correlators.IE, self.__correlators.QE, self.__correlators.IL, self.__correlators.QL, self.__tap_spacing
    #   ) * self.__chip_width
    # freq_error = fll_error(
    #     self.__correlators.ip1, self.__correlators.ip2, self.__correlators.qp1, self.__correlators.qp2, 3*self.T
    #   ) * -self.__wavelength
    # dy = np.concatenate((chip_error, freq_error))
    dy = np.concatenate((meas_prange - self.__pred_prange, 
                         meas_prange_rate - self.__pred_prange_rate))
    
    # innovation filtering
    S = self.__H @ self.rx_cov @ self.__H.T + self.__R
    norm_z = np.abs(dy / np.sqrt(np.diag(S))) # norm_z = np.abs(dy / np.diag(cholesky(S)))
    pass_idx = np.nonzero(norm_z < self.__innovation_std)[0]
    self.__H = self.__H[pass_idx,:]
    self.__R = np.diag(self.__R[pass_idx,pass_idx])
    dy = dy[pass_idx]
    
    # update
    PHt = self.rx_cov @ self.__H.T
    K = PHt @ inv(self.__H @ PHt + self.__R)
    I_KH = np.eye(self.rx_state.size) - K @ self.__H
    self.rx_cov = (I_KH @ self.rx_cov @ I_KH.T) + (K @ self.__R @ K.T)
    self.rx_state += K @ dy
    
    # closed loop correction (error state)
    p0, p1, p2, p3 = 1, *(-self.rx_state[:3]/2)
    q0, q1, q2, q3 = self.__q_b_e
    self.__q_b_e = np.array(
                      [(p0*q0 - p1*q1 - p2*q2 - p3*q3), 
                       (p0*q1 + p1*q0 + p2*q3 - p3*q2), 
                       (p0*q2 - p1*q3 + p2*q0 + p3*q1), 
                       (p0*q3 + p1*q2 - p2*q1 + p3*q0)]
                    )
    self.__C_b_e = quat2dcm(self.__q_b_e)
    # self.__C_b_e = (I3 - skew(self.rx_state[:3])) @ self.__C_b_e
    self.__v_eb_e -= self.rx_state[3:6]
    self.__r_eb_e -= self.rx_state[6:9]
    self.__acc_err += self.rx_state[9:12]
    self.__gyr_err += self.rx_state[12:15]
    self.__clk_bias = self.rx_state[-2]
    self.__clk_drift = self.rx_state[-1]
    self.rx_state[:15] = np.zeros(15)

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
      q_new_old = np.array([np.cos(mag_a_ib_b/2), *np.sin(mag_a_ib_b/2)/mag_a_ib_b*a_ib_b], dtype=np.double)
    else:
      q_new_old = np.array([1,0,0,0], dtype=np.double)
    
    p0, p1, p2, p3 = self.__q_b_e
    q0, q1, q2, q3 = q_new_old
    a1, a2, a3 = OMEGA_IE * self.T / 2
    
    # (Groves E.40) quaternion multiplication attitude update
    self.__q_b_e = np.array([(p0*q0 - p1*q1 - p2*q2 - p3*q3) - (-a1*p1 - a2*p2 - a3*p3), 
                             (p0*q1 + p1*q0 + p2*q3 - p3*q2) - ( a1*p0 + a2*p3 - a3*p2), 
                             (p0*q2 - p1*q3 + p2*q0 + p3*q1) - (-a1*p3 + a2*p0 + a3*p1), 
                             (p0*q3 + p1*q2 - p2*q1 + p3*q0) - ( a1*p2 - a2*p1 + a3*p0)])
    
    # (Groves E.43) quaternion normalization
    self.__q_b_e /= norm(self.__q_b_e)
    
    # convert to DCM and euler angles
    self.__C_b_e = quat2dcm(self.__q_b_e)
    C_avg = quat2dcm((q_old + self.__q_b_e)/2)
    
    # # (Groves 2.145) determine earth rotation over update interval
    # alpha_ie = GNSS_OMEGA_EARTH * self.T
    # sin_aie = np.sin(alpha_ie)
    # cos_aie = np.cos(alpha_ie)
    # C_earth = np.array([[ cos_aie, sin_aie, 0.0], \
    #                     [-sin_aie, cos_aie, 0.0], \
    #                     [     0.0,     0.0, 1.0]])
    
    # # attitude increment
    # alpha = w_ib_b * self.T
    # alpha_n = np.linalg.norm(alpha) # norm of alpha
    # Alpha = skew(alpha)             # skew symmetric form of alpha
    
    # # (Groves 5.73) obtain dcm from new->old attitude
    # sina = np.sin(alpha_n)
    # cosa = np.cos(alpha_n)
    # if alpha_n > 1.0e-8:
    #   C_new_old = I3 + (sina / alpha_n * Alpha) + \
    #                     ((1 - cosa) / alpha_n**2 * Alpha @ Alpha)
    # else:
    #   C_new_old = I3 + Alpha
    
    # # (Groves 5.75) attitude update
    # C_new = C_earth @ self.__C_b_e @ C_new_old

    # # (Groves 5.84/5.85) calculate average body-to-ECEF dcm
    # Alpha_ie = skew(np.array([0.0, 0.0, alpha_ie]))
    # if alpha_n > 1.0e-8:
    #   C_avg = self.__C_b_e @ (I3 + ((1 - cosa) / alpha_n**2 \
    #                        * Alpha) + ((1 - sina / alpha_n) / alpha_n**2 \
    #                        * Alpha @ Alpha)) - (0.5 * Alpha_ie @ self.__C_b_e)
    # else:
    #   C_avg = self.__C_b_e - 0.5 * Alpha_ie @ self.__C_b_e
    # self.__C_b_e = C_new
    
    # (Groves 5.85) specific force transformation body-to-ECEF
    f_ib_e = C_avg@f_ib_b

    # (Groves 5.36) velocity update
    gravity,_ = ned2ecefg(self.__r_eb_e)
    # gravity,_ = ecefg(self.__r_eb_e)
    self.__v_eb_e = v_old + self.T*(f_ib_e + gravity - 2*OMEGA_IE_E@v_old)

    # (Groves 5.38) position update
    self.__r_eb_e = p_old + self.T*(self.__v_eb_e + v_old)/2
    
  #! === Predict GNSS observables (pseudorange and pseudorange-rate) ===
  def predict_observables(self, emitter_states: dict):
    n = len(emitter_states)
    self.__range_unit_vectors = np.zeros((n,3), dtype=np.double)
    self.__rate_unit_vectors = np.zeros((n,3), dtype=np.double)
    self.__pred_prange = np.zeros(n, dtype=np.double)
    self.__pred_prange_rate = np.zeros(n, dtype=np.double)
    self.__range_avbl = np.zeros(n, dtype=bool)
    self.__rate_avbl = np.zeros(n, dtype=bool)

    for j,emitter in enumerate(emitter_states.values()):
      dr = emitter.pos - self.__r_eb_e
      dv = emitter.vel - self.__v_eb_e
      
      r = norm(dr)
      ur = dr / r
      v = ur @ dv
      uv = np.cross(ur, np.cross(ur, dv/r))
      
      self.__range_unit_vectors[j,:] = ur
      self.__rate_unit_vectors[j,:] = uv
      self.__pred_prange[j] = r
      self.__pred_prange_rate[j] = v
      # self.__range_avbl[j] = 
      # self.__rate_avbl[j] = 
      
    self.__pred_prange += self.__clk_bias         #* add clock bias
    self.__pred_prange_rate += self.__clk_drift   #* add clock drift
    return self.__pred_prange, self.__pred_prange_rate

  #! === Update Correlators ===
  def update_correlators(self, correlators: Correlators):
    self.__correlators = correlators


#--------------------------------------------------------------------------------------------------#
  #! === State Tranisition Matrix ===
  def __generate_F(self, f_ib_b: np.ndarray):
    # radii of curvature and gravity
    lla = ecef2lla(self.__r_eb_e)
    r_es_e = geocentricRadius(lla[0])
    _,gamma = ned2ecefg(self.__r_eb_e)
    # _,gamma = ecefg(self.__r_eb_e)
  
    # (Groves 14.18/87) state transition matrix discretization
    self.__f21 = -skew(self.__C_b_e @ f_ib_b)
    OC = OMEGA_IE_E@self.__C_b_e
    OF = OMEGA_IE_E@self.__f21
    FO = self.__f21@OMEGA_IE_E
    FC = self.__f21@self.__C_b_e
    
    # (Groves Appendix I)
    F11 = I3 - OMEGA_IE_E*self.T
    F15 = self.__C_b_e*self.T - 1/2*OC*self.T**2
    F21 = self.__f21*self.T - 1/2*FO*self.T**2 - OF*self.T**2
    F22 = I3 - 2*OMEGA_IE_E*self.T
    F23 = np.outer(-(2 * gamma / r_es_e), (self.__r_eb_e / norm(self.__r_eb_e))) * self.T
    F24 = self.__C_b_e*self.T - OC*self.T**2
    F25 = 1/2*FC*self.T**2 - 1/6*FO@self.__C_b_e*self.T**3 - 1/3*OF@self.__C_b_e*self.T**3
    F31 = 1/2*self.__f21*self.T**2 - 1/6*FO*self.T**3 - 1/3*OF*self.T**3
    F32 = I3*self.T - OMEGA_IE_E*self.T**2
    F34 = 1/2*self.__C_b_e*self.T**2 - 1/3*OC*self.T**3
    F35 = 1/6*FC*self.T**3
    
    self.__F = np.block(
                [[F11, Z33, Z33, Z33, F15, Z32],
                 [F21, F22, F23, F24, F25, Z32], 
                 [F31, F32,  I3, F34, F35, Z32], 
                 [Z33, Z33, Z33,  I3, Z33, Z32], 
                 [Z33, Z33, Z33, Z33,  I3, Z32], 
                 [Z23, Z23, Z23, Z23, Z23, self.__clk_f]]
              )
    
  #! === Process Noise Covariance ===
  def __generate_Q(self):
    FFt = self.__f21@self.__f21.T
    FC = self.__f21@self.__C_b_e
    
    Q11 = (self.__Srg*self.T + 1/3*self.__Sbgd*self.T**3) * I3
    Q21 = (1/2*self.__Srg*self.T**2 + 1/4*self.__Sbgd*self.T**4) * self.__f21
    Q22 = (self.__Sra*self.T + 1/3*self.__Sbad*self.T**3) * I3 + (1/3*self.__Srg*self.T**3 + 1/5*self.__Sbgd*self.T**5) * (FFt)
    Q31 = (1/3*self.__Srg*self.T**3 + 1/5*self.__Sbgd*self.T**5) * self.__f21
    Q32 = (1/2*self.__Sra*self.T**2 + 1/4*self.__Sbad*self.T**4) * I3 + (1/4*self.__Srg*self.T**4 + 1/6*self.__Sbgd*self.T**6) * (FFt)
    Q33 = (1/3*self.__Sra*self.T**3 + 1/5*self.__Sbad*self.T**5) * I3 + (1/5*self.__Srg*self.T**5 + 1/7*self.__Sbgd*self.T**7) * (FFt)
    Q34 = 1/3*self.__Sbad*self.T**3*self.__C_b_e
    Q35 = 1/4*self.__Sbgd*self.T**4*FC
    Q15 = 1/2*self.__Sbgd*self.T**2*self.__C_b_e
    Q24 = 1/2*self.__Sbad*self.T**2*self.__C_b_e
    Q25 = 1/3*self.__Sbgd*self.T**3*FC
    Q44 = self.__Sbad*self.T*I3
    Q55 = self.__Sbgd*self.T*I3
    
    self.__Q = np.block(
                [[Q11, Q21.T, Q31.T, Z33, Q15, Z32], \
                 [Q21,   Q22, Q32.T, Q24, Q25, Z32], \
                 [Q31,   Q32,   Q33, Q34, Q35, Z32], \
                 [Z33,   Q24, Q34.T, Q44, Z33, Z32], \
                 [Q15, Q25.T, Q35.T, Z33, Q55, Z32], \
                 [Z23,   Z23,   Z23, Z23, Z23, self.__clk_q]], \
              )
    
  #! === Observation Matrix ===
  def __generate_H(self):
    Z = np.zeros(self.__range_unit_vectors.shape, dtype=np.double)
    z1 = np.zeros(self.__range_unit_vectors.shape[0], dtype=np.double)
    i1 = np.ones(self.__range_unit_vectors.shape[0], dtype=np.double)
    self.__H = np.block(
                [[Z,                         Z, self.__range_unit_vectors, Z, Z, i1[:,None], z1[:,None]], 
                 [Z, self.__range_unit_vectors,  self.__rate_unit_vectors, Z, Z, z1[:,None], i1[:,None]]]
              )
    # self.__H = np.block(
    #             [[Z,                         Z, self.__range_unit_vectors, Z, Z, i1[:,None], z1[:,None]], 
    #              [Z, self.__range_unit_vectors,                         Z, Z, Z, z1[:,None], i1[:,None]]]
    #           )
    # self.__H = np.block(
    #             [[Z,                         Z,                         Z, Z, Z, z1[:,None], z1[:,None]], 
    #              [Z, self.__range_unit_vectors,  self.__rate_unit_vectors, Z, Z, z1[:,None], i1[:,None]]]
    #           )
    
  #! === Measurement Noise Covariance ===
  def __generate_R(self):
    self.__R = np.diag( 
                np.concatenate((prange_residual_var(self.cn0, 3*self.T, self.__chip_width), 
                                prange_rate_residual_var(self.cn0, 3*self.T, self.__wavelength))) 
              )
    

#--------------------------------------------------------------------------------------------------#
  #! === Estimate CN0 ===
  def __estimate_cn0(self):
    self.__cn0_ip_buffer.append(self.__correlators.IP)
    self.__cn0_qp_buffer.append(self.__correlators.QP)
    self.__cn0_counter += 1
    if self.__cn0_counter == self.__cn0_buffer_len:
      ip_tmp = np.array(self.__cn0_ip_buffer, dtype=np.double)
      qp_tmp = np.array(self.__cn0_qp_buffer, dtype=np.double)
      for i in range(self.cn0.size):
        self.cn0[i] = CN0_m2m4_estimator(
            ip_tmp[:,i], 
            qp_tmp[:,i], 
            self.cn0[i], 
            3*self.T
          )
        # self.cn0[i] = CN0_beaulieu_estimator(
        #     ip_tmp[:,i], 
        #     qp_tmp[:,i], 
        #     self.cn0[i], 
        #     3*self.T,
        #     False,
        #   )
      self.__cn0_counter = 0
      self.__cn0_ip_buffer = []
      self.__cn0_qp_buffer = []

  #! === Initialize Receiver Covariance Matrix ===
  def __initialize_covariance(self):
    I = np.eye(self.__F.shape[0])
    delta_diag_P = np.diag(self.rx_cov)

    while np.any(delta_diag_P > 1e-4):
      previous_P = self.rx_cov

      self.rx_cov = self.__F @ self.rx_cov @ self.__F.T + self.__Q
      K = self.rx_cov @ self.__H.T @ inv(self.__H @ self.rx_cov @ self.__H.T + self.__R)
      self.rx_cov = (I - K @ self.__H) @ self.rx_cov @ (I - K @ self.__H).T + K @ self.__R @ K.T

      delta_diag_P = np.diag(previous_P - self.rx_cov)
      
    self.rx_cov *= 10
    