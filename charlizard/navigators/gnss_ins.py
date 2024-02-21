'''
|========================================== soop_ins.py ===========================================|
|                                                                                                  |
|   Property of Daniel Sturdivant. Unauthorized copying of this file via any medium would be       |
|   super sad and unfortunate for me. Proprietary and confidential.                                |
|                                                                                                  |
| ------------------------------------------------------------------------------------------------ |
|                                                                                                  |
|   @file     charlizard/navigators/soop_ins.py                                                    |
|   @brief    Deeply coupled GNSS-INS (optional pseudorange-rates only).                           |
|               - Results in the local tangent frame (ENU) for comparisons!                        |
|   @author   Daniel Sturdivant <sturdivant20@gmail.com>                                           |
|   @date     February 2024                                                                        |
|                                                                                                  |
|==================================================================================================|
'''

import numpy as np
from scipy.linalg import solve_discrete_are, solve_continuous_are, inv

from charlizard.models.discriminator import prange_rate_residual_var, prange_residual_var, fll_error, dll_error
from charlizard.models.lock_detectors import CN0_m2m4_estimator, CN0_beaulieu_estimator
from charlizard.models.correlators import Correlators
from charlizard.navigators.structures import GNSSINSConfig
from navsim.error_models.clock import NavigationClock, get_clock_allan_variance_values

from navtools.conversions.coordinates import ecef2enuDcm, ecef2lla
from navtools.conversions.attitude import euler2dcm, dcm2quat, quat2dcm, dcm2euler
from navtools.conversions.skew import skew, deskew
from navtools.measurements.gravity import ned2ecefg
from navtools.measurements.radii_of_curvature import geocentricRadius
from navtools.constants import GNSS_OMEGA_EARTH, SPEED_OF_LIGHT

R2D = 180 / np.pi
D2R = np.pi / 180
LLA_R2D = np.array([R2D, R2D, 1])
LLA_D2R = np.array([D2R, D2R, 1])
I3 = np.eye(3)
Z33 = np.zeros((3,3))
Z32 = np.zeros((3,2))
Z23 = np.zeros((2,3))
w_ie = np.array([0.0, 0.0, GNSS_OMEGA_EARTH])
OMEGA_IE_E = skew(w_ie)


class GnssIns:
  @property
  def extract_states(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # update ENU state
    self.att = dcm2euler((self._C_e_n @ self._C).T) * R2D
    self.vel = self._C_n_e @ self._v_eb_e
    self.pos = self._C_n_e @ (self._r_eb_e - self._ecef0)
    self.lla = ecef2lla(self._r_eb_e) * LLA_R2D
    self.clk = self.rx_state[-2:]
    return self.pos, self.vel, self.att, self.lla, self.clk
  
  @property
  def extract_stds(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # extract ENU standard deviations
    att = np.sqrt(np.diag(self._C_e_n @ self.rx_cov[0:3,0:3] @ self._C_n_e)) * R2D
    vel = np.sqrt(np.diag(self._C_e_n @ self.rx_cov[3:6,3:6] @ self._C_n_e))
    pos = np.sqrt(np.diag(self._C_e_n @ self.rx_cov[6:9,6:9] @ self._C_n_e))
    clk = np.sqrt(np.diag(self.rx_cov[-2:,-2:]))
    return pos, vel, att, clk
  
  @property
  def extract_dops(self) -> tuple[float, float, float, float, float, int]:
    n = self._range_unit_vectors.shape[0]
    i1 = np.ones(n)
    H = np.block([self._range_unit_vectors, i1[:,None]])
    dop = inv(H.T @ H)
    dop[:3,:3] = self._C_e_n @ dop[:3,:3] @ self._C_n_e
    gdop = np.sqrt(dop.trace())
    pdop = np.sqrt(dop[:3,:3].trace())
    hdop = np.sqrt(dop[:3,:3].trace())
    vdop = np.sqrt(dop[2,2].trace())
    tdop = np.sqrt(dop[2,2].trace())
    return gdop, pdop, hdop, vdop, tdop, n
  
  def __init__(self, config: GNSSINSConfig):
    self.T = config.T
    self.cn0 = config.cn0
    
    self._Srg = config.Srg
    self._Sra = config.Sra
    self._Sbad = config.Sbad
    self._Sbgd = config.Sbgd
    self._innovation_stdev = config.innovation_stdev
    if config.clock_type is None:
      clock_config = NavigationClock(h0=0.0, h1=0.0, h2=0.0)
    else:
      clock_config = get_clock_allan_variance_values(config.clock_type)
    self._Sb = clock_config.h0 / 2
    self._Sd = clock_config.h2 * 2 * np.pi**2
    
    # signal settings
    # sig_config = config.signal_config.get('gps'.casefold())
    # self._chip_width = SPEED_OF_LIGHT / sig_config.fchip_data
    # self._wavelength = SPEED_OF_LIGHT / sig_config.fcarrier
    self._chip_width = SPEED_OF_LIGHT / 1.023e6
    self._wavelength = SPEED_OF_LIGHT / 1575.42e6
    
    # correlators
    self._correlators = Correlators(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    # cn0 estimation config
    self.cn0 = config.cn0
    self._cn0_counter = 0
    self._cn0_buffer_len = config.cn0_buffer_len
    self._cn0_ip_buffer = []
    self._cn0_qp_buffer = []
      
    # basis PVA
    self._lla0 = ecef2lla(config.pos)
    self._ecef0 = config.pos
    self._C_n_e = ecef2enuDcm(self._lla0)
    self._C_e_n = self._C_n_e.T
    
    # initialize ENU state (comparison state)
    self.lla = self._lla0 * LLA_R2D                           #* LLA position [deg, deg, m]
    self.pos = self._C_n_e @ (config.pos - self._ecef0)       #* ENU position
    self.vel = self._C_n_e @ config.vel                       #* ENU velocity
    self.att = config.att                                     #* roll, pitch, yaw [deg]
    self.clk = np.array(config.clock_bias, config.clock_drift)#* clock bias and drift
    
    # initialize ECEF state (mechanized rx state)
    self._C = self._C_n_e @ euler2dcm(D2R * config.att).T   #* attitude dcm
    self._q = dcm2quat(self._C)                             #* attitude quaternion
    self._r_eb_e = config.pos                               #* ECEF position
    self._v_eb_e = config.vel                               #* ECEF velocity
    
    # initialize state and covariance
    self.rx_state = np.zeros(17)
    self.rx_cov = np.eye(17)
    self.rx_state[-2:] = self.clk
    self._is_cov_initialized = False
    
    
#--------------------------------------------------------------------------------------------------#
  #! === Kalman Filter Time Update (prediction) ===
  def time_update(self, w_ib_b: np.ndarray, f_ib_b: np.ndarray):
    # mechanize imu
    self._gravity, self._gamma_ib_e = ned2ecefg(self._r_eb_e)
    self.mechanize(w_ib_b, f_ib_b)
    
    # updated gravity and geocentric radius
    self.lla = ecef2lla(self._r_eb_e)
    self._gravity, self._gamma_ib_e = ned2ecefg(self._r_eb_e)
    self._r_eS_e = geocentricRadius(self.lla[0]*D2R)
    
    # generate transition matrices
    self.__generate_A(f_ib_b)
    self.__generate_Q()
    
    # time update (assumed constant T)
    self.rx_state[-2] += self.rx_state[-1] * self.T           #* only need to predict clock states
    self.rx_cov = self._F @ self.rx_cov @ self._F.T + self._Q
    # self.clk = self.rx_state[-2:]
    
  #! === Kalman Filter Measurement Update (correction) ===
  def measurement_update(self, meas_prange: np.ndarray, meas_prange_rate: np.ndarray):
    # generate observation and covariance matrix
    self.__estimate_cn0()
    self.__generate_C()
    self.__generate_R()
    
    # compute initial covariance if necessary
    if not self._is_cov_initialized:
      self.__initialize_covariance()
      self._is_cov_initialized = True
      
    # measurement residuals (half chip spacing to minimize correlation between early and late)
    chip_error = dll_error(
        self._correlators.IE, self._correlators.QE, self._correlators.IL, self._correlators.QL, self._tap_spacing
      ) * self._chip_width
    freq_error = fll_error(
        self._correlators.ip1, self._correlators.ip2, self._correlators.qp1, self._correlators.qp2, self.T
      ) * -self._wavelength
    dy = np.concatenate((chip_error + meas_prange - self._pred_prange, 
                         freq_error + meas_prange_rate - self._pred_prange_rate))
    # dy = np.concatenate((chip_error, freq_error))
    
    # innovation filtering
    S = self._H @ self.rx_cov @ self._H.T + self._R
    norm_z = np.abs(dy / np.sqrt(np.diag(S))) # norm_z = np.abs(dy / np.diag(np.linalg.cholesky(S)))
    fault_idx = np.nonzero(norm_z < self._innovation_stdev)[0]
    self._H = self._H[fault_idx,:]
    self._R = np.diag(self._R[fault_idx,fault_idx])
    dy = dy[fault_idx]
    
    # update
    PHt = self.rx_cov @ self._H.T
    K = PHt @ inv(self._H @ PHt + self._R)
    I_KH = np.eye(self.rx_state.size) - K @ self._H
    self.rx_cov = (I_KH @ self.rx_cov @ I_KH.T) + (K @ self._R @ K.T)
    self.rx_state += K @ dy
    
    # closed loop correction (error state)
    p0, p1, p2, p3 = 1, *(-self.rx_state[:3]/2)
    q0, q1, q2, q3 = self._q
    self._q = np.array([(p0*q0 - p1*q1 - p2*q2 - p3*q3), 
                        (p0*q1 + p1*q0 + p2*q3 - p3*q2), 
                        (p0*q2 - p1*q3 + p2*q0 + p3*q1), 
                        (p0*q3 + p1*q2 - p2*q1 + p3*q0)])
    self._C = quat2dcm(self._q)
    self._v_eb_e -= self.rx_state[3:6]
    self._r_eb_e -= self.rx_state[6:9]
    ab_drift = ab_drift + self.rx_state[9:12]
    gb_drift = gb_drift + self.rx_state[12:15]
  
  #! ===IMU Mechanization (ECEF Frame) ===
  def mechanize(self, w_ib_b: np.ndarray, f_ib_b: np.ndarray):
    # copy old values
    q_old = self._q.copy()
    v_old = self._v_eb_e.copy()
    p_old = self._r_eb_e.copy()
    
    # rotational phase increment
    a_ib_b = w_ib_b * self.T
    mag_a_ib_b = np.linalg.norm(a_ib_b)
    
    # (Groves E.39) precision quaternion from old to new attitude
    q_new_old = np.array([np.cos(mag_a_ib_b/2), np.sin(mag_a_ib_b/2)/mag_a_ib_b*a_ib_b])
    
    p0, p1, p2, p3 = self._q
    q0, q1, q2, q3 = q_new_old
    a1, a2, a3 = w_ie * self.T / 2
    
    # (Groves E.40) quaternion multiplication attitude update
    self._q = np.array([(p0*q0 - p1*q1 - p2*q2 - p3*q3) - (-a1*q1 - a2*q2 - a3*q3), 
                        (p0*q1 + p1*q0 + p2*q3 - p3*q2) - ( a1*q0 + a2*q3 - a3*q2), 
                        (p0*q2 - p1*q3 + p2*q0 + p3*q1) - (-a1*q3 + a2*q0 + a3*q1), 
                        (p0*q3 + p1*q2 - p2*q1 + p3*q0) - ( a1*q2 - a2*q1 + a3*q0)])
    
    # (Groves E.43) quaternion normalization
    self._q /= np.linalg.norm(self._q)
    
    # convert to DCM and euler angles
    self._C = quat2dcm(self._q)
    # C_avg = quat2dcm((q_old + self._q)/2)
    C_avg = quat2dcm(q_old + self._q)
    
    # (Groves 5.85) specific force transformation body-to-ECEF
    f_ib_e = C_avg@f_ib_b

    # (Groves 5.36) velocity update
    self._v_eb_e = v_old + self.T*(f_ib_e + self._gravity - 2*OMEGA_IE_E@self._v_eb_e)

    # (Groves 5.38) position update
    self._r_eb_e = p_old + self.T*(self._v_eb_e + v_old)/2
    
  #! === Predict GNSS observables (pseudorange and pseudorange-rate) ===
  def predict_observables(self, emitter_states: dict):
    URs = []
    UVs = []
    ranges = []
    range_rates = []

    for emitter in emitter_states.values():
      dr = emitter.pos - self._r_eb_e
      dv = emitter.vel - self._v_eb_e
      
      r = np.linalg.norm(dr)
      ur = dr / r
      v = ur @ dv
      uv = np.cross(ur, np.cross(ur, dv/r))
      
      URs.append(ur)
      UVs.append(uv)
      ranges.append(r)
      range_rates.append(v)
      
    self._pred_prange = np.array(ranges) + self.rx_state[-2]            #* add clock bias
    self._pred_prange_rate = np.array(range_rates) + self.rx_state[-1]  #* add clock drift
    self._range_unit_vectors = np.array(URs)
    self._rate_unit_vectors = np.array(UVs)


#--------------------------------------------------------------------------------------------------#
  #! === State Tranition Matrix ===
  def __generate_A(self, f_ib_b: np.ndarray):
    # (Groves 14.18/87) state transition matrix discretization
    self._f21 = -skew(self._C @ f_ib_b)
    self._f23 = np.outer(-(2*self._gamma_ib_e/self._r_es_e), (self._r_eb_e/np.linalg.norm(self._r_eb_e)))
    
    # temp variables for reducing math
    oc = OMEGA_IE_E@self._C
    of = OMEGA_IE_E@self._f21
    fc = self._f21@self._C
    foc = self._f21@oc
    
    # (Groves appendix I) third order expansion
    A11 = I3 - OMEGA_IE_E*self.T
    A15 = self._C*self.T - (oc*self.T**2)/2
    A21 = self._f21*self.T - (foc*self.T**2)/2 - of*self.T**2
    A22 = I3 - 2*OMEGA_IE_E*self.T
    A23 = self._f23*self.T
    A24 = self._C*self.T - oc*self.T**2
    A25 = (fc*self.T**2)/2 - (foc*self.T**3)/6 - (of@self._C*self.T**3)/3
    A31 = (self._f21*self.T**2)/2 - (self._f21@OMEGA_IE_E*self.T**3)/6 - (of*self.T)/3
    A32 = I3*self.T - OMEGA_IE_E*self.T**2
    A34 = (self._C*self.T**2)/2 - (oc*self.T**3)/3
    A35 = (fc*self.T**3)/6
    A66 = np.array(
            [[1.0,  self.T], \
             [0.0,    1.0]]
          )
    
    self._F = np.block(
                [[A11, Z33, Z33, Z33, A15, Z32],
                 [A21, A22, A23, A24, A25, Z32], 
                 [A31, A32,  I3, A34, A35, Z32], 
                 [Z33, Z33, Z33,  I3, Z33, Z32], 
                 [Z33, Z33, Z33, Z33,  I3, Z32], 
                 [Z23, Z23, Z23, Z23, Z23, A66]]
              )

  #! === Process Covariance Matrix ===
  def __generate_Q(self):
    # temp variables for reducing math
    f21_f21t = self._f21 @ self._f21.T
    fc = self._f21 @ self._C
    
    # (Groves 14.80/88) process noise covariance
    Q11 = (self._Srg*self.T + (self._Sbgd*self.T**3)/3)*I3
    Q21 = ((self._Srg*self.T**2)/2 + (self._Sbgd*self.T**4)/4)*self._f21
    Q22 = (self._Sra*self.T + (self._Sbad*self.T**3)/3)*I3 + ((self._Srg*self.T**3)/3 + (self._Sbgd*self.T**5)/5)*f21_f21t
    Q31 = ((self._Srg*self.T**3)/3 + (self._Sbgd*self.T**5)/5)*self._f21
    Q32 = ((self._Sra*self.T**2)/2 + (self._Sbad*self.T**4)/4)*I3 + ((self._Srg*self.T**4)/4 + (self._Sbgd*self.T**6)/6)*f21_f21t
    Q33 = ((self._Sra*self.T**3)/3 + (self._Sbad*self.T**5)/5)*I3 + ((self._Srg*self.T**5)/5 + (self._Sbgd*self.T**7)/7)*f21_f21t
    Q34 = (self._Sbad*self.T**3)/3*self._C
    Q35 = (self._Sbgd*self.T**4)/4*fc
    Q15 = (self._Sbgd*self.T**2)/2*self._C
    Q24 = (self._Sbad*self.T**2)/2*self._C
    Q25 = (self._Sbgd*self.T**3)/3*fc
    Q44 = self._Sbad*self.T*I3
    Q55 = self._Sbgd*self.T*I3
    Q66 = SPEED_OF_LIGHT * np.array(
            [[self._Sb*self.T + 1/3*self._Sd*self.T**3, 1/2*self._Sd*self.T**2], \
             [                  1/2*self._Sd*self.T**2,        self._Sd*self.T]]
          )
          
    self._Q = np.block(
                [[  Q11, Q21.T, Q31.T, Z33, Q15, Z32], 
                 [  Q21,   Q22, Q32.T, Q24, Q25, Z32], 
                 [  Q31,   Q32,   Q33, Q34, Q35, Z32], 
                 [  Z33, Q24.T, Q32.T, Q44, Z33, Z32], 
                 [Q15.T, Q25.T, Q35.T, Z33, Q55, Z32], 
                 [  Z23,   Z23,   Z23, Z23, Z23, Q66]]
              )
    
  #! === Observation Matrix ===
  def __generate_C(self):
    Z = np.zeros(self._range_unit_vectors.shape)
    z1 = np.zeros(self._range_unit_vectors.shape[0])
    i1 = np.ones(self._range_unit_vectors.shape[0])
    self._H = np.block(
                [[Z,                        Z, self._range_unit_vectors, Z, Z, i1[:,None], z1[:,None]], 
                 [Z, self._range_unit_vectors,  self._rate_unit_vectors, Z, Z, z1[:,None], i1[:,None]]]
              )
    
  #! === Measurement Noise Covariance ===
  def __generate_R(self):
    self._R = np.diag( 
                np.concatenate((prange_residual_var(self.cn0, self.T, self._chip_width), 
                                prange_rate_residual_var(self.cn0, self.T, self._wavelength))) 
              )


#--------------------------------------------------------------------------------------------------#
  #! === Update Correlators ===
  def update_correlators(self, correlators: Correlators):
    self._correlators = correlators

  #! === Estimate CN0 ===
  def __estimate_cn0(self):
    self._cn0_ip_buffer.append(self._correlators.IP)
    self._cn0_qp_buffer.append(self._correlators.QP)
    self._cn0_counter += 1
    if self._cn0_counter == self._cn0_buffer_len:
      ip_tmp = np.array(self._cn0_ip_buffer)
      qp_tmp = np.array(self._cn0_qp_buffer)
      for i in range(self.cn0.size):
        self.cn0[i] = CN0_m2m4_estimator(
            ip_tmp[:,i], 
            qp_tmp[:,i], 
            self.cn0[i], 
            self.T
          )
        # self.cn0[i] = CN0_beaulieu_estimator(
        #     ip_tmp[:,i], 
        #     qp_tmp[:,i], 
        #     self.cn0[i], 
        #     self.T,
        #     False,
        #   )
      self._cn0_counter = 0
      self._cn0_ip_buffer = []
      self._cn0_qp_buffer = []
  
    
#--------------------------------------------------------------------------------------------------#
  #! === Initialize Receiver Covariance Matrix ===
  def __initialize_covariance(self):
    # this allows for quick convergence
    self.rx_cov = 10*solve_discrete_are(self._F.T, self._H.T, self._Q, self._R)
    