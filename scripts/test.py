
import numpy as np
import matplotlib.pyplot as plt
from navtools.constants import SPEED_OF_LIGHT
from charlizard.models.correlators import Correlators, CorrelatorErrors
# from navsim.error_models.clock import get_clock_allan_variance_values

# Z3 = np.zeros((3,3))
# Z32 = np.zeros((3,2))
# Z23 = np.zeros((2,3))
# I3 = np.eye(3)

# # Brown & Hwang eq. 9.3.5 and 9.3.11
# # HIGH_QUALITY_TCXO = NavigationClock(h0=2e-21, h1=1e-22, h2=2e-20)
# h0 = 2e-21
# h1 = 1e-22
# h2 = 2e-20
# wb = np.sqrt(SPEED_OF_LIGHT**2 * h0 / 2)
# wd = np.sqrt(SPEED_OF_LIGHT**2 * h2 * 2 * np.pi**2)

# # system model
# A = np.block(
#               [[Z3,   I3,       Z32], 
#                [Z3,   Z3,       Z32], 
#                [Z23, Z23, I3[1:,:2]]]
#             )
# B = np.block(
#               [[ Z3,       Z32], 
#                [ I3,       Z32], 
#                [Z23, I3[:2,:2]]]
#             )
# w = np.array([0.1,0.1,0.1,wb,wd])

# # simulate system @ 10 Hz for 100s (continuous)
# x = np.zeros((8,10*100))
# x[:,0] = B @ (w*np.random.randn(5))
# T = 0.1
# for i in range(1,10*100):
#   x_dot = A @ x[:,i-1] + B @ (w*np.random.randn(5))
#   x[:,i] = x[:,i-1] + T*x_dot

# # plot simulated error
# plt.plot(x[0,:], label='x')
# plt.plot(x[1,:], label='y')
# plt.plot(x[2,:], label='z')
# plt.plot(x[3,:], label='vx')
# plt.plot(x[4,:], label='vy')
# plt.plot(x[5,:], label='vz')
# plt.plot(x[6,:], label='cb')
# plt.plot(x[7,:], label='cd')
# plt.legend()
# plt.show()

# # print(A)
# # print(B)
# print(x[:,0])
# print(x[:,-1])
# print(x[:,-1]-x[:,0])


def correlator_model(err: CorrelatorErrors, cn0: np.ndarray, tau: float, T: float) -> Correlators:
  # convert out of dB-Hz
  n = cn0.size
  raw_cn0 = 10**(0.1*cn0)
  
  # amplitude
  p = np.pi * err.freq * T
  A = np.sqrt(2*raw_cn0*T) * np.sin(p) / p
  
  # data bit +/- 1
  D = 1 # if np.random.random() < 0.5 else -1
  
  # autocorrelation
  RE = 1 - np.abs(err.chip + tau)
  RP = 1 - np.abs(err.chip)
  RL = 1 - np.abs(err.chip - tau)
  
  # linear sub-phase intervals
  m = 10            # number of phase points
  subphase_time = T * np.arange(m,0,-1) / m
  subphase_offset_linear = np.outer(err.freq, subphase_time)
  subphase_error = err.phase[:,None] - subphase_offset_linear
  
  # subphase carrier replicas
  inphase = np.cos(p[:,None] + subphase_error)
  quadrature = np.sin(p[:,None] + subphase_error)
  
  # subphase correlators
  sub_ie = np.array([A[i]*RE[i]*D*inphase[i,:] + np.random.randn(m) for i in range(n)])
  sub_ip = np.array([A[i]*RP[i]*D*inphase[i,:] + np.random.randn(m) for i in range(n)])
  sub_il = np.array([A[i]*RL[i]*D*inphase[i,:] + np.random.randn(m) for i in range(n)])
  sub_qe = np.array([A[i]*RE[i]*D*quadrature[i,:] + np.random.randn(m) for i in range(n)])
  sub_qp = np.array([A[i]*RP[i]*D*quadrature[i,:] + np.random.randn(m) for i in range(n)])
  sub_ql = np.array([A[i]*RL[i]*D*quadrature[i,:] + np.random.randn(m) for i in range(n)])
  
  # correlators
  ip1 = sub_ip[:,:5].sum(axis=1)
  ip2 = sub_ip[:,5:].sum(axis=1)
  qp1 = sub_qp[:,:5].sum(axis=1)
  qp2 = sub_qp[:,5:].sum(axis=1)
  IE = sub_ie.sum(axis=1)
  IP = ip1 + ip2
  IL = sub_il.sum(axis=1)
  QE = sub_qe.sum(axis=1)
  QP = qp1 + qp2
  QL = sub_ql.sum(axis=1)
  
  return Correlators(IE, IP, IL, QE, QP, QL, ip1, qp1, ip2, qp2)
  
  
err = CorrelatorErrors(np.array([0.0, 0.0]),
                       np.array([0.0, 0.0]),
                       np.array([0.0, 0.0]),
                       np.array([0.01, -0.01]),
                       np.array([0.01, -0.01]),
                       np.array([0.01, -0.01]),
                      )
cn0 = np.array([45,40])
tau = 0.1
T = 0.02

c = correlator_model(err, cn0, tau, T)
print(c)
