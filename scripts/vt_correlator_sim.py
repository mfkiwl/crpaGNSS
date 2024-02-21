'''
|====================================== vt_correlator_sim.py ======================================|
|                                                                                                  |
|   Property of Daniel Sturdivant. Unauthorized copying of this file via any medium would be       |
|   super sad and unfortunate for me. Proprietary and confidential.                                |
|                                                                                                  |
| ------------------------------------------------------------------------------------------------ |
|                                                                                                  |
|   @file     scripts/vt_correlator_sim.py                                                         |
|   @brief    Correlator simulation based on vector tracking filter.                               |
|   @author   Daniel Sturdivant <sturdivant20@gmail.com>                                           |
|   @date     February 2024                                                                        |
|                                                                                                  |
|==================================================================================================|
'''

import numpy as np
from tqdm import tqdm
from pathlib import Path
from log_utils import *
import matplotlib.pyplot as plt

from navtools.constants import SPEED_OF_LIGHT
from navtools.conversions.coordinates import ecef2lla, ecef2enu, ecef2enuv, ecef2enuDcm
from navsim.configuration import get_configuration
from navsim.common import get_signal_simulation

from charlizard.models.correlators import correlator_error, correlator_model
from charlizard.navigators.structures import VDFLLConfig
from charlizard.navigators.vt import VectorTrack

R2D, D2R = 180 / np.pi, np.pi / 180
LLA_R2D = np.array([R2D, R2D, 1.0], dtype=float)
CHIP_WIDTH = SPEED_OF_LIGHT / 1.023e6
WAVELENGTH = SPEED_OF_LIGHT / 1575.42e6

PROJECT_PATH = Path(__file__).parents[1]
CONFIG_PATH = PROJECT_PATH / "configs"
DATA_PATH = PROJECT_PATH / "data"
RESULTS_PATH = PROJECT_PATH / "results"

DISABLE_PROGRESS = False

# Dynamic solution
arr = np.loadtxt(DATA_PATH / "deval_drive_50Hz.csv", delimiter=",", dtype=float, skiprows=1)
tt = arr[:,0]
truth_pos = arr[:,1:4]
truth_vel = arr[:,4:7]
del arr

if __name__ == "__main__":
  t0 =  tic("Starting...")
  
  # configuration (navsim)
  config = get_configuration(CONFIG_PATH)
  
  # simulate measurements (navsim)
  meas_sim = get_signal_simulation("measurement", config, DISABLE_PROGRESS)
  meas_sim.generate_truth(truth_pos, truth_vel)
  meas_sim.simulate()
  
  # initialize output
  vt_pos_err_enu = np.zeros(truth_pos.shape)
  vt_vel_err_enu = np.zeros(truth_vel.shape)
  vt_clk_bias_err = np.zeros(truth_pos.shape[0])
  vt_clk_drift_err = np.zeros(truth_vel.shape[0])
  vt_pos_var_enu = np.zeros(truth_pos.shape)
  vt_vel_var_enu = np.zeros(truth_vel.shape)
  vt_corr = np.zeros((truth_pos.shape[0], 6, len(meas_sim.observables[0])))
  vt_cn0 = np.zeros((truth_pos.shape[0], len(meas_sim.observables[0])))
  vt_dop = np.zeros((truth_pos.shape[0], 6))
  
  # vector tracking configuration
  vt_config = VDFLLConfig(
    order = 3,
    process_noise_stdev = 0.5,
    tap_spacing = 1.0,
    innovation_stdev = 3.0,
    cn0_buffer_len = 100,
    cn0 = np.array([emitter.cn0 for emitter in meas_sim.observables[0].values()]),
    pos = truth_pos[0,:],
    vel = truth_vel[0,:],
    clock_bias = meas_sim.rx_states.clock_bias[0],
    clock_drift = meas_sim.rx_states.clock_drift[0],
    clock_type = config.errors.rx_clock,
    is_signal_level = False,
    T = 0.02,
  )
  
  #* begin correlator simulation
  lla0 = ecef2lla(truth_pos[0,:])
  C_e_n = ecef2enuDcm(lla0)       # ecef to enu rotation matrix
  vdfll = VectorTrack(vt_config)
  for i, obs in tqdm(
                enumerate(meas_sim.observables), 
                total=len(meas_sim.observables),
                desc=default_logger.GenerateSring("[charlizard] VT Sim", Level.Info, Color.Info), 
                disable=DISABLE_PROGRESS, 
                ascii='.>#', 
                bar_format='{desc:<100}{percentage:3.0f}%|{bar:50}| {n_fmt}/{total_fmt} [{rate_fmt}]',
              ):
    # if i > 1:
    # predict observables
    pranges, prange_rates = vdfll.predict_observables(meas_sim.emitter_states.truth[i])
    
    # generate correlators
    true_cn0 = np.array([emitter.cn0 for emitter in meas_sim.observables[i].values()])
    corr_err = correlator_error(obs, pranges, prange_rates, CHIP_WIDTH, WAVELENGTH)
    corr = correlator_model(corr_err, true_cn0, 0.5, 0.02)
    vdfll.update_correlators(corr)
    
    # measurement update
    vdfll.measurement_update()
    
    # log performance
    vt_pos_err_enu[i,:] = ecef2enu(truth_pos[i,:], lla0) - ecef2enu(vdfll.rx_state[0:3], lla0)
    vt_vel_err_enu[i,:] = ecef2enuv(truth_vel[i,:], lla0) - ecef2enuv(vdfll.rx_state[3:6], lla0)
    vt_clk_bias_err[i] = meas_sim.rx_states.clock_bias[i] - vdfll.rx_state[-2]
    vt_clk_drift_err[i] = meas_sim.rx_states.clock_drift[i] - vdfll.rx_state[-1]
    vt_pos_var_enu[i,:] = np.diag(C_e_n @ vdfll.rx_cov[0:3, 0:3] @ C_e_n.T)
    vt_vel_var_enu[i,:] = np.diag(C_e_n @ vdfll.rx_cov[3:6, 3:6] @ C_e_n.T)
    vt_corr[i,0,:] = corr.IE
    vt_corr[i,1,:] = corr.IP
    vt_corr[i,2,:] = corr.IL
    vt_corr[i,3,:] = corr.QE
    vt_corr[i,4,:] = corr.QP
    vt_corr[i,5,:] = corr.QL 
    vt_cn0[i,:] = vdfll.cn0
    # vt_dop[i,:] = ?
    
    # if (epoch) % 500 == 0: # every 10 s
    #   t = ecef2enu(truth_pos[epoch,:], ecef2lla(truth_pos[0,:]))
    #   r = ecef2enu(vdfll.rx_state[0:3], ecef2lla(truth_pos[0,:]))
    #   print(#f'Rx ENU Pos: {ecef2enu(vdfll.rx_state[0:3], ecef2lla(truth_pos[0,:])):.2f}',
    #         f'Pos ENU Err: [{(t[0]-r[0]):+.2f}, {(t[1]-r[1]):+.2f}, {(t[2]-r[2]):+.2f}] ', 
    #         #f'Vel Err: {(truth_vel[epoch,:] - vdfll.rx_state[3:6]):.2f}', 
    #         f'chip err: {np.mean(corr_err.chip):+.2f} ',
    #         f'freq err: {np.mean(corr_err.freq):+.2f} ',
    #         f'phase err: {np.mean(corr_err.phase):+.2f} ',
    #         f'Clk Bias Err: {(meas_sim.rx_states.clock_bias[epoch] - vdfll.rx_state[-2]):+.2f} ', 
    #         f'CN0: {np.mean(vdfll.cn0):+.2f} ', 
    #       )
      
    # time update
    vdfll.time_update()
    
  # print(ecef2lla(truth_pos[0,:]) * LLA_R2D)
  t = ecef2enu(truth_pos[-1,:], ecef2lla(truth_pos[0,:]))
  r = ecef2enu(vdfll.rx_state[0:3], ecef2lla(truth_pos[0,:]))
  default_logger.Warn(f"Final ENU Error = [{(t[0]-r[0]):+.3f}, {(t[1]-r[1]):+.3f}, {(t[2]-r[2]):+.3f}], Norm = {np.linalg.norm(t-r):+.3f}")
  default_logger.Warn(f"Final Mean CN0 = {vdfll.cn0.mean()}")
  
  #* position, velocity, bias, drift plot results
  h1, ax1 = plt.subplots(nrows=3, ncols=1)
  h1.suptitle('ENU Position Error')
  ax1[0].plot(tt, vt_pos_err_enu[:,0] + 3*vt_pos_var_enu[:,0], 'r')
  ax1[0].plot(tt, vt_pos_err_enu[:,0] - 3*vt_pos_var_enu[:,0], 'r')
  ax1[0].plot(tt, vt_pos_err_enu[:,0], 'k')
  ax1[1].plot(tt, vt_pos_err_enu[:,1] + 3*vt_pos_var_enu[:,1], 'r')
  ax1[1].plot(tt, vt_pos_err_enu[:,1] - 3*vt_pos_var_enu[:,1], 'r')
  ax1[1].plot(tt, vt_pos_err_enu[:,1], 'k')
  ax1[2].plot(tt, vt_pos_err_enu[:,2] + 3*vt_pos_var_enu[:,2], 'r')
  ax1[2].plot(tt, vt_pos_err_enu[:,2] - 3*vt_pos_var_enu[:,2], 'r')
  ax1[2].plot(tt, vt_pos_err_enu[:,2], 'k')
  ax1[0].set_ylabel('East [m]')
  ax1[1].set_ylabel('North [m]')
  ax1[2].set_ylabel('Up [m]')
  ax1[2].set_xlabel('Time [s]')
  
  h2, ax2 = plt.subplots(nrows=3, ncols=1)
  h2.suptitle('ENU Velocity Error')
  ax2[0].plot(tt, vt_vel_err_enu[:,0] + 3*vt_vel_var_enu[:,0], 'r')
  ax2[0].plot(tt, vt_vel_err_enu[:,0] - 3*vt_vel_var_enu[:,0], 'r')
  ax2[0].plot(tt, vt_vel_err_enu[:,0], 'k')
  ax2[1].plot(tt, vt_vel_err_enu[:,1] + 3*vt_vel_var_enu[:,1], 'r')
  ax2[1].plot(tt, vt_vel_err_enu[:,1] - 3*vt_vel_var_enu[:,1], 'r')
  ax2[1].plot(tt, vt_vel_err_enu[:,1], 'k')
  ax2[2].plot(tt, vt_vel_err_enu[:,2] + 3*vt_vel_var_enu[:,2], 'r')
  ax2[2].plot(tt, vt_vel_err_enu[:,2] - 3*vt_vel_var_enu[:,2], 'r')
  ax2[2].plot(tt, vt_vel_err_enu[:,2], 'k')
  ax2[0].set_ylabel('East [m/s]')
  ax2[1].set_ylabel('North [m/s]')
  ax2[2].set_ylabel('Up [m/s]')
  ax2[2].set_xlabel('Time [s]')
  
  h3, ax3 = plt.subplots(nrows=2, ncols=1)
  h3.suptitle('Clock Errors')
  ax3[0].plot(tt, vt_clk_bias_err)
  ax3[1].plot(tt, vt_clk_drift_err)
  ax3[0].set_ylabel('Bias [m]')
  ax3[1].set_ylabel('Drift [m/s]')
  ax3[1].set_xlabel('Time [s]')
  
  #* cn0
  h4, ax4 = plt.subplots()
  h4.suptitle('CN0')
  ax4.plot(tt, vt_cn0)
  ax4.set_ylabel('Ratio [dB-Hz]')
  ax4.set_xlabel('Time [s]')
  
  plt.show()
  
  
  toc(t0, "Done!")
  