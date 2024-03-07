
import numpy as np
from tqdm import tqdm
from pathlib import Path
from log_utils import *

from navtools.constants import SPEED_OF_LIGHT
from navtools.conversions.coordinates import ecef2lla, ecef2enu, ecef2enuv, ecef2enuDcm
from navsim.configuration import get_configuration
from navsim.simulations.ins import INSSimulation
from navsim.simulations.measurement import MeasurementSimulation
from navsim.error_models.imu import HG1700, IMU

from charlizard.navigators.soop_ins import GnssIns
from charlizard.navigators.structures import GNSSINSConfig
from charlizard.models.correlators import correlator_error, correlator_model
from charlizard.models.discriminator import prange_rate_residual_var, prange_residual_var

from dataclasses import dataclass, field

import matplotlib.pyplot as plt

PROJECT_PATH = Path(__file__).parents[1]
CONFIG_PATH = PROJECT_PATH / "configs"
DATA_PATH = PROJECT_PATH / "data"
RESULTS_PATH = PROJECT_PATH / "results"

CHIP_WIDTH = SPEED_OF_LIGHT / 1.023e6
WAVELENGTH = SPEED_OF_LIGHT / 1575.42e6
DISABLE_PROGRESS = False


#* SIMULATE OBSERVABLES
config = get_configuration(CONFIG_PATH)

ins_sim = INSSimulation(config)
ins_sim.motion_commands(DATA_PATH)
ins_sim.simulate()

f_update = int(ins_sim.imu_model.f / 50)
meas_sim = MeasurementSimulation(config)
meas_sim.generate_truth(ins_sim.ecef_position[::f_update,:], ins_sim.ecef_velocity[::f_update,:])
meas_sim.simulate()

gnss_ins_config = GNSSINSConfig( 
    T = 1/ins_sim.imu_model.f,
    tap_spacing = 0.5, 
    innovation_stdev = 3, 
    cn0_buffer_len = 100, 
    cn0 = np.array([emitter.cn0 for emitter in meas_sim.observables[0].values()]),
    pos = ins_sim.ecef_position[0,:], 
    vel = ins_sim.ecef_velocity[0,:], 
    att = ins_sim.euler_angles[0,:], 
    clock_bias = meas_sim.rx_states.clock_bias[0], 
    clock_drift = meas_sim.rx_states.clock_drift[0], 
    clock_type = config.errors.rx_clock,
    imu_model = config.imu.model,
    coupling = 'tight',
  )

#* INITIALIZE OUTPUTS
n = len(meas_sim.observables)
dc_pos_enu = np.zeros((n,3))
dc_vel_enu = np.zeros((n,3))
dc_att_rpy = np.zeros((n,3))
dc_clk = np.zeros((n,2))
dc_pos_err_enu = np.zeros((n,3))
dc_vel_err_enu = np.zeros((n,3))
dc_att_err_rpy = np.zeros((n,3))
dc_clk_err = np.zeros((n,2))
dc_pos_var_enu = np.zeros((n,3))
dc_vel_var_enu = np.zeros((n,3))
dc_att_var_rpy = np.zeros((n,3))
dc_clk_var = np.zeros((n,2))
dc_corr = np.zeros((n, 6, len(meas_sim.observables[0])))
dc_cn0 = np.zeros((n, len(meas_sim.observables[0])))
dc_dop = np.zeros((n, 6))


#* RUN SIMULATION
gnss_ins = GnssIns(gnss_ins_config)
j = 0
for i in tqdm(
          range(1, ins_sim.time.size), 
          total=ins_sim.time.size,
          desc=default_logger.GenerateSring("[charlizard] Deep Coupling", Level.Info, Color.Info), 
          disable=DISABLE_PROGRESS, 
          ascii='.>#', 
          bar_format='{desc:<100}{percentage:3.0f}%|{bar:50}| {n_fmt}/{total_fmt} [{rate_fmt}]',
         ):
  gnss_ins.mechanize(ins_sim.true_angular_velocity[i,:], ins_sim.true_specific_force[i,:])
  gnss_ins.mechanize(ins_sim.angular_velocity[i,:], ins_sim.specific_force[i,:])
  
  # time update
  gnss_ins.time_update(ins_sim.true_angular_velocity[i,:], ins_sim.true_specific_force[i,:])
  gnss_ins.time_update(ins_sim.angular_velocity[i,:], ins_sim.specific_force[i,:])
  
  if i % f_update == 0:
    pranges, prange_rates = gnss_ins.predict_observables(meas_sim.emitter_states.truth[j])
    
    # generate correlators
    true_cn0 = np.array([emitter.cn0 for emitter in meas_sim.observables[j].values()])
    corr_err = correlator_error(meas_sim.observables[j], pranges, prange_rates, CHIP_WIDTH, WAVELENGTH)
    corr = correlator_model(corr_err, true_cn0, 0.5, 0.02)
    gnss_ins.update_correlators(corr)
    
    # measurement update
    meas_pranges = np.random.randn(true_cn0.size)*prange_residual_var(true_cn0, 0.02, CHIP_WIDTH) \
      + np.array([emitter.code_pseudorange for emitter in meas_sim.observables[j].values()])
    meas_prange_rates = np.random.randn(true_cn0.size)*prange_rate_residual_var(true_cn0, 0.02, WAVELENGTH) \
      + np.array([emitter.pseudorange_rate for emitter in meas_sim.observables[j].values()])
    gnss_ins.measurement_update(meas_pranges, meas_prange_rates)
  
    # log
    dc_pos_enu[j,:], dc_vel_enu[j,:], dc_att_rpy[j,:], _, dc_clk[j,:] = gnss_ins.extract_states
    dc_pos_err_enu[j,:] = ins_sim.tangent_position[i,:] - dc_pos_enu[j,:]
    dc_vel_err_enu[j,:] = ins_sim.tangent_velocity[i,:] - dc_vel_enu[j,:]
    dc_att_err_rpy[j,:] = ins_sim.euler_angles[i,:] - dc_att_rpy[j,:]
    dc_clk_err[j] = np.array([meas_sim.rx_states.clock_bias[j], meas_sim.rx_states.clock_drift[j]]) - dc_clk[j,:]
    dc_pos_var_enu[j,:], dc_vel_var_enu[j,:], dc_att_var_rpy[j,:], dc_clk_var[j,:] = gnss_ins.extract_stds
    dc_corr[j,0,:] = corr.IE
    dc_corr[j,1,:] = corr.IP
    dc_corr[j,2,:] = corr.IL
    dc_corr[j,3,:] = corr.QE
    dc_corr[j,4,:] = corr.QP
    dc_corr[j,5,:] = corr.QL
    dc_cn0[j,:] = gnss_ins.cn0
    dc_dop[j,:] = gnss_ins.extract_dops
    
default_logger.Warn(f"Final ENU Error = [{dc_pos_err_enu[-1,0]:+.3f}, {dc_pos_err_enu[-1,1]:+.3f}, {dc_pos_err_enu[-1,2]:+.3f}], Norm = {np.linalg.norm(dc_pos_err_enu[-1,:]):+.3f}")
default_logger.Warn(f"Final Mean CN0 = {gnss_ins.cn0.mean()}")
  

#* position, velocity, bias, drift plot results
h1, ax1 = plt.subplots(nrows=3, ncols=1)
h1.suptitle('ENU Position Error')
ax1[0].plot(ins_sim.time[::f_update], dc_pos_err_enu[:,0] + 3*dc_pos_var_enu[:,0], 'r')
ax1[0].plot(ins_sim.time[::f_update], dc_pos_err_enu[:,0] - 3*dc_pos_var_enu[:,0], 'r')
ax1[0].plot(ins_sim.time[::f_update], dc_pos_err_enu[:,0], 'k')
ax1[1].plot(ins_sim.time[::f_update], dc_pos_err_enu[:,1] + 3*dc_pos_var_enu[:,1], 'r')
ax1[1].plot(ins_sim.time[::f_update], dc_pos_err_enu[:,1] - 3*dc_pos_var_enu[:,1], 'r')
ax1[1].plot(ins_sim.time[::f_update], dc_pos_err_enu[:,1], 'k')
ax1[2].plot(ins_sim.time[::f_update], dc_pos_err_enu[:,2] + 3*dc_pos_var_enu[:,2], 'r')
ax1[2].plot(ins_sim.time[::f_update], dc_pos_err_enu[:,2] - 3*dc_pos_var_enu[:,2], 'r')
ax1[2].plot(ins_sim.time[::f_update], dc_pos_err_enu[:,2], 'k')
ax1[0].set_ylabel('East [m]')
ax1[1].set_ylabel('North [m]')
ax1[2].set_ylabel('Up [m]')
ax1[2].set_xlabel('Time [s]')

h2, ax2 = plt.subplots(nrows=3, ncols=1)
h2.suptitle('ENU Velocity Error')
ax2[0].plot(ins_sim.time[::f_update], dc_vel_err_enu[:,0] + 3*dc_vel_var_enu[:,0], 'r')
ax2[0].plot(ins_sim.time[::f_update], dc_vel_err_enu[:,0] - 3*dc_vel_var_enu[:,0], 'r')
ax2[0].plot(ins_sim.time[::f_update], dc_vel_err_enu[:,0], 'k')
ax2[1].plot(ins_sim.time[::f_update], dc_vel_err_enu[:,1] + 3*dc_vel_var_enu[:,1], 'r')
ax2[1].plot(ins_sim.time[::f_update], dc_vel_err_enu[:,1] - 3*dc_vel_var_enu[:,1], 'r')
ax2[1].plot(ins_sim.time[::f_update], dc_vel_err_enu[:,1], 'k')
ax2[2].plot(ins_sim.time[::f_update], dc_vel_err_enu[:,2] + 3*dc_vel_var_enu[:,2], 'r')
ax2[2].plot(ins_sim.time[::f_update], dc_vel_err_enu[:,2] - 3*dc_vel_var_enu[:,2], 'r')
ax2[2].plot(ins_sim.time[::f_update], dc_vel_err_enu[:,2], 'k')
ax2[0].set_ylabel('East [m/s]')
ax2[1].set_ylabel('North [m/s]')
ax2[2].set_ylabel('Up [m/s]')
ax2[2].set_xlabel('Time [s]')

dc_att_err_rpy[dc_att_err_rpy > 360] -= 360 
h3, ax3 = plt.subplots(nrows=3, ncols=1)
h3.suptitle('RPY Attitude Error')
ax3[0].plot(ins_sim.time[::f_update], dc_att_err_rpy[:,0] + 3*dc_att_var_rpy[:,0], 'r')
ax3[0].plot(ins_sim.time[::f_update], dc_att_err_rpy[:,0] - 3*dc_att_var_rpy[:,0], 'r')
ax3[0].plot(ins_sim.time[::f_update], dc_att_err_rpy[:,0], 'k')
ax3[1].plot(ins_sim.time[::f_update], dc_att_err_rpy[:,1] + 3*dc_att_var_rpy[:,1], 'r')
ax3[1].plot(ins_sim.time[::f_update], dc_att_err_rpy[:,1] - 3*dc_att_var_rpy[:,1], 'r')
ax3[1].plot(ins_sim.time[::f_update], dc_att_err_rpy[:,1], 'k')
ax3[2].plot(ins_sim.time[::f_update], dc_att_err_rpy[:,2] + 3*dc_att_var_rpy[:,2], 'r')
ax3[2].plot(ins_sim.time[::f_update], dc_att_err_rpy[:,2] - 3*dc_att_var_rpy[:,2], 'r')
ax3[2].plot(ins_sim.time[::f_update], dc_att_err_rpy[:,2], 'k')
ax3[0].set_ylabel('Roll [deg]')
ax3[1].set_ylabel('Pitch [deg]')
ax3[2].set_ylabel('Yaw [deg]')
ax3[2].set_xlabel('Time [s]')

h4, ax4 = plt.subplots(nrows=2, ncols=1)
h4.suptitle('Clock Errors')
ax4[0].plot(ins_sim.time[::f_update], dc_clk_err[:,0] + 3*dc_clk_var[:,0], 'r')
ax4[0].plot(ins_sim.time[::f_update], dc_clk_err[:,0] - 3*dc_clk_var[:,0], 'r')
ax4[0].plot(ins_sim.time[::f_update], dc_clk_err[:,0], 'k')
ax4[1].plot(ins_sim.time[::f_update], dc_clk_err[:,1] + 3*dc_clk_var[:,1], 'r')
ax4[1].plot(ins_sim.time[::f_update], dc_clk_err[:,1] - 3*dc_clk_var[:,1], 'r')
ax4[1].plot(ins_sim.time[::f_update], dc_clk_err[:,1], 'k')
ax4[0].set_ylabel('Bias [m]')
ax4[1].set_ylabel('Drift [m/s]')
ax4[1].set_xlabel('Time [s]')

#* cn0
h5, ax5 = plt.subplots()
h5.suptitle('CN0')
ax5.plot(ins_sim.time[::f_update], dc_cn0)
ax5.set_ylabel('Ratio [dB-Hz]')
ax5.set_xlabel('Time [s]')

#* 2d position plot
h6, ax6 = plt.subplots()
h6.suptitle('2D ENU Position')
ax6.plot(ins_sim.tangent_position[:,0], ins_sim.tangent_position[:,1])
ax6.plot(dc_pos_enu[:,0], dc_pos_enu[:,1])
ax6.set_ylabel('North [m]')
ax6.set_xlabel('East [m]')

#* dops
h7, ax7 = plt.subplots()
h7.suptitle('Dilution of Precision')
ax7.plot(ins_sim.time[::f_update], dc_dop[:,0:-1])
ax7.set_ylabel('DOP')
ax7.set_xlabel('Time [s]')
ax7.legend(['GDOP', 'PDOP', 'HDOP', 'VDOP', 'TDOP'])

#* correlators
h8, ax8 = plt.subplots()
h8.suptitle('Correlators')
ax8.plot(dc_corr[:,4,0], dc_corr[:,1,0], 'r.')
ax8.plot(dc_corr[:,3,0], dc_corr[:,0,0], 'g.')
ax8.plot(dc_corr[:,5,0], dc_corr[:,2,0], 'b.')
ax8.set_ylabel('Imaginary')
ax8.set_xlabel('Real')
ax8.legend(['Prompt', 'Early', 'Late'])

plt.show()

print()