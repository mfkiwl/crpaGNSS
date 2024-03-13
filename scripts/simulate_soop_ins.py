import numpy as np
from tqdm import tqdm
from pathlib import Path
from log_utils import *

from navsim.configuration import get_configuration, SimulationConfiguration
from navsim.simulations.measurement import MeasurementSimulation
from navsim.simulations.ins import INSSimulation
from navtools.constants import SPEED_OF_LIGHT

from charlizard.navigators.soop_ins import SoopIns
from charlizard.navigators.structures import GNSSINSConfig
from charlizard.models.discriminator import (
    prange_rate_residual_var,
    prange_residual_var,
)

import matplotlib.pyplot as plt

PROJECT_PATH = Path(__file__).parents[1]
CONFIG_PATH = PROJECT_PATH / "configs"
DATA_PATH = PROJECT_PATH / "data"
RESULTS_PATH = PROJECT_PATH / "results"

CHIP_WIDTH = SPEED_OF_LIGHT / 1.023e6
WAVELENGTH = SPEED_OF_LIGHT / 1575.42e6
DISABLE_PROGRESS = False
MEAS_UPDATE = True


def run_simulation(
    config: SimulationConfiguration,
    meas_sim: MeasurementSimulation,
    ins_sim: INSSimulation,
    measurement_update: bool = True,
    DISABLE_PROGRESS: bool = True,
):
    # * INITIALIZE OUTPUTS
    n = len(meas_sim.observables)
    pos_enu = np.zeros((n, 3))
    vel_enu = np.zeros((n, 3))
    att_rpy = np.zeros((n, 3))
    clk = np.zeros((n, 2))
    pos_err_enu = np.zeros((n, 3))
    vel_err_enu = np.zeros((n, 3))
    att_err_rpy = np.zeros((n, 3))
    clk_err = np.zeros((n, 2))
    pos_var_enu = np.zeros((n, 3))
    vel_var_enu = np.zeros((n, 3))
    att_var_rpy = np.zeros((n, 3))
    clk_var = np.zeros((n, 2))
    dop = np.zeros((n, 6))

    soop_ins_config = GNSSINSConfig(
        T=1 / ins_sim.imu_model.f,
        tap_spacing=0.5,
        innovation_stdev=3,
        cn0_buffer_len=100,
        cn0=np.array([emitter.cn0 for emitter in meas_sim.observables[0].values()]),
        pos=ins_sim.ecef_position[0, :] + np.random.randn(3) * 0.3,
        vel=ins_sim.ecef_velocity[0, :] + np.random.randn(3) * 0.01,
        att=ins_sim.euler_angles[0, :] + np.random.randn(3) * 0.1,
        clock_bias=meas_sim.rx_states.clock_bias[0],
        clock_drift=meas_sim.rx_states.clock_drift[0],
        clock_type=config.errors.rx_clock,
        imu_model=config.imu.model,
        coupling="tight",
        T_rcvr=1 / config.time.fsim,
    )
    f_update = int(ins_sim.imu_model.f / config.time.fsim)

    # * RUN SIMULATION
    soop_ins = SoopIns(soop_ins_config)
    j = 0
    for i in tqdm(
        range(ins_sim.time.size),
        total=ins_sim.time.size,
        desc=default_logger.GenerateSring(
            "[charlizard] SOOP-INS", Level.Info, Color.Info
        ),
        disable=DISABLE_PROGRESS,
        ascii=".>#",
        bar_format="{desc}{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{rate_fmt}]",
        ncols=120,
    ):
        # time update
        # soop_ins.time_update(ins_sim.true_angular_velocity[i,:], ins_sim.true_specific_force[i,:])
        soop_ins.time_update(
            ins_sim.angular_velocity[i, :], ins_sim.specific_force[i, :]
        )

        if i % f_update == 0:
            true_cn0 = np.array(
                [emitter.cn0 for emitter in meas_sim.observables[j].values()]
            )
            pranges, prange_rates = soop_ins.predict_observables(
                meas_sim.emitter_states.truth[j]
            )
            wavelength = np.array(
                [
                    meas_sim.signal_properties[emitter.constellation].wavelength
                    for emitter in meas_sim.observables[j].values()
                ]
            )

            # measurement update
            if measurement_update:
                soop_ins.cn0 = true_cn0
                meas_pranges = np.zeros(true_cn0.size)
                meas_prange_rates = np.array(
                    [
                        emitter.pseudorange_rate
                        for emitter in meas_sim.observables[j].values()
                    ]
                )
                meas_prange_rates += np.random.randn(
                    true_cn0.size
                ) * prange_rate_residual_var(true_cn0, 0.02, wavelength)
                soop_ins.measurement_update(
                    meas_pranges,
                    meas_prange_rates,
                    meas_sim.emitter_states.truth[j],
                    wavelength,
                )

            # log
            pos_enu[j, :], vel_enu[j, :], att_rpy[j, :], _, clk[j, :] = (
                soop_ins.extract_states
            )
            pos_err_enu[j, :] = ins_sim.tangent_position[i, :] - pos_enu[j, :]
            vel_err_enu[j, :] = ins_sim.tangent_velocity[i, :] - vel_enu[j, :]
            att_err_rpy[j, :] = ins_sim.euler_angles[i, :] - att_rpy[j, :]
            clk_err[j] = (
                np.array(
                    [
                        meas_sim.rx_states.clock_bias[j],
                        meas_sim.rx_states.clock_drift[j],
                    ]
                )
                - clk[j, :]
            )
            pos_var_enu[j, :], vel_var_enu[j, :], att_var_rpy[j, :], clk_var[j, :] = (
                soop_ins.extract_stds
            )
            dop[j, :] = soop_ins.extract_dops

            j += 1

    return (
        pos_enu,
        vel_enu,
        att_rpy,
        clk,
        pos_err_enu,
        vel_err_enu,
        att_err_rpy,
        clk_err,
        pos_var_enu,
        vel_var_enu,
        att_var_rpy,
        clk_var,
        dop,
    )


if __name__ == "__main__":
    # * SIMULATE OBSERVABLES
    config = get_configuration(CONFIG_PATH)

    ins_sim = INSSimulation(config)
    ins_sim.motion_commands(DATA_PATH)
    ins_sim.simulate()

    f_rcvr = config.time.fsim
    f_update = int(ins_sim.imu_model.f / f_rcvr)
    meas_sim = MeasurementSimulation(config)
    meas_sim.generate_truth(
        ins_sim.ecef_position[::f_update, :], ins_sim.ecef_velocity[::f_update, :]
    )
    meas_sim.simulate()

    (
        pos_enu,
        vel_enu,
        att_rpy,
        clk,
        pos_err_enu,
        vel_err_enu,
        att_err_rpy,
        clk_err,
        pos_var_enu,
        vel_var_enu,
        att_var_rpy,
        clk_var,
        dop,
    ) = run_simulation(config, meas_sim, ins_sim, MEAS_UPDATE, DISABLE_PROGRESS)
    default_logger.Warn(
        f"Final ENU Error = [{pos_err_enu[-1,0]:+.3f}, {pos_err_enu[-1,1]:+.3f}, {pos_err_enu[-1,2]:+.3f}], Norm = {np.linalg.norm(pos_err_enu[-1,:]):+.3f}"
    )

    # * position, velocity, bias, drift plot results
    h1, ax1 = plt.subplots(nrows=3, ncols=1)
    h1.suptitle("ENU Position Error")
    ax1[0].plot(
        ins_sim.time[::f_update], pos_err_enu[:, 0] + 3 * pos_var_enu[:, 0], "r"
    )
    ax1[0].plot(
        ins_sim.time[::f_update], pos_err_enu[:, 0] - 3 * pos_var_enu[:, 0], "r"
    )
    ax1[0].plot(ins_sim.time[::f_update], pos_err_enu[:, 0], "k")
    ax1[1].plot(
        ins_sim.time[::f_update], pos_err_enu[:, 1] + 3 * pos_var_enu[:, 1], "r"
    )
    ax1[1].plot(
        ins_sim.time[::f_update], pos_err_enu[:, 1] - 3 * pos_var_enu[:, 1], "r"
    )
    ax1[1].plot(ins_sim.time[::f_update], pos_err_enu[:, 1], "k")
    ax1[2].plot(
        ins_sim.time[::f_update], pos_err_enu[:, 2] + 3 * pos_var_enu[:, 2], "r"
    )
    ax1[2].plot(
        ins_sim.time[::f_update], pos_err_enu[:, 2] - 3 * pos_var_enu[:, 2], "r"
    )
    ax1[2].plot(ins_sim.time[::f_update], pos_err_enu[:, 2], "k")
    ax1[0].set_ylabel("East [m]")
    ax1[1].set_ylabel("North [m]")
    ax1[2].set_ylabel("Up [m]")
    ax1[2].set_xlabel("Time [s]")
    ax1[0].grid()
    ax1[1].grid()
    ax1[2].grid()

    h2, ax2 = plt.subplots(nrows=3, ncols=1)
    h2.suptitle("ENU Velocity Error")
    ax2[0].plot(
        ins_sim.time[::f_update], vel_err_enu[:, 0] + 3 * vel_var_enu[:, 0], "r"
    )
    ax2[0].plot(
        ins_sim.time[::f_update], vel_err_enu[:, 0] - 3 * vel_var_enu[:, 0], "r"
    )
    ax2[0].plot(ins_sim.time[::f_update], vel_err_enu[:, 0], "k")
    ax2[1].plot(
        ins_sim.time[::f_update], vel_err_enu[:, 1] + 3 * vel_var_enu[:, 1], "r"
    )
    ax2[1].plot(
        ins_sim.time[::f_update], vel_err_enu[:, 1] - 3 * vel_var_enu[:, 1], "r"
    )
    ax2[1].plot(ins_sim.time[::f_update], vel_err_enu[:, 1], "k")
    ax2[2].plot(
        ins_sim.time[::f_update], vel_err_enu[:, 2] + 3 * vel_var_enu[:, 2], "r"
    )
    ax2[2].plot(
        ins_sim.time[::f_update], vel_err_enu[:, 2] - 3 * vel_var_enu[:, 2], "r"
    )
    ax2[2].plot(ins_sim.time[::f_update], vel_err_enu[:, 2], "k")
    ax2[0].set_ylabel("East [m/s]")
    ax2[1].set_ylabel("North [m/s]")
    ax2[2].set_ylabel("Up [m/s]")
    ax2[2].set_xlabel("Time [s]")
    ax2[0].grid()
    ax2[1].grid()
    ax2[2].grid()

    att_err_rpy[att_err_rpy > 360] -= 360
    h3, ax3 = plt.subplots(nrows=3, ncols=1)
    h3.suptitle("RPY Attitude Error")
    ax3[0].plot(
        ins_sim.time[::f_update], att_err_rpy[:, 0] + 3 * att_var_rpy[:, 0], "r"
    )
    ax3[0].plot(
        ins_sim.time[::f_update], att_err_rpy[:, 0] - 3 * att_var_rpy[:, 0], "r"
    )
    ax3[0].plot(ins_sim.time[::f_update], att_err_rpy[:, 0], "k")
    ax3[1].plot(
        ins_sim.time[::f_update], att_err_rpy[:, 1] + 3 * att_var_rpy[:, 1], "r"
    )
    ax3[1].plot(
        ins_sim.time[::f_update], att_err_rpy[:, 1] - 3 * att_var_rpy[:, 1], "r"
    )
    ax3[1].plot(ins_sim.time[::f_update], att_err_rpy[:, 1], "k")
    ax3[2].plot(
        ins_sim.time[::f_update], att_err_rpy[:, 2] + 3 * att_var_rpy[:, 2], "r"
    )
    ax3[2].plot(
        ins_sim.time[::f_update], att_err_rpy[:, 2] - 3 * att_var_rpy[:, 2], "r"
    )
    ax3[2].plot(ins_sim.time[::f_update], att_err_rpy[:, 2], "k")
    ax3[0].set_ylabel("Roll [deg]")
    ax3[1].set_ylabel("Pitch [deg]")
    ax3[2].set_ylabel("Yaw [deg]")
    ax3[2].set_xlabel("Time [s]")
    ax3[0].grid()
    ax3[1].grid()
    ax3[2].grid()

    h4, ax4 = plt.subplots(nrows=2, ncols=1)
    h4.suptitle("Clock Errors")
    ax4[0].plot(ins_sim.time[::f_update], clk_err[:, 0] + 3 * clk_var[:, 0], "r")
    ax4[0].plot(ins_sim.time[::f_update], clk_err[:, 0] - 3 * clk_var[:, 0], "r")
    ax4[0].plot(ins_sim.time[::f_update], clk_err[:, 0], "k")
    ax4[1].plot(ins_sim.time[::f_update], clk_err[:, 1] + 3 * clk_var[:, 1], "r")
    ax4[1].plot(ins_sim.time[::f_update], clk_err[:, 1] - 3 * clk_var[:, 1], "r")
    ax4[1].plot(ins_sim.time[::f_update], clk_err[:, 1], "k")
    ax4[0].set_ylabel("Bias [m]")
    ax4[1].set_ylabel("Drift [m/s]")
    ax4[1].set_xlabel("Time [s]")
    ax4[0].grid()
    ax4[1].grid()

    # * 2d position plot
    h5, ax5 = plt.subplots()
    h5.suptitle("2D ENU Position")
    ax5.plot(ins_sim.tangent_position[:, 0], ins_sim.tangent_position[:, 1])
    ax5.plot(pos_enu[:, 0], pos_enu[:, 1])
    ax5.set_ylabel("North [m]")
    ax5.set_xlabel("East [m]")
    ax5.axis("square")
    ax5.grid()

    # * dops
    h6, ax6 = plt.subplots()
    h6.suptitle("Dilution of Precision")
    ax6.plot(ins_sim.time[::f_update], dop[:, 0:-1])
    ax6.set_ylabel("DOP")
    ax6.set_xlabel("Time [s]")
    ax6.legend(["GDOP", "PDOP", "HDOP", "VDOP", "TDOP"])

    plt.show()
    print()
