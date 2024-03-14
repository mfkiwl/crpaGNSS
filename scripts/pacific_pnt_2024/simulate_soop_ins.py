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

PROJECT_PATH = Path(__file__).parents[2]
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
    results = {
        "lla": np.zeros((n, 3)),
        "position": np.zeros((n, 3)),
        "velocity": np.zeros((n, 3)),
        "attitude": np.zeros((n, 3)),
        "clock": np.zeros((n, 2)),
        "position_error": np.zeros((n, 3)),
        "velocity_error": np.zeros((n, 3)),
        "attitude_error": np.zeros((n, 3)),
        "clock_error": np.zeros((n, 2)),
        "position_std_filter": np.zeros((n, 3)),
        "velocity_std_filter": np.zeros((n, 3)),
        "attitude_std_filter": np.zeros((n, 3)),
        "clock_std_filter": np.zeros((n, 2)),
        "dop": np.zeros((n, 6)),
    }

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
        desc=default_logger.GenerateSring("[charlizard] SOOP-INS ", Level.Info, Color.Info),
        disable=DISABLE_PROGRESS,
        ascii=".>#",
        bar_format="{desc}{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{rate_fmt}]",
        ncols=120,
    ):
        # time update
        # soop_ins.time_update(ins_sim.true_angular_velocity[i,:], ins_sim.true_specific_force[i,:])
        soop_ins.time_update(ins_sim.angular_velocity[i, :], ins_sim.specific_force[i, :])

        if i % f_update == 0:
            # grab true values from path simulator
            true_cn0 = np.array([emitter.cn0 for emitter in meas_sim.observables[j].values()])
            true_clock = np.array(
                [
                    meas_sim.rx_states.clock_bias[j],
                    meas_sim.rx_states.clock_drift[j],
                ]
            )
            wavelength = np.array(
                [
                    meas_sim.signal_properties[emitter.constellation].wavelength
                    for emitter in meas_sim.observables[j].values()
                ]
            )

            # always predict to update H matrix
            pranges, prange_rates = soop_ins.predict_observables(meas_sim.emitter_states.truth[j])

            # measurement update
            if measurement_update:
                # update filter with new observables
                soop_ins.cn0 = true_cn0
                meas_pranges = np.zeros(true_cn0.size)
                meas_prange_rates = np.array(
                    [emitter.pseudorange_rate for emitter in meas_sim.observables[j].values()]
                ) + np.random.randn(true_cn0.size) * prange_rate_residual_var(
                    true_cn0, 0.02, wavelength
                )
                soop_ins.measurement_update(
                    meas_pranges,
                    meas_prange_rates,
                    meas_sim.emitter_states.truth[j],
                    wavelength,
                )

            # * LOG RESULTS
            (
                results["position"][j, :],
                results["velocity"][j, :],
                results["attitude"][j, :],
                results["lla"][j, :],
                results["clock"][j, :],
            ) = soop_ins.extract_states
            results["position_error"][j, :] = (
                ins_sim.tangent_position[i, :] - results["position"][j, :]
            )
            results["velocity_error"][j, :] = (
                ins_sim.tangent_velocity[i, :] - results["velocity"][j, :]
            )
            results["attitude_error"][j, :] = ins_sim.euler_angles[i, :] - results["attitude"][j, :]
            results["clock_error"][j, :] = true_clock - results["clock"][j, :]
            (
                results["position_std_filter"][j, :],
                results["velocity_std_filter"][j, :],
                results["attitude_std_filter"][j, :],
                results["clock_std_filter"][j, :],
            ) = soop_ins.extract_stds
            results["dop"][j, :] = soop_ins.extract_dops

            j += 1

    return results


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

    results = run_simulation(config, meas_sim, ins_sim, MEAS_UPDATE, DISABLE_PROGRESS)
    default_logger.Warn(
        f"Final ENU Error = [{results['position_error'][-1,0]:+.3f}, "
        + f"{results['position_error'][-1,1]:+.3f}, "
        + f"{results['position_error'][-1,2]:+.3f}], "
        + f"Norm = {np.linalg.norm(results['position_error'][-1,:]):+.3f}"
    )

    # * position, velocity, bias, drift plot results
    h1, ax1 = plt.subplots(nrows=3, ncols=1)
    h1.suptitle("ENU Position Error")
    ax1[0].plot(
        ins_sim.time[::f_update],
        results["position_error"][:, 0] + 3 * results["position_error"][:, 0],
        "r",
    )
    ax1[0].plot(
        ins_sim.time[::f_update],
        results["position_error"][:, 0] - 3 * results["position_error"][:, 0],
        "r",
    )
    ax1[0].plot(ins_sim.time[::f_update], results["position_error"][:, 0], "k")
    ax1[1].plot(
        ins_sim.time[::f_update],
        results["position_error"][:, 1] + 3 * results["position_error"][:, 1],
        "r",
    )
    ax1[1].plot(
        ins_sim.time[::f_update],
        results["position_error"][:, 1] - 3 * results["position_error"][:, 1],
        "r",
    )
    ax1[1].plot(ins_sim.time[::f_update], results["position_error"][:, 1], "k")
    ax1[2].plot(
        ins_sim.time[::f_update],
        results["position_error"][:, 2] + 3 * results["position_error"][:, 2],
        "r",
    )
    ax1[2].plot(
        ins_sim.time[::f_update],
        results["position_error"][:, 2] - 3 * results["position_error"][:, 2],
        "r",
    )
    ax1[2].plot(ins_sim.time[::f_update], results["position_error"][:, 2], "k")
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
        ins_sim.time[::f_update],
        results["velocity_error"][:, 0] + 3 * results["velocity_error"][:, 0],
        "r",
    )
    ax2[0].plot(
        ins_sim.time[::f_update],
        results["velocity_error"][:, 0] - 3 * results["velocity_error"][:, 0],
        "r",
    )
    ax2[0].plot(ins_sim.time[::f_update], results["velocity_error"][:, 0], "k")
    ax2[1].plot(
        ins_sim.time[::f_update],
        results["velocity_error"][:, 1] + 3 * results["velocity_error"][:, 1],
        "r",
    )
    ax2[1].plot(
        ins_sim.time[::f_update],
        results["velocity_error"][:, 1] - 3 * results["velocity_error"][:, 1],
        "r",
    )
    ax2[1].plot(ins_sim.time[::f_update], results["velocity_error"][:, 1], "k")
    ax2[2].plot(
        ins_sim.time[::f_update],
        results["velocity_error"][:, 2] + 3 * results["velocity_error"][:, 2],
        "r",
    )
    ax2[2].plot(
        ins_sim.time[::f_update],
        results["velocity_error"][:, 2] - 3 * results["velocity_error"][:, 2],
        "r",
    )
    ax2[2].plot(ins_sim.time[::f_update], results["velocity_error"][:, 2], "k")
    ax2[0].set_ylabel("East [m/s]")
    ax2[1].set_ylabel("North [m/s]")
    ax2[2].set_ylabel("Up [m/s]")
    ax2[2].set_xlabel("Time [s]")
    ax2[0].grid()
    ax2[1].grid()
    ax2[2].grid()

    results["attitude_error"][results["attitude_error"] > 360] -= 360
    results["attitude_error"][results["attitude_error"] < -360] += 360
    h3, ax3 = plt.subplots(nrows=3, ncols=1)
    h3.suptitle("RPY Attitude Error")
    ax3[0].plot(
        ins_sim.time[::f_update],
        results["attitude_error"][:, 0] + 3 * results["attitude_error"][:, 0],
        "r",
    )
    ax3[0].plot(
        ins_sim.time[::f_update],
        results["attitude_error"][:, 0] - 3 * results["attitude_error"][:, 0],
        "r",
    )
    ax3[0].plot(ins_sim.time[::f_update], results["attitude_error"][:, 0], "k")
    ax3[1].plot(
        ins_sim.time[::f_update],
        results["attitude_error"][:, 1] + 3 * results["attitude_error"][:, 1],
        "r",
    )
    ax3[1].plot(
        ins_sim.time[::f_update],
        results["attitude_error"][:, 1] - 3 * results["attitude_error"][:, 1],
        "r",
    )
    ax3[1].plot(ins_sim.time[::f_update], results["attitude_error"][:, 1], "k")
    ax3[2].plot(
        ins_sim.time[::f_update],
        results["attitude_error"][:, 2] + 3 * results["attitude_error"][:, 2],
        "r",
    )
    ax3[2].plot(
        ins_sim.time[::f_update],
        results["attitude_error"][:, 2] - 3 * results["attitude_error"][:, 2],
        "r",
    )
    ax3[2].plot(ins_sim.time[::f_update], results["attitude_error"][:, 2], "k")
    ax3[0].set_ylabel("Roll [deg]")
    ax3[1].set_ylabel("Pitch [deg]")
    ax3[2].set_ylabel("Yaw [deg]")
    ax3[2].set_xlabel("Time [s]")
    ax3[0].grid()
    ax3[1].grid()
    ax3[2].grid()

    h4, ax4 = plt.subplots(nrows=2, ncols=1)
    h4.suptitle("Clock Errors")
    ax4[0].plot(
        ins_sim.time[::f_update],
        results["clock_error"][:, 0] + 3 * results["clock_error"][:, 0],
        "r",
    )
    ax4[0].plot(
        ins_sim.time[::f_update],
        results["clock_error"][:, 0] - 3 * results["clock_error"][:, 0],
        "r",
    )
    ax4[0].plot(ins_sim.time[::f_update], results["clock_error"][:, 0], "k")
    ax4[1].plot(
        ins_sim.time[::f_update],
        results["clock_error"][:, 1] + 3 * results["clock_error"][:, 1],
        "r",
    )
    ax4[1].plot(
        ins_sim.time[::f_update],
        results["clock_error"][:, 1] - 3 * results["clock_error"][:, 1],
        "r",
    )
    ax4[1].plot(ins_sim.time[::f_update], results["clock_error"][:, 1], "k")
    ax4[0].set_ylabel("Bias [m]")
    ax4[1].set_ylabel("Drift [m/s]")
    ax4[1].set_xlabel("Time [s]")
    ax4[0].grid()
    ax4[1].grid()

    # * 2d position plot
    h5, ax5 = plt.subplots()
    h5.suptitle("2D ENU Position")
    ax5.plot(ins_sim.tangent_position[:, 0], ins_sim.tangent_position[:, 1])
    ax5.plot(results["position"][:, 0], results["position"][:, 1])
    ax5.set_ylabel("North [m]")
    ax5.set_xlabel("East [m]")
    ax5.axis("square")
    ax5.grid()

    # * dops
    h6, ax6 = plt.subplots()
    h6.suptitle("Dilution of Precision")
    ax6.plot(ins_sim.time[::f_update], results["dop"])
    ax6.set_ylabel("DOP")
    ax6.set_xlabel("Time [s]")
    ax6.legend(["GDOP", "PDOP", "HDOP", "VDOP", "TDOP", "# Emitters"])

    plt.show()
    print()
