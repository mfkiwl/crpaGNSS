import numpy as np
from tqdm import tqdm
from pathlib import Path
from log_utils import *
from datetime import datetime, timedelta, timezone

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
    # * INITIALIZE SOOP INS FILTER
    soop_ins_config = GNSSINSConfig(
        T=1 / ins_sim.imu_model.f,
        tap_spacing=0.5,
        innovation_stdev=3,
        cn0_buffer_len=100,
        cn0=np.array([emitter.cn0 for emitter in meas_sim.observables[0].values()]),
        pos=ins_sim.ecef_position[0, :],
        vel=ins_sim.ecef_velocity[0, :],
        att=ins_sim.euler_angles[0, :],
        clock_bias=meas_sim.rx_states.clock_bias[0],
        clock_drift=meas_sim.rx_states.clock_drift[0],
        clock_type=config.errors.rx_clock,
        imu_model=config.imu.model,
        coupling="tight",
        T_rcvr=0.2,
    )
    f_sim = int(config.time.fsim)
    f_update = 10

    # * INITIALIZE SIMULATION
    soop_ins = SoopIns(soop_ins_config)
    time = datetime(
        config.time.year, config.time.month, config.time.day, config.time.hour, config.time.minute, config.time.second
    ).replace(tzinfo=timezone.utc)
    delta = timedelta(seconds=0.2)
    delta2 = timedelta(seconds=0.4)
    sim_delta = timedelta(seconds=1 / f_sim)

    # * INITIALIZE OUTPUTS
    n = int(len(meas_sim.observables) / f_update)
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

    # * RUN SIMULATION
    # sim_idx = 1
    out_idx = 0
    for imu_idx in tqdm(
        range(ins_sim.time.size),
        total=ins_sim.time.size,
        desc=default_logger.GenerateSring("[charlizard] SOOP-INS ", Level.Info, Color.Info),
        disable=DISABLE_PROGRESS,
        ascii=".>#",
        bar_format="{desc}{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{rate_fmt}]",
        ncols=120,
    ):
        # time update
        # soop_ins.time_update(ins_sim.true_angular_velocity[imu_idx, :], ins_sim.true_specific_force[imu_idx, :])
        soop_ins.time_update(ins_sim.angular_velocity[imu_idx, :], ins_sim.specific_force[imu_idx, :])

        if imu_idx % f_update == 0:  # and (imu_idx > 0):
            # grab true values from path simulator
            true_clock = np.array([meas_sim.rx_states.clock_bias[imu_idx], meas_sim.rx_states.clock_drift[imu_idx]])

            # measurement update
            if measurement_update:
                # ? check if observables are available, sort by key
                keys = meas_sim.observables[imu_idx].keys()
                emitter_states = dict(sorted(meas_sim.emitter_states.truth[imu_idx].items()))
                observables = dict(sorted(meas_sim.observables[imu_idx].items()))
                # emitters = []

                emitters = [e for e in meas_sim._MeasurementSimulation__emitters._skyfield_satellites if e.name in keys]
                names = [e.name for e in emitters]
                emitters = [emitters[i] for i in sorted(range(len(names)), key=lambda index: names[index])]
                times = [time - delta2, time - delta, time, time + delta, time + delta2]
                times = meas_sim._MeasurementSimulation__emitters._ts.from_datetimes(times)

                # update filter with new observables
                wavelength = np.array(
                    [
                        SPEED_OF_LIGHT / meas_sim.signal_properties[emitter.constellation].fcarrier
                        for emitter in observables.values()
                    ]
                )
                soop_ins.cn0 = np.array([emitter.cn0 for emitter in observables.values()])
                meas_prange_rates = np.array([emitter.pseudorange_rate for emitter in observables.values()])
                noise = np.random.randn(soop_ins.cn0.size) * prange_rate_residual_var(soop_ins.cn0, 0.02, wavelength)

                soop_ins.measurement_update(
                    noise,
                    emitter_states,
                    wavelength,
                    ins_sim.ecef_position[imu_idx, :],
                    ins_sim.ecef_velocity[imu_idx, :],
                    true_clock / SPEED_OF_LIGHT,
                    emitters,
                    times,
                )

            # * LOG RESULTS
            (
                results["position"][out_idx, :],
                results["velocity"][out_idx, :],
                results["attitude"][out_idx, :],
                results["lla"][out_idx, :],
                results["clock"][out_idx, :],
            ) = soop_ins.extract_states
            results["position_error"][out_idx, :] = (
                ins_sim.tangent_position[imu_idx, :] - results["position"][out_idx, :]
            )
            results["velocity_error"][out_idx, :] = (
                ins_sim.tangent_velocity[imu_idx, :] - results["velocity"][out_idx, :]
            )
            results["attitude_error"][out_idx, :] = ins_sim.euler_angles[imu_idx, :] - results["attitude"][out_idx, :]
            results["clock_error"][out_idx, :] = true_clock - results["clock"][out_idx, :]
            (
                results["position_std_filter"][out_idx, :],
                results["velocity_std_filter"][out_idx, :],
                results["attitude_std_filter"][out_idx, :],
                results["clock_std_filter"][out_idx, :],
            ) = soop_ins.extract_stds
            results["dop"][out_idx, :] = soop_ins.extract_dops

            out_idx += 1
            # print(time)
            time += sim_delta
            # sim_idx += 1

    return results


if __name__ == "__main__":
    # * SIMULATE OBSERVABLES
    config = get_configuration(CONFIG_PATH)

    ins_sim = INSSimulation(config)
    ins_sim.motion_commands(DATA_PATH)
    ins_sim.simulate()

    f_rcvr = config.time.fsim
    f_update = 10
    meas_sim = MeasurementSimulation(config)
    # meas_sim.generate_truth(ins_sim.ecef_position[::f_update, :], ins_sim.ecef_velocity[::f_update, :])
    meas_sim.generate_truth(ins_sim.ecef_position, ins_sim.ecef_velocity)
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
        results["position_error"][:, 0] + 3 * results["position_std_filter"][:, 0],
        "r",
    )
    ax1[0].plot(
        ins_sim.time[::f_update],
        results["position_error"][:, 0] - 3 * results["position_std_filter"][:, 0],
        "r",
    )
    ax1[0].plot(ins_sim.time[::f_update], results["position_error"][:, 0], "k")
    ax1[1].plot(
        ins_sim.time[::f_update],
        results["position_error"][:, 1] + 3 * results["position_std_filter"][:, 1],
        "r",
    )
    ax1[1].plot(
        ins_sim.time[::f_update],
        results["position_error"][:, 1] - 3 * results["position_std_filter"][:, 1],
        "r",
    )
    ax1[1].plot(ins_sim.time[::f_update], results["position_error"][:, 1], "k")
    ax1[2].plot(
        ins_sim.time[::f_update],
        results["position_error"][:, 2] + 3 * results["position_std_filter"][:, 2],
        "r",
    )
    ax1[2].plot(
        ins_sim.time[::f_update],
        results["position_error"][:, 2] - 3 * results["position_std_filter"][:, 2],
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
        results["velocity_error"][:, 0] + 3 * results["velocity_std_filter"][:, 0],
        "r",
    )
    ax2[0].plot(
        ins_sim.time[::f_update],
        results["velocity_error"][:, 0] - 3 * results["velocity_std_filter"][:, 0],
        "r",
    )
    ax2[0].plot(ins_sim.time[::f_update], results["velocity_error"][:, 0], "k")
    ax2[1].plot(
        ins_sim.time[::f_update],
        results["velocity_error"][:, 1] + 3 * results["velocity_std_filter"][:, 1],
        "r",
    )
    ax2[1].plot(
        ins_sim.time[::f_update],
        results["velocity_error"][:, 1] - 3 * results["velocity_std_filter"][:, 1],
        "r",
    )
    ax2[1].plot(ins_sim.time[::f_update], results["velocity_error"][:, 1], "k")
    ax2[2].plot(
        ins_sim.time[::f_update],
        results["velocity_error"][:, 2] + 3 * results["velocity_std_filter"][:, 2],
        "r",
    )
    ax2[2].plot(
        ins_sim.time[::f_update],
        results["velocity_error"][:, 2] - 3 * results["velocity_std_filter"][:, 2],
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
        results["attitude_error"][:, 0] + 3 * results["attitude_std_filter"][:, 0],
        "r",
    )
    ax3[0].plot(
        ins_sim.time[::f_update],
        results["attitude_error"][:, 0] - 3 * results["attitude_std_filter"][:, 0],
        "r",
    )
    ax3[0].plot(ins_sim.time[::f_update], results["attitude_error"][:, 0], "k")
    ax3[1].plot(
        ins_sim.time[::f_update],
        results["attitude_error"][:, 1] + 3 * results["attitude_std_filter"][:, 1],
        "r",
    )
    ax3[1].plot(
        ins_sim.time[::f_update],
        results["attitude_error"][:, 1] - 3 * results["attitude_std_filter"][:, 1],
        "r",
    )
    ax3[1].plot(ins_sim.time[::f_update], results["attitude_error"][:, 1], "k")
    ax3[2].plot(
        ins_sim.time[::f_update],
        results["attitude_error"][:, 2] + 3 * results["attitude_std_filter"][:, 2],
        "r",
    )
    ax3[2].plot(
        ins_sim.time[::f_update],
        results["attitude_error"][:, 2] - 3 * results["attitude_std_filter"][:, 2],
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
        results["clock_error"][:, 0] + 3 * results["clock_std_filter"][:, 0],
        "r",
    )
    ax4[0].plot(
        ins_sim.time[::f_update],
        results["clock_error"][:, 0] - 3 * results["clock_std_filter"][:, 0],
        "r",
    )
    ax4[0].plot(ins_sim.time[::f_update], results["clock_error"][:, 0], "k")
    ax4[1].plot(
        ins_sim.time[::f_update],
        results["clock_error"][:, 1] + 3 * results["clock_std_filter"][:, 1],
        "r",
    )
    ax4[1].plot(
        ins_sim.time[::f_update],
        results["clock_error"][:, 1] - 3 * results["clock_std_filter"][:, 1],
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
