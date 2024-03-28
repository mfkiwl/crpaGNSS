"""
|================================= monte_carlo_sim_pacific_pnt.py =================================|
|                                                                                                  |
|   Property of Daniel Sturdivant. Unauthorized copying of this file via any medium would be       |
|   super sad and unfortunate for me. Proprietary and confidential.                                |
|                                                                                                  |
| ------------------------------------------------------------------------------------------------ |
|                                                                                                  |
|   @file     scripts/monte_carlo_sim_pacific_pnt.py                                               |
|   @brief    Run monte carlo simulation on specified navigator (ion pacific pnt 2024 results).    |
|   @author   Daniel Sturdivant <sturdivant20@gmail.com>                                           |
|   @date     March 2024                                                                           |
|                                                                                                  |
|==================================================================================================|
"""

import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
from multiprocessing import pool, cpu_count, freeze_support

import navtools as nt
import navsim as ns
import simulate_soop_ins as soop

try:
    is_log_utils_available = True
    from log_utils import *
except:
    is_log_utils_available = False
# is_log_utils_available = False


NAVIGATOR = "soop_ins"
DISABLE_PROGRESS = True
N_RUNS = 10

PROJECT_PATH = Path(__file__).parents[2]
RESULTS_PATH = PROJECT_PATH / "results" / "pacific_pnt"
SCENARIOS = ["imu"]
# SCENARIOS = ["leo", "buoy", "leo_and_buoy", "imu"]


# * map scenario to actual emitters used
def scenario_to_emitters(scenario: str):
    mapping = {
        "leo": {
            "iridium-next": ns.SignalConfiguration(signal="iridium"),
            # "orbcomm": ns.SignalConfiguration(signal="orbcomm"),
        },
        "buoy": {"buoy": ns.SignalConfiguration(signal="buoy")},
        "leo_and_buoy": {
            "iridium-next": ns.SignalConfiguration(signal="iridium"),
            # "orbcomm": ns.SignalConfiguration(signal="orbcomm"),
            "buoy": ns.SignalConfiguration(signal="buoy"),
        },
        "imu": {},
    }
    return mapping[scenario]


#! setup monte carlo from yaml file
def setup_mc(
    config: ns.SimulationConfiguration,
    ins_sim: ns.simulations.INSSimulation,
    scenario: str,
):
    # * SIMULATE OBSERVABLES
    config.constellations.emitters = scenario_to_emitters(scenario)

    f_rcvr = config.time.fsim
    f_update = int(ins_sim.imu_model.f / f_rcvr)
    meas_sim = ns.simulations.MeasurementSimulation(config, DISABLE_PROGRESS)
    meas_sim.generate_truth(ins_sim.ecef_position[::f_update, :], ins_sim.ecef_velocity[::f_update, :])

    return config, meas_sim


# #! monte carlo for loop
# def for_loop(config, meas_sim, ins_sim, is_signal_available, disable_progress, i):
#     # config, meas_sim, ins_sim, is_signal_available, disable_progress, i = x[0]

#     # re-simulate measurement noise
#     meas_sim.simulate()
#     ins_sim.add_noise()

#     match NAVIGATOR.casefold():
#         case "soop_ins":
#             results = soop.run_simulation(
#                 config, meas_sim, ins_sim, is_signal_available, disable_progress
#             )
#         case "gnss_ins":
#             pass

#     return i, results


#! run monte carlo iterations
def monte_carlo():
    # only generate path once
    config = ns.get_configuration(PROJECT_PATH / "configs")

    ins_sim = ns.simulations.INSSimulation(config)
    ins_sim.motion_commands(PROJECT_PATH / "data")
    ins_sim.simulate()

    for scenario in SCENARIOS:
        # update scenario and regenerate observables for correct emitters
        if scenario.casefold() != "imu":
            config, meas_sim = setup_mc(config, ins_sim, scenario)
            is_signal_available = True
        else:
            config, meas_sim = setup_mc(config, ins_sim, "leo")
            is_signal_available = False

        # ensure correct output directory exists
        nt.io.ensure_exist(RESULTS_PATH / scenario)

        # generate tqdm description with a timestamp if available
        if is_log_utils_available:
            prompt_string = default_logger.GenerateSring(
                f"[charlizard] Monte Carlo for {scenario.upper()} ",
                Level.Info,
                Color.Info,
            )
        else:
            prompt_string = f"[charlizard] Monte Carlo for {scenario.upper()} "

        # # TODO: figure out why picking 'Satrec' object does not work
        # with pool.Pool(processes=cpu_count()) as p:
        #     args = [
        #         (config, meas_sim, ins_sim, is_signal_available, DISABLE_PROGRESS, i)
        #         for i in range(N_RUNS)
        #     ]
        #     for i, results in tqdm(
        #         p.starmap(for_loop, args),
        #         total=N_RUNS,
        #         desc=prompt_string,
        #         ascii=".>#",
        #         bar_format="{desc}{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{rate_fmt}]",
        #         ncols=120,
        #     ):
        #         np.savez_compressed(RESULTS_PATH / scenario / f"run{i+1}", **results)

        # * MONTE CARLO SIMULATIONS
        for i in tqdm(
            range(N_RUNS),
            desc=prompt_string,
            ascii=".>#",
            bar_format="{desc}{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{rate_fmt}]",
            ncols=120,
        ):
            # re-simulate measurement noise
            meas_sim.simulate()
            ins_sim.add_noise()

            match NAVIGATOR.casefold():
                case "soop_ins":
                    results = soop.run_simulation(config, meas_sim, ins_sim, is_signal_available, DISABLE_PROGRESS)
                case "gnss_ins":
                    continue

            # * SAVE INDIVIDUAL RUNS
            np.savez_compressed(RESULTS_PATH / scenario / f"run{i+1}", **results)

            # reset simulations and results
            meas_sim.clear_observables()
            # ins_sim.clear_noise() ???????????????????
            results.clear()

    # * PROCESS MONTE CARLO RESULTS
    f = int(ins_sim.imu_model.f / config.time.fsim)
    process_mc_results(time=ins_sim.time[::f])


#! save results from monte carlo runs
def process_mc_results(time: np.ndarray):
    if is_log_utils_available:
        default_logger.Info("[charlizard] compiling monte carlo results!")
    else:
        print("[charlizard] compiling monte carlo results!")
    for scenario in SCENARIOS:
        for i, file in enumerate(os.listdir(RESULTS_PATH / scenario)):
            if i >= N_RUNS:
                break

            if "run" not in file:
                continue

            # load saved data
            data = np.load(RESULTS_PATH / scenario / file)

            # initialize monte carlo results
            if i == 0:
                n = time.size
                results = {
                    "time": time,
                    "position_mean": np.zeros((n, 3)),
                    "velocity_mean": np.zeros((n, 3)),
                    "attitude_mean": np.zeros((n, 3)),
                    "clock_mean": np.zeros((n, 2)),
                    "position_error_mean": np.zeros((n, 3)),
                    "velocity_error_mean": np.zeros((n, 3)),
                    "attitude_error_mean": np.zeros((n, 3)),
                    "clock_error_mean": np.zeros((n, 2)),
                    "position_rmse": np.zeros((n, 3)),
                    "velocity_rmse": np.zeros((n, 3)),
                    "attitude_rmse": np.zeros((n, 3)),
                    "clock_rmse": np.zeros((n, 2)),
                    "position_filter_std": np.zeros((n, 3)),
                    "velocity_filter_std": np.zeros((n, 3)),
                    "attitude_filter_std": np.zeros((n, 3)),
                    "clock_filter_std": np.zeros((n, 2)),
                    "position_mc_std": np.zeros((n, 3)),
                    "velocity_mc_std": np.zeros((n, 3)),
                    "attitude_mc_std": np.zeros((n, 3)),
                    "clock_mc_std": np.zeros((n, 2)),
                    "dop": np.zeros((n, 6)),
                }

            # add state to mean absolute difference (mae)
            results["position_mean"] += data["position"][:n, :]
            results["velocity_mean"] += data["velocity"][:n, :]
            results["attitude_mean"] += data["attitude"][:n, :]
            results["clock_mean"] += data["clock"][:n, :]

            # attitude error need to be restricted to specific range
            data["attitude_error"][data["attitude_error"] > 360] -= 360
            data["attitude_error"][data["attitude_error"] < -360] += 360

            # add error state to mean absolute difference (mae)
            results["position_error_mean"] += data["position_error"][:n, :]
            results["velocity_error_mean"] += data["velocity_error"][:n, :]
            results["attitude_error_mean"] += data["attitude_error"][:n, :]
            results["clock_error_mean"] += data["clock_error"][:n, :]

            # add error state to root-mean-squared-error (rmse)
            results["position_rmse"] += data["position_error"][:n, :] ** 2
            results["velocity_rmse"] += data["velocity_error"][:n, :] ** 2
            results["attitude_rmse"] += data["attitude_error"][:n, :] ** 2
            results["clock_rmse"] += data["clock_error"][:n, :] ** 2

            # add filter standard deviations to mean absolute difference (mae)
            results["position_filter_std"] += data["position_std_filter"][:n, :]
            results["velocity_filter_std"] += data["velocity_std_filter"][:n, :]
            results["attitude_filter_std"] += data["attitude_std_filter"][:n, :]
            results["clock_filter_std"] += data["clock_std_filter"][:n, :]

            # add filter dop values (includes number of emitters)
            results["dop"] += data["dop"][:n, :]

        # calculate monte carlo standard deviation of errors
        results["position_mc_std"] = np.sqrt(
            results["position_rmse"] - (results["position_error_mean"] ** 2 / N_RUNS) / N_RUNS
        )
        results["velocity_mc_std"] = np.sqrt(
            results["velocity_rmse"] - (results["velocity_error_mean"] ** 2 / N_RUNS) / N_RUNS
        )
        results["attitude_mc_std"] = np.sqrt(
            results["attitude_rmse"] - (results["attitude_error_mean"] ** 2 / N_RUNS) / N_RUNS
        )
        results["clock_mc_std"] = np.sqrt(results["clock_rmse"] - (results["clock_error_mean"] ** 2 / N_RUNS) / N_RUNS)

        # calculate mean of states
        results["position_mean"] /= N_RUNS
        results["velocity_mean"] /= N_RUNS
        results["attitude_mean"] /= N_RUNS
        results["clock_mean"] /= N_RUNS

        # calculate mean of errors
        results["position_error_mean"] /= N_RUNS
        results["velocity_error_mean"] /= N_RUNS
        results["attitude_error_mean"] /= N_RUNS
        results["clock_error_mean"] /= N_RUNS

        # calculate mean of filter std
        results["position_filter_std"] = np.sqrt(results["position_filter_std"] / N_RUNS)
        results["velocity_filter_std"] = np.sqrt(results["velocity_filter_std"] / N_RUNS)
        results["attitude_filter_std"] = np.sqrt(results["attitude_filter_std"] / N_RUNS)
        results["clock_filter_std"] = np.sqrt(results["clock_filter_std"] / N_RUNS)

        # calculate rmse of errors
        results["position_rmse"] = np.sqrt(results["position_rmse"] / N_RUNS)
        results["velocity_rmse"] = np.sqrt(results["velocity_rmse"] / N_RUNS)
        results["attitude_rmse"] = np.sqrt(results["attitude_rmse"] / N_RUNS)
        results["clock_rmse"] = np.sqrt(results["clock_rmse"] / N_RUNS)

        # calculate mean dop values
        results["dop"] /= N_RUNS

        np.savez_compressed(RESULTS_PATH / scenario / "mc_results", **results)
        results.clear()


if __name__ == "__main__":
    # freeze_support()
    # monte_carlo()
    process_mc_results(np.arange(210))
