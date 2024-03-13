"""
|======================================= monte_carlo_sim.py =======================================|
|                                                                                                  |
|   Property of Daniel Sturdivant. Unauthorized copying of this file via any medium would be       |
|   super sad and unfortunate for me. Proprietary and confidential.                                |
|                                                                                                  |
| ------------------------------------------------------------------------------------------------ |
|                                                                                                  |
|   @file     scripts/monte_carlo_sim.py                                                           |
|   @brief    Run monte carlo simulation on specified navigator.                                   |
|   @author   Daniel Sturdivant <sturdivant20@gmail.com>                                           |
|   @date     March 2024                                                                           |
|                                                                                                  |
|==================================================================================================|
"""

import numpy as np
from tqdm import tqdm
from pathlib import Path

# from navsim.configuration import get_configuration
# from navsim.simulations.ins import INSSimulation
# from navsim.simulations.measurement import MeasurementSimulation
# from navtools.io.common import ensure_exist
import navtools as nt
import navsim as ns

# import simulate_gnss_ins as gnss
import simulate_soop_ins as soop

# try:
#     is_log_utils_available = True
#     from log_utils import *
# except:
#     is_log_utils_available = False
is_log_utils_available = False


NAVIGATOR = "soop_ins"
DISABLE_PROGRESS = True
N_RUNS = 1

PROJECT_PATH = Path(__file__).parents[1]
RESULTS_PATH = PROJECT_PATH / "results" / "pacific_pnt"
# SCENARIOS = ["leo", "buoy", "leo_and_buoy", "imu"]
SCENARIOS = ["leo"]


#! setup monte carlo from yaml file
def setup_mc():
    # * SIMULATE OBSERVABLES
    config = ns.get_configuration(PROJECT_PATH / "configs")

    ns.simulations.INSSimulation
    ins_sim = ns.simulations.INSSimulation(config)
    ins_sim.motion_commands(PROJECT_PATH / "data")
    ins_sim.simulate()

    f_rcvr = config.time.fsim
    f_update = int(ins_sim.imu_model.f / f_rcvr)
    meas_sim = ns.simulations.MeasurementSimulation(config, DISABLE_PROGRESS)
    meas_sim.generate_truth(
        ins_sim.ecef_position[::f_update, :], ins_sim.ecef_velocity[::f_update, :]
    )
    # meas_sim.simulate()

    return config, ins_sim, meas_sim


#! run monte carlo iterations
def monte_carlo():

    for scenario in SCENARIOS:
        config, ins_sim, meas_sim = setup_mc()
        nt.io.ensure_exist(RESULTS_PATH / scenario)

        # # * SAVE TRUTH TO FILE
        # f_update = int(ins_sim.imu_model.f / config.time.fsim)
        # truth = {
        #     "time": ins_sim.time[::f_update],
        #     "position": ins_sim.tangent_position[::f_update, :],
        #     "velocity": ins_sim.tangent_velocity[::f_update, :],
        #     "attitude": ins_sim.euler_angles[::f_update, :],
        #     "clock": np.block(
        #         [
        #             meas_sim.rx_states.clock_bias[:, None],
        #             meas_sim.rx_states.clock_drift[:, None],
        #         ]
        #     ),
        # }
        # np.savez_compressed(RESULTS_PATH / scenario / "truth", truth)

        if is_log_utils_available:
            prompt_string = default_logger.GenerateSring(
                f"[charlizard] Monte Carlo for {scenario.capitalize()} ",
                Level.Info,
                Color.Info,
            )
        else:
            prompt_string = f"[charlizard] Monte Carlo for {scenario.capitalize()} "

        is_signal_available = scenario.casefold() != "imu"

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
                    ) = soop.run_simulation(
                        config, meas_sim, ins_sim, is_signal_available, DISABLE_PROGRESS
                    )
                case "gnss_ins":
                    continue

            # * SAVE INDIVIDUAL RUNS
            results = {
                "position": pos_enu,
                "velocity": vel_enu,
                "attitude": att_rpy,
                "clock": clk,
                "position_error": pos_err_enu,
                "velocity_error": vel_err_enu,
                "attitude_error": att_err_rpy,
                "clock_error": clk_err,
                "position_std_filter": pos_var_enu,
                "velocity_std_filter": vel_var_enu,
                "attitude_std_filter": att_var_rpy,
                "clock_std_filter": clk_var,
                "dop": dop,
            }
            np.savez_compressed(RESULTS_PATH / scenario / f"run{i+1}", **results)

            # reset simulations and results
            meas_sim.clear_observables()
            results.clear()

    # * PROCESS MONTE CARLO RESULTS


#! save results from monte carlo runs


if __name__ == "__main__":
    monte_carlo()
