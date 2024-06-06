"""
|======================================== simulate_rcvr.py ========================================|
|                                                                                                  |
|   Property of Daniel Sturdivant. Unauthorized copying of this file via any medium would be       |
|   super sad and unfortunate for me. Proprietary and confidential.                                |
|                                                                                                  |
| ------------------------------------------------------------------------------------------------ |
|                                                                                                  |
|   @file     charlizard/navigators/vector_processing.py                                           |
|   @brief    Capable of simulating vector processing, deep-coupling, and CRPA-coupling.           |
|   @author   Daniel Sturdivant <sturdivant20@gmail.com>                                           |
|   @date     June 2024                                                                            |
|                                                                                                  |
|==================================================================================================|
"""

import numpy as np
from scipy.linalg import norm
from tqdm import tqdm
from pathlib import Path

from yaml_parser import *
from log_utils import *
import navtools as nt
import navsim as ns

import charlizard.models.bpsk_correlator as bpsk
import charlizard.navigators.vector_processing as vp
from charlizard.plotting.geoplot import geoplot, Geoplot

import matplotlib.pyplot as plt

PROJECT_PATH = Path(__file__).parents[2]
CONFIG_FILE = PROJECT_PATH / "configs" / "gnss+_sim.yaml"
DATA_PATH = PROJECT_PATH / "data" / "gnss+_2024"
RESULTS_PATH = PROJECT_PATH / "results" / "gnss+_2024"
SCENARIOS = ["static", "dynamic", "static_w_crpa", "dynamic_w_crpa"]

INIT_PVA_ERROR_ENU_RPY = np.array([3.0, 3.0, 5.0, 0.1, 0.1, 0.15, 1.0, 1.0, 5.0])
# INIT_PVA_ERROR_ENU_RPY = np.zeros(9)
LIGHT_SPEED = 299792458.0  #! [m/s]
R2D = 180 / np.pi
LLA_R2D = np.array([R2D, R2D, 1])
WAVELENGTH = LIGHT_SPEED / 1575.42e6
CHIP_WIDTH = LIGHT_SPEED / 1.023e6
DISABLE_PROGRESS = False


def calc_true_observables(
    pos: np.ndarray,
    vel: np.ndarray,
    sv_pos: np.ndarray,
    sv_vel: np.ndarray,
):
    dr = pos - sv_pos
    dv = vel - sv_vel
    r = norm(dr, axis=1)
    u = dr / r[:, None]

    psr = r
    psr_dot = np.sum(u * dv, axis=1)
    return psr, psr_dot, u


def init_rcvr_states(
    scenario: str,
    j2s: float,
    yml: dict,
) -> tuple[nt.io.hdf5Slicer, vp.VPConfig, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    # load sensor models
    clock_model = ns.error_models.get_clock_allan_variance_values(yml["errors"]["rx_clock"])
    imu_model = ns.error_models.get_imu_allan_variance_values(yml["imu"]["model"])

    # initialize truth data parser
    keys = [
        "time",
        # "gps_week",
        "gps_tow",
        "sv_id",
        "sv_pos",
        "sv_vel",
        "sv_aoa",
        "sv_cn0",
        "rcvr_lla",
        "rcvr_pos",
        "rcvr_vel",
        "rcvr_att",
    ]
    truth_filename = RESULTS_PATH / "truth_data" / f"{scenario}.h5"
    h5 = nt.io.hdf5Slicer(truth_filename, keys)

    # number of time points and integration period
    N = h5.load_scalar("N")  #! number of data points
    tsim = 1.0 / yml["time"]["fsim"]
    timu = 1.0 / imu_model.f  #! update period [s]

    # load imu states to generate noise (average measurement over 'fupd' samples)
    data = h5.load_slice(np.arange(N), ["imu_acc", "imu_gyr"])
    fupd = int(yml["time"]["fsim"] / yml["time"]["fimu"])
    imu_gyr = data["imu_gyr"].T.reshape((-1, fupd)).mean(axis=1).reshape((3, -1)).T
    imu_acc = data["imu_acc"].T.reshape((-1, fupd)).mean(axis=1).reshape((3, -1)).T

    # load initial slice of truth to memory
    h5.set_keys(keys)
    data = h5.load_slice(0)
    C_n_e = nt.enu2ecefDcm(data["rcvr_lla"] / LLA_R2D)

    # generate noise on imu and clock
    gyr_noise, acc_noise = ns.error_models.compute_imu_errors(int(N / fupd), imu_model)
    clk_bias, clk_drift = ns.error_models.compute_clock_states2(N, tsim, clock_model)

    # generate first set of observables needed for vector processing
    p = data["rcvr_pos"] + np.random.randn(3) * (C_n_e @ INIT_PVA_ERROR_ENU_RPY[0:3])
    v = data["rcvr_vel"] + np.random.randn(3) * (C_n_e @ INIT_PVA_ERROR_ENU_RPY[3:6])
    a = data["rcvr_att"] + np.random.randn(3) * INIT_PVA_ERROR_ENU_RPY[6:9] / R2D
    r, r_dot, _ = calc_true_observables(p, v, data["sv_pos"], data["sv_vel"])

    # generate rcvr config
    conf = vp.VPConfig(
        T=0.02,
        order=2,
        tap_spacing=0.5,
        innovation_stdev=3.0,
        process_noise_stdev=0.5,
        cn0_buffer_len=50,
        cn0=data["sv_cn0"] - j2s,
        chip_width=CHIP_WIDTH * np.ones(data["sv_cn0"].shape),
        wavelength=WAVELENGTH * np.ones(data["sv_cn0"].shape),
        pos=p,
        vel=v,
        att=a,
        clock_bias=clk_bias[0],
        clock_drift=clk_drift[0],
        clock_type=yml["errors"]["rx_clock"],
        imu_type=yml["imu"]["model"],
        TOW=data["gps_tow"],
        meas=np.append(r, r_dot),
        ecef_ref=data["rcvr_pos"],
    )

    return (h5, conf, imu_gyr + gyr_noise, imu_acc + acc_noise, clk_bias, clk_drift, N)


def run_rcvr(
    scenario: str,
    j2s: float,
    yml: dict,
    run_idx: int = 0,
    save: bool = True,
):
    # initialize truth data and config
    h5, conf, imu_gyr, imu_acc, clk_bias, clk_drift, N = init_rcvr_states(scenario, j2s, yml)

    # update intervals
    N_imu = int(yml["time"]["fimu"] / yml["time"]["frcvr"])
    N_int = int((yml["time"]["fsim"] / yml["time"]["frcvr"]) / N_imu)
    N_main = int(N / (yml["time"]["fsim"] / yml["time"]["frcvr"]))
    dt = 1.0 / yml["time"]["fsim"]
    imu_dt = 1.0 / yml["time"]["fimu"]
    rcvr_dt = 1.0 / yml["time"]["frcvr"]

    # start receiver
    rcvr = vp.VectorProcess(conf)
    corr = bpsk.CorrelatorSim()
    corr.set_chip_width(conf.chip_width)
    corr.set_wavelength(conf.wavelength)
    corr.set_integration_period(rcvr_dt)
    corr.set_correlator_spacing(0.5)

    # initialize
    chips_nco = np.zeros((conf.chip_width.size, N_imu * N_int))
    phase_nco = np.zeros(chips_nco.shape)
    doppler_nco = np.zeros(chips_nco.shape)
    range_true = np.zeros(chips_nco.shape)
    range_rate_true = np.zeros(chips_nco.shape)
    iono_delay_true = np.zeros(chips_nco.shape)
    trop_delay_true = np.zeros(chips_nco.shape)
    results = {
        "dop": np.zeros((N_main, 6)),
        "lla": np.zeros((N_main, 3)),
        "position": np.zeros((N_main, 3)),
        "position_error": np.zeros((N_main, 3)),
        "position_std_filter": np.zeros((N_main, 3)),
        "velocity": np.zeros((N_main, 3)),
        "velocity_error": np.zeros((N_main, 3)),
        "velocity_std_filter": np.zeros((N_main, 3)),
        "attitude": np.zeros((N_main, 3)),
        "attitude_error": np.zeros((N_main, 3)),
        "attitude_std_filter": np.zeros((N_main, 3)),
        "clock_bias": np.zeros(N_main),
        "clock_drift": np.zeros(N_main),
        "clock_bias_error": np.zeros(N_main),
        "clock_drift_error": np.zeros(N_main),
        "clock_bias_std_filter": np.zeros(N_main),
        "clock_drift_std_filter": np.zeros(N_main),
    }

    # main loop
    h5_idx = 0
    imu_idx = 0
    for i in tqdm(
        range(N_main),
        desc="[\u001b[31;1mcharlizard\u001b[0m] running receiver ",
        ascii=".>#",
        bar_format="{desc}{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{rate_fmt}]",
        ncols=120,
        disable=DISABLE_PROGRESS,
    ):

        # intermediate tracking accumulation
        h5_arr_idx = np.arange(h5_idx, int(h5_idx + N_imu * N_int))
        data = h5.load_slice(h5_arr_idx)
        int_idx = 0
        for _ in range(N_imu):
            T_int = dt

            # accumulate dynamic nco parameters
            for _ in range(N_int):
                # nco
                (chips_nco[:, int_idx], doppler_nco[:, int_idx], phase_nco[:, int_idx]) = rcvr.vdfll_nco_predict(
                    T_int, data["sv_pos"][int_idx], data["sv_vel"][int_idx], imu_gyr[imu_idx, :], imu_acc[imu_idx, :]
                )

                # true
                range_true[:, int_idx], range_rate_true[:, int_idx], _ = calc_true_observables(
                    data["rcvr_pos"][int_idx, :],
                    data["rcvr_vel"][int_idx, :],
                    data["sv_pos"][int_idx, :, :],
                    data["sv_vel"][int_idx, :, :],
                )

                int_idx += 1
                T_int += dt

            # kalman filter propagation
            rcvr.propagate(imu_dt, imu_gyr[imu_idx, :], imu_acc[imu_idx, :])
            imu_idx += 1

        # call nco correction
        rcvr.vdfll_nco_correct(rcvr_dt, data["sv_pos"][-1, :, :], data["sv_vel"][-1, :, :])

        # update correlators
        corr.set_cn0(10 ** ((data["sv_cn0"].mean(axis=0) - j2s) / 10))
        rcvr_correlators = corr.simulate(
            range_true,
            range_rate_true,
            clk_bias[h5_arr_idx],
            clk_drift[h5_arr_idx],
            iono_delay_true,
            trop_delay_true,
            chips_nco,
            phase_nco,
            doppler_nco,
        )
        rcvr.update_correlators(rcvr_correlators)

        # kalman filter update
        rcvr.correct()

        # increment
        h5_idx += int(N_imu * N_int)
        # print(f"h5_idx = {h5_idx}, i = {i}")
        # print(f"h5_arr_idx = {h5_arr_idx}")

        # save data to results
        lla = data["rcvr_lla"][-1] / LLA_R2D
        C_e_enu = nt.ecef2enuDcm(lla)
        results["dop"][i, :] = rcvr.extract_dops()
        (
            results["position_std_filter"][i, :],
            results["velocity_std_filter"][i, :],
            results["attitude_std_filter"][i, :],
            results["clock_bias_std_filter"][i],
            results["clock_drift_std_filter"][i],
        ) = rcvr.extract_stds(lla)
        results["lla"][i, :] = rcvr.extract_lla() * LLA_R2D
        results["position"][i, :], results["velocity"][i, :], results["attitude"][i, :] = rcvr.extract_enu_pva(lla)
        results["clock_bias"][i], results["clock_drift"][i] = rcvr.extract_clock()
        p, v = rcvr.extract_ecef_pv()
        results["position_error"][i, :] = C_e_enu @ (p - data["rcvr_pos"][-1])
        results["velocity_error"][i, :] = C_e_enu @ (v - data["rcvr_vel"][-1])
        results["attitude_error"][i, :] = results["attitude"][i, :] - data["rcvr_att"][-1]
        results["clock_bias_error"][i] = results["clock_bias"][i] - clk_bias[h5_arr_idx[-1]]
        results["clock_drift_error"][i] = results["clock_drift"][i] - clk_drift[h5_arr_idx[-1]]

        # print(f'error = {results["position_error"][i, :]}')

    data = h5.load_slice(np.arange(0, N, yml["time"]["fsim"]), ["rcvr_lla"])

    # f, ax = plt.subplots()
    # ax.plot(data["rcvr_lla"][:, 1], data["rcvr_lla"][:, 0], color="k", linewidth=4, label="Truth")
    # ax.plot(results["lla"][:, 1], results["lla"][:, 0], color="r", linewidth=2, label="VP")
    # ax.plot(results["lla"][0, 1], results["lla"][0, 0], color="limegreen", marker="*", label="Initial Position")
    # ax.set_aspect("equal", adjustable="box")
    # ax.minorticks_on()
    # ax.grid(True, which="both")

    f, ax = geoplot(
        lon=data["rcvr_lla"][:, 1],
        lat=data["rcvr_lla"][:, 0],
        figsize=(10, 8),
        plot_init_pos=False,
        tiles="satellite",
        **{"color": "k", "s": 20, "label": "Truth"},
    )
    _, _ = geoplot(
        lon=results["lla"][::50, 1],
        lat=results["lla"][::50, 0],
        fig=f,
        ax=ax,
        plot_init_pos=True,
        tiles="satellite",
        **{"color": "r", "s": 5, "label": "VP"},
    )
    ax.legend()

    plt.show()

    if save:
        path = RESULTS_PATH / f"{scenario}" / f"{j2s}"
        nt.io.ensure_exist(path)
        dump_filename = path / f"run{run_idx}"
        # nt.io.savehdf5(dump_filename, results)  # , {"compression": "gzip", "compression_opts": 9, "shuffle": True})
        np.savez_compressed(dump_filename, **results)


if __name__ == "__main__":
    yp = YamlParser(CONFIG_FILE)
    yml = yp.Yaml2Dict()

    run_rcvr("imu_devall_drive_1000hz", 20.0, yml, 2)
