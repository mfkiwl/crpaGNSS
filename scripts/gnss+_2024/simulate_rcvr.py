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
import navtools as nt
import navsim as ns
from scipy.linalg import norm
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

import charlizard.models.bpsk_correlator as bpsk
import charlizard.models.crpa_gain_pattern as gain
import charlizard.navigators.crpa_imu_gnss as cig
from charlizard.plotting.geoplot import geoplot, Geoplot


PROJECT_PATH = Path(__file__).parents[2]
CONFIG_FILE = PROJECT_PATH / "configs" / "gnss+_sim.yaml"
DATA_PATH = PROJECT_PATH / "data" / "gnss+_2024"
RESULTS_PATH = PROJECT_PATH / "results" / "gnss+_2024"

LIGHT_SPEED = 299792458.0  #! [m/s]
R2D = 180 / np.pi
LLA_R2D = np.array([R2D, R2D, 1])
WAVELENGTH = LIGHT_SPEED / 1575.42e6
CHIP_WIDTH = LIGHT_SPEED / 1.023e6
DISABLE_PROGRESS = False


#! === calc_true_observables ===
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


#! === generate_antenna_array ===
def generate_antenna_array(n_ant: int) -> np.ndarray:
    if n_ant == 1:
        Z = np.zeros(3)
    elif n_ant == 2:  # 2 ELEMENT
        Z = 0.5 * np.array([[0, 0, 0], [WAVELENGTH, 0, 0]])
    elif n_ant == 3:  # 3 ELEMENT
        Z = 0.5 * np.array([[0, 0, 0], [WAVELENGTH, 0, 0], [WAVELENGTH / 2, -np.sqrt(0.75 * WAVELENGTH**2), 0]])
    elif n_ant == 4:  # 4 ELEMENT
        Z = 0.5 * np.array([[0, 0, 0], [WAVELENGTH, 0, 0], [0, -WAVELENGTH, 0], [WAVELENGTH, -WAVELENGTH, 0]])
    return Z


#! === generate_imu_noise ===
def generate_imu_noise(
    model: str, f_sim: int, f_imu: int, true_gyr: np.ndarray, true_acc: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray]:
    # average imu measurement to actual simulation frequency
    f_upd = int(f_sim / f_imu)
    meas_gyr = true_gyr.T.reshape((-1, f_upd)).mean(axis=1).reshape((3, -1)).T
    meas_acc = true_acc.T.reshape((-1, f_upd)).mean(axis=1).reshape((3, -1)).T

    # misalignment, bias, awgn+markov
    imu = ns.error_models.get_imu_allan_variance_values(model)
    M_gyr, M_acc = nt.euler2dcm(np.random.randn(3) * 1.0 / R2D).T, nt.euler2dcm(np.random.randn(3) * 1.0 / R2D).T
    b_gyr, b_acc = np.random.randn(3) * 1.0 / R2D, np.random.randn(3) * 0.25
    n_gyr, n_acc = ns.error_models.compute_imu_errors(n, imu)

    return (meas_gyr @ M_gyr + b_gyr + n_gyr, meas_acc @ M_acc + b_acc + n_acc)


#! === generate_clock_noise ===
def generate_clock_noise(model: str, f_sim: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    clk = ns.error_models.get_clock_allan_variance_values(model)
    clk_bias, clk_drift = ns.error_models.compute_clock_states2(n, 1.0 / f_sim, clk)
    return (clk_bias, clk_drift)


#! === generate_rcvr_config ===
def initialize_rcvr(
    scenario: str,
    imu_model: str,
    clk_model: str,
    n_ant: int,
    attenuation: float,
    init_pva_err: np.ndarray,
    f_sim: int,
    f_imu: int,
    f_rcvr: int,
) -> tuple[cig.DIConfig, dict, nt.io.hdf5Slicer, cig.CRPA_IMU_GNSS, bpsk.CorrelatorSim, gain.CRPAGainPattern]:
    # load scenario data
    truth_filename = RESULTS_PATH / "truth_data" / f"{scenario}.h5"
    keys = ["time", "sv_id", "sv_pos", "sv_vel", "sv_aoa", "sv_cn0", "rcvr_lla", "rcvr_pos", "rcvr_vel", "rcvr_att"]
    h5 = nt.io.hdf5Slicer(truth_filename, keys)
    n = int(h5.load_scalar("N"))  # number of data points
    data = h5.load_slice(np.arange(n), ["imu_acc", "imu_gyr"])

    # generate measurements
    meas = {}
    ant_body_pos = generate_antenna_array(n_ant)
    meas["gyr"], meas["acc"] = generate_imu_noise(
        imu_model, f_sim, f_imu, data["imu_gyr"], data["imu_acc"], int(n / (f_sim / f_imu))
    )
    meas["bias"], meas["drift"] = generate_clock_noise(clk_model, f_sim, n)

    # initialize receiver
    data = h5.load_slice(0, keys)
    lla0 = data["rcvr_lla"] / LLA_R2D
    pos = nt.enu2lla(np.random.randn(3) * init_pva_err[0:3], lla0) * LLA_R2D
    vel = nt.ecef2enuv(data["rcvr_vel"], lla0) + np.random.randn(3) * init_pva_err[3:6]
    att = data["rcvr_att"] + np.random.randn(3) * init_pva_err[6:9]
    cn0 = (data["sv_cn0"] - attenuation) if n_ant == 1 else (data["sv_cn0"] - attenuation + 10 * np.log10(n_ant))

    # create config
    n_sv = cn0.size
    conf = cig.DIConfig(
        T=1.0 / f_rcvr,
        innovation_stdev=3.0,
        cn0_buffer_len=50,
        cn0=cn0,
        tap_spacing=0.5,
        chip_width=CHIP_WIDTH * np.ones(n_sv),
        wavelength=WAVELENGTH * np.ones(n_sv),
        pos=pos,
        vel=vel,
        att=att,
        clock_bias=meas["bias"][0],  # + 3.0 * np.random.randn(),
        clock_drift=meas["drift"][0],  # + 0.1 * np.random.randn(),
        clock_type=clk_model,
        imu_type=imu_model,
        TOW=0.0,
        frame="enu",
        ant_body_pos=ant_body_pos,
        mode=0 if n_ant == 1 else 1,  # 0: DI, 1: DI + CRPA, 2: VP, 3: VP + CRPA
    )

    # start rcvr
    rcvr = cig.CRPA_IMU_GNSS(conf)
    corr = bpsk.CorrelatorSim(conf.wavelength, conf.chip_width, conf.tap_spacing, conf.T)
    crpa = gain.CRPAGainPattern(conf.ant_body_pos, conf.wavelength)

    return (conf, meas, h5, rcvr, corr, crpa, lla0, n)


#! === run_rcvr ===
def run_rcvr(args):
    params, run_idx, save, return_lla, disable_progress, path = args

    # initialize truth data and receiver
    (conf, meas, h5, rcvr, corr, crpa, lla0, n) = initialize_rcvr(
        params["scenario"],
        params["imu_model"],
        params["clock_model"],
        params["n_ant"],
        params["attenuation"],
        params["init_pva_err"],
        params["f_sim"],
        params["f_imu"],
        params["f_rcvr"],
    )

    # data sizes
    n_ant = params["n_ant"]
    n_sv = conf.cn0.size

    # accumulation/update intervals
    N_imu = int(params["f_imu"] / params["f_rcvr"])
    N_int = int((params["f_sim"] / params["f_rcvr"]) / N_imu)
    N_main = int(n / (params["f_sim"] / params["f_rcvr"]))
    dt = 1.0 / params["f_sim"]
    imu_dt = 1.0 / params["f_imu"]
    rcvr_dt = 1.0 / params["f_rcvr"]

    # correlator model inputs
    range_nco = np.zeros((n_sv, N_imu * N_int))
    range_rate_nco = np.zeros(range_nco.shape)
    range_true = np.zeros((n_sv, N_imu * N_int, n_ant))
    range_rate_true = np.zeros(range_true.shape)
    iono_delay_true = np.zeros(range_true.shape)
    trop_delay_true = np.zeros(range_true.shape)
    gain_true = np.ones((n_sv, N_imu * N_int))

    # init results
    results = {
        "dop": np.zeros((N_main, 6)),
        "cn0": np.zeros((N_main, n_sv)),
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

    # ----- main loop ------------------------------------------------------------------------------
    h5_idx = 0
    imu_idx = 0
    for i in tqdm(
        range(N_main),
        desc="[\u001b[31;1mcharlizard\u001b[0m] running receiver ",
        ascii=".>#",
        bar_format="{desc}{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{rate_fmt}]",
        ncols=120,
        disable=disable_progress,
    ):

        # ----- intermediate tracking accumulation -------------------------------------------------
        h5_arr_idx = np.arange(h5_idx, int(h5_idx + N_imu * N_int))
        data = h5.load_slice(h5_arr_idx)
        int_idx = 0
        for j in range(N_imu):
            T_int = dt

            # accumulate dynamic nco parameters
            for k in range(N_int):
                # nco
                range_nco[:, int_idx], range_rate_nco[:, int_idx], az, el = rcvr.propagate(
                    data["sv_pos"][int_idx],
                    data["sv_vel"][int_idx],
                    meas["acc"][imu_idx, :],
                    meas["gyr"][imu_idx, :],
                    T_int,
                    False,
                )
                if j == 0 and k == 0:
                    crpa.set_expected_doa(az, el)

                # true observables
                for k in range(n_ant):
                    ant_pos = (
                        nt.enu2ecefDcm(data["rcvr_lla"][int_idx] / LLA_R2D)
                        @ nt.euler2dcm(data["rcvr_att"][int_idx] / R2D, "enu")
                        @ conf.ant_body_pos[k, :]
                        + data["rcvr_pos"][int_idx]
                    )
                    range_true[:, int_idx, k], range_rate_true[:, int_idx, k], _ = calc_true_observables(
                        ant_pos,
                        data["rcvr_vel"][int_idx],
                        data["sv_pos"][int_idx],
                        data["sv_vel"][int_idx],
                    )
                    if k == 0 and conf.mode == 1:
                        az, el, _ = nt.ecef2aer2d(data["sv_pos"][int_idx], data["rcvr_pos"][int_idx])
                        gain_true[:, int_idx] = crpa.calc_beamstear_gain(az, el)

                int_idx += 1
                T_int += dt

            # kalman filter propagation (nco)
            _, _, _, _ = rcvr.propagate(
                data["sv_pos"][int_idx - 1],
                data["sv_vel"][int_idx - 1],
                meas["acc"][imu_idx, :],
                meas["gyr"][imu_idx, :],
                imu_dt,
                True,
            )
            imu_idx += 1

        # ----- update filter ----------------------------------------------------------------------
        # update correlators
        rcvr.update_correlators(
            corr.simulate(
                range_true,
                range_rate_true,
                meas["bias"][h5_arr_idx],
                meas["drift"][h5_arr_idx],
                iono_delay_true,
                trop_delay_true,
                10.0 ** ((data["sv_cn0"].T - params["attenuation"]) / 10.0) * gain_true,
                range_nco,
                range_rate_nco,
            )
        )

        # kalman filter update
        rcvr.update()

        # ----- increment --------------------------------------------------------------------------
        h5_idx += int(N_imu * N_int)

        # save data to results
        lla_r = data["rcvr_lla"][-1, :] / LLA_R2D
        (
            results["lla"][i, :],
            results["velocity"][i, :],
            results["attitude"][i, :],
            results["clock_bias"][i],
            results["clock_drift"][i],
        ) = rcvr.extract_states()
        (
            results["position_std_filter"][i, :],
            results["velocity_std_filter"][i, :],
            results["attitude_std_filter"][i, :],
            results["clock_bias_std_filter"][i],
            results["clock_drift_std_filter"][i],
        ) = rcvr.extract_stds()
        results["dop"][i, :] = rcvr.extract_dops()
        results["position"][i, :] = nt.lla2enu(results["lla"][i, :] / LLA_R2D, lla0)
        results["position_error"][i, :] = nt.lla2enu(results["lla"][i, :] / LLA_R2D, lla_r)
        results["velocity_error"][i, :] = results["velocity"][i, :] - nt.ecef2enuv(data["rcvr_vel"][-1, :], lla_r)
        results["attitude_error"][i, :] = results["attitude"][i, :] - data["rcvr_att"][-1, :]
        results["clock_bias_error"][i] = results["clock_bias"][i] - meas["bias"][h5_arr_idx[-1]]
        results["clock_drift_error"][i] = results["clock_drift"][i] - meas["drift"][h5_arr_idx[-1]]
        results["cn0"][i, :] = rcvr.extract_cn0()

    # ----- done! ----------------------------------------------------------------------------------
    if save:
        nt.io.ensure_exist(path)
        dump_filename = path / f"run{run_idx}"
        np.savez_compressed(dump_filename, **results)

    if return_lla:
        data = h5.load_slice(np.arange(0, n, 1), ["rcvr_lla"])
        return results, data
    else:
        return True


#! -------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    params = {
        "scenario": "imu_devall_drive_1000hz",
        "imu_model": "consumer",
        "clock_model": "high_quality_tcxo",
        "n_ant": 4,
        "attenuation": 0.0,
        "init_pva_err": [3.0, 3.0, 5.0, 0.1, 0.1, 0.15, 3.0, 3.0, 10.0],
        "f_sim": 1000,
        "f_imu": 100,
        "f_rcvr": 50,
    }
    path = RESULTS_PATH / f"{params['scenario']}" / f"{params['attenuation']}"
    args = (params, 0, False, True, False, path)
    results, data = run_rcvr(args)
    print("done!")

    # fmt: off
    t = np.arange(results["lla"].shape[0]) / 50.0

    # * position, velocity, bias, drift plot results
    h1, ax1 = plt.subplots(nrows=3, ncols=1)
    h1.suptitle("ENU Position Error")
    ax1[0].plot(t, results["position_error"][:, 0] + 3 * results["position_std_filter"][:, 0], "r")
    ax1[0].plot(t, results["position_error"][:, 0] - 3 * results["position_std_filter"][:, 0], "r")
    ax1[0].plot(t, results["position_error"][:, 0], "k")
    ax1[1].plot(t, results["position_error"][:, 1] + 3 * results["position_std_filter"][:, 1], "r")
    ax1[1].plot(t, results["position_error"][:, 1] - 3 * results["position_std_filter"][:, 1], "r")
    ax1[1].plot(t, results["position_error"][:, 1], "k")
    ax1[2].plot(t, results["position_error"][:, 2] + 3 * results["position_std_filter"][:, 2], "r")
    ax1[2].plot(t, results["position_error"][:, 2] - 3 * results["position_std_filter"][:, 2], "r")
    ax1[2].plot(t, results["position_error"][:, 2], "k")
    ax1[0].set_ylabel("East [m]")
    ax1[1].set_ylabel("North [m]")
    ax1[2].set_ylabel("Up [m]")
    ax1[2].set_xlabel("Time [s]")
    ax1[0].minorticks_on()
    ax1[0].grid(True, which="both")
    ax1[1].minorticks_on()
    ax1[1].grid(True, which="both")
    ax1[2].minorticks_on()
    ax1[2].grid(True, which="both")

    h2, ax2 = plt.subplots(nrows=3, ncols=1)
    h2.suptitle("ENU Velocity Error")
    ax2[0].plot(t, results["velocity_error"][:, 0] + 3 * results["velocity_std_filter"][:, 0], "r")
    ax2[0].plot(t, results["velocity_error"][:, 0] - 3 * results["velocity_std_filter"][:, 0], "r")
    ax2[0].plot(t, results["velocity_error"][:, 0], "k")
    ax2[1].plot(t, results["velocity_error"][:, 1] + 3 * results["velocity_std_filter"][:, 1], "r")
    ax2[1].plot(t, results["velocity_error"][:, 1] - 3 * results["velocity_std_filter"][:, 1], "r")
    ax2[1].plot(t, results["velocity_error"][:, 1], "k")
    ax2[2].plot(t, results["velocity_error"][:, 2] + 3 * results["velocity_std_filter"][:, 2], "r")
    ax2[2].plot(t, results["velocity_error"][:, 2] - 3 * results["velocity_std_filter"][:, 2], "r")
    ax2[2].plot(t, results["velocity_error"][:, 2], "k")
    ax2[0].set_ylabel("East [m/s]")
    ax2[1].set_ylabel("North [m/s]")
    ax2[2].set_ylabel("Up [m/s]")
    ax2[2].set_xlabel("Time [s]")
    ax2[0].minorticks_on()
    ax2[0].grid(True, which="both")
    ax2[1].minorticks_on()
    ax2[1].grid(True, which="both")
    ax2[2].minorticks_on()
    ax2[2].grid(True, which="both")

    results["attitude_error"][results["attitude_error"] > 180] -= 180
    results["attitude_error"][results["attitude_error"] < -180] += 180
    results["attitude_error"][results["attitude_error"] > 90] -= 180
    results["attitude_error"][results["attitude_error"] < -90] += 180
    h3, ax3 = plt.subplots(nrows=3, ncols=1)
    h3.suptitle("RPY Attitude Error")
    ax3[0].plot(t, results["attitude_error"][:, 0] + 3 * results["attitude_std_filter"][:, 0], "r")
    ax3[0].plot(t, results["attitude_error"][:, 0] - 3 * results["attitude_std_filter"][:, 0], "r")
    ax3[0].plot(t, results["attitude_error"][:, 0], "k")
    ax3[1].plot(t, results["attitude_error"][:, 1] + 3 * results["attitude_std_filter"][:, 1], "r")
    ax3[1].plot(t, results["attitude_error"][:, 1] - 3 * results["attitude_std_filter"][:, 1], "r")
    ax3[1].plot(t, results["attitude_error"][:, 1], "k")
    ax3[2].plot(t, results["attitude_error"][:, 2] + 3 * results["attitude_std_filter"][:, 2], "r")
    ax3[2].plot(t, results["attitude_error"][:, 2] - 3 * results["attitude_std_filter"][:, 2], "r")
    ax3[2].plot(t, results["attitude_error"][:, 2], "k")
    ax3[0].set_ylabel("Roll [deg]")
    ax3[1].set_ylabel("Pitch [deg]")
    ax3[2].set_ylabel("Yaw [deg]")
    ax3[2].set_xlabel("Time [s]")
    ax3[0].minorticks_on()
    ax3[0].grid(True, which="both")
    ax3[1].minorticks_on()
    ax3[1].grid(True, which="both")
    ax3[2].minorticks_on()
    ax3[2].grid(True, which="both")

    h4, ax4 = plt.subplots(nrows=2, ncols=1)
    h4.suptitle("Clock Errors")
    ax4[0].plot(t, results["clock_bias_error"] + 3 * results["clock_bias_std_filter"], "r")
    ax4[0].plot(t, results["clock_bias_error"] - 3 * results["clock_bias_std_filter"], "r")
    ax4[0].plot(t, results["clock_bias_error"], "k")
    ax4[1].plot(t, results["clock_drift_error"] + 3 * results["clock_drift_std_filter"], "r")
    ax4[1].plot(t, results["clock_drift_error"] - 3 * results["clock_drift_std_filter"], "r")
    ax4[1].plot(t, results["clock_drift_error"], "k")
    ax4[0].set_ylabel("Bias [m]")
    ax4[1].set_ylabel("Drift [m/s]")
    ax4[1].set_xlabel("Time [s]")
    ax4[0].minorticks_on()
    ax4[0].grid(True, which="both")
    ax4[1].minorticks_on()
    ax4[1].grid(True, which="both")

    # * 2d position plot
    h5, ax5 = plt.subplots()
    h5.suptitle("2D ENU Position")
    ax5.plot(data["rcvr_lla"][:, 1], data["rcvr_lla"][:, 0], color="b", linewidth=5, label="Truth")
    ax5.plot(results["lla"][:, 1], results["lla"][:, 0], color="r", linewidth=2, label="VP")
    ax5.plot(results["lla"][0, 1], results["lla"][0, 0], color="limegreen", marker="*", markersize=15, label="Initial Position")
    ax5.set_aspect("equal", adjustable="box")
    ax5.set_ylabel("North [deg]")
    ax5.set_xlabel("East [deg]")
    ax5.axis("square")
    ax5.grid()
    ax5.legend()

    # * dops
    h6, ax6 = plt.subplots()
    h6.suptitle("Dilution of Precision")
    ax6.plot(t, results["dop"])
    ax6.set_ylabel("DOP")
    ax6.set_xlabel("Time [s]")
    ax6.legend(["GDOP", "PDOP", "HDOP", "VDOP", "TDOP", "# Emitters"])

    # * cn0
    h7, ax7 = plt.subplots()
    h7.suptitle("Carrier to Noise Density Ratio")
    ax7.plot(t, results["cn0"])
    ax7.set_ylabel("dB-Hz")
    ax7.set_xlabel("Time [s]")
    ax7.legend()

    # #* phase discriminator
    # h8, ax8 = plt.subplots()
    # # results["phase_disc"][i, n_sv, n_ant]
    # h8.suptitle("Phase Discriminator")
    # ax8.plot(t, results["phase_disc"][:,:,0])
    # ax8.set_ylabel("Cycles")
    # ax8.set_xlabel("Time [s]")

    #* geoplot
    h9, ax9 = geoplot(
        lon=data["rcvr_lla"][:, 1],
        lat=data["rcvr_lla"][:, 0],
        figsize=(10, 8),
        plot_init_pos=False,
        tiles="satellite",
        **{"color": "k", "s": 30, "label": "Truth"},
    )
    _, _ = geoplot(
        lon=results["lla"][:, 1],
        lat=results["lla"][:, 0],
        fig=h9,
        ax=ax9,
        plot_init_pos=True,
        tiles="satellite",
        **{"color": "r", "s": 5, "label": "VP"},
    )
    ax9.legend()

    plt.show()
    # fmt: on
