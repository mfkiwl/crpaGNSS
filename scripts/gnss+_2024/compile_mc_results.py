import os
import numpy as np
import navtools as nt
from tqdm import tqdm
from multiprocessing import pool, cpu_count, freeze_support
from logutils import tic, toc
from yamlparser import YamlParser

from simulate_rcvr import CONFIG_FILE, RESULTS_PATH, CHIP_WIDTH, LLA_R2D


#! add DOP and CN0 ?????
def process_result(args):
    dump_file, file_path, sv_pos, rng = args
    for i, file in enumerate(os.listdir(file_path)):
        if i >= conf["mc_params"]["n_runs"]:
            break
        if "run" not in file:
            continue

        # load saved data
        data = np.load(file_path / file)

        if i == 0:
            # initialize results (at 1 Hz not 50 Hz)
            n = data["clock_bias"][::50].size
            time = (np.arange(n),)
            pos = np.zeros((n, 3, conf["mc_params"]["n_runs"]))
            vel = np.zeros((n, 3, conf["mc_params"]["n_runs"]))
            att = np.zeros((n, 3, conf["mc_params"]["n_runs"]))
            clk_b = np.zeros((n, conf["mc_params"]["n_runs"]))
            clk_d = np.zeros((n, conf["mc_params"]["n_runs"]))
            pos_err = np.zeros((n, 3, conf["mc_params"]["n_runs"]))
            vel_err = np.zeros((n, 3, conf["mc_params"]["n_runs"]))
            att_err = np.zeros((n, 3, conf["mc_params"]["n_runs"]))
            clk_b_err = np.zeros((n, conf["mc_params"]["n_runs"]))
            clk_d_err = np.zeros((n, conf["mc_params"]["n_runs"]))
            pos_std_filt = np.zeros((n, 3, conf["mc_params"]["n_runs"]))
            vel_std_filt = np.zeros((n, 3, conf["mc_params"]["n_runs"]))
            att_std_filt = np.zeros((n, 3, conf["mc_params"]["n_runs"]))
            clk_b_std_filt = np.zeros((n, conf["mc_params"]["n_runs"]))
            clk_d_std_filt = np.zeros((n, conf["mc_params"]["n_runs"]))
            chip_err = np.zeros((sv_pos.shape[0], conf["mc_params"]["n_runs"]))

            pos[:, :, 0] = data["position"][::50, :]
            vel[:, :, 0] = data["velocity"][::50, :]
            att[:, :, 0] = data["attitude"][::50, :]
            clk_b[:, 0] = data["clock_bias"][::50]
            clk_d[:, 0] = data["clock_drift"][::50]
            pos_err[:, :, 0] = data["position_error"][::50, :]
            vel_err[:, :, 0] = data["velocity_error"][::50, :]
            att_err[:, :, 0] = data["attitude_error"][::50, :]
            clk_b_err[:, 0] = data["clock_bias_error"][::50]
            clk_d_err[:, 0] = data["clock_drift_error"][::50]
            pos_std_filt[:, :, 0] = data["position_std_filter"][::50, :]
            vel_std_filt[:, :, 0] = data["velocity_std_filter"][::50, :]
            att_std_filt[:, :, 0] = data["attitude_std_filter"][::50, :]
            clk_b_std_filt[:, 0] = data["clock_bias_std_filter"][::50]
            clk_d_std_filt[:, 0] = data["clock_drift_std_filter"][::50]
            chip_err[:, 0] = np.linalg.norm(sv_pos - nt.lla2ecef(data["lla"][-1, :] / LLA_R2D)[None, :], axis=1) - rng

        else:
            pos[:, :, i] = data["position"][::50, :]
            vel[:, :, i] = data["velocity"][::50, :]
            att[:, :, i] = data["attitude"][::50, :]
            clk_b[:, i] = data["clock_bias"][::50]
            clk_d[:, i] = data["clock_drift"][::50]
            pos_err[:, :, i] = data["position_error"][::50, :]
            vel_err[:, :, i] = data["velocity_error"][::50, :]
            att_err[:, :, i] = data["attitude_error"][::50, :]
            clk_b_err[:, i] = data["clock_bias_error"][::50]
            clk_d_err[:, i] = data["clock_drift_error"][::50]
            pos_std_filt[:, :, i] = data["position_std_filter"][::50, :]
            vel_std_filt[:, :, i] = data["velocity_std_filter"][::50, :]
            att_std_filt[:, :, i] = data["attitude_std_filter"][::50, :]
            clk_b_std_filt[:, i] = data["clock_bias_std_filter"][::50]
            clk_d_std_filt[:, i] = data["clock_drift_std_filter"][::50]
            chip_err[:, i] = np.linalg.norm(sv_pos - nt.lla2ecef(data["lla"][-1, :] / LLA_R2D)[None, :], axis=1) - rng

    att_err[att_err > 180] -= 180
    att_err[att_err < -180] += 180
    att_err[att_err > 90] -= 180
    att_err[att_err < -90] += 180
    results = {
        "time": time,
        "position_mean": pos.mean(axis=2),
        "velocity_mean": vel.mean(axis=2),
        "attitude_mean": att.mean(axis=2),
        "position_error_mean": pos_err.mean(axis=2),
        "velocity_error_mean": vel_err.mean(axis=2),
        "attitude_error_mean": att_err.mean(axis=2),
        "position_rmse": np.sqrt((pos_err**2).mean(axis=2)),
        "velocity_rmse": np.sqrt((vel_err**2).mean(axis=2)),
        "attitude_rmse": np.sqrt((att_err**2).mean(axis=2)),
        "final_position_rmse": np.sqrt((pos_err**2).mean()),
        "final_velocity_rmse": np.sqrt((vel_err**2).mean()),
        "final_attitude_rmse": np.sqrt((att_err**2).mean()),
        "position_filter_std": pos_std_filt.mean(axis=2),
        "velocity_filter_std": vel_std_filt.mean(axis=2),
        "attitude_filter_std": att_std_filt.mean(axis=2),
        "position_mc_std": pos_err.std(axis=2),
        "velocity_mc_std": vel_err.std(axis=2),
        "attitude_mc_std": att_err.std(axis=2),
        "prob_tracking": (np.abs(chip_err) < (0.5 * CHIP_WIDTH)).sum() / chip_err.size,
    }
    np.savez_compressed(dump_file, **results)
    results.clear()


if __name__ == "__main__":
    freeze_support()
    t0 = tic(f"[\u001b[31;1mcharlizard\u001b[0m] Compiling GNSS+ Monte Carlo results ...")

    # parse yaml config file
    yp = YamlParser(CONFIG_FILE)
    conf = yp.Yaml2Dict()
    nt.io.ensure_exist(RESULTS_PATH / "monte_carlo")
    h5 = nt.io.hdf5Slicer(RESULTS_PATH / "truth_data" / "static.h5", ["sv_pos", "rcvr_pos"])
    data = h5.load_slice(599999)
    sv_pos = data["sv_pos"]
    rng = np.linalg.norm(sv_pos - data["rcvr_pos"][None, :], axis=1)

    # filename: "{scenario}_{n_ant}element_{imu_model}_{attenuation}dB.npz"
    with pool.Pool(processes=14) as p:
        args = []
        for s in conf["mc_params"]["scenarios"]:
            for n in conf["mc_params"]["n_ant"]:
                for i in conf["mc_params"]["imu_models"]:
                    for a in conf["mc_params"]["attenuation"]:
                        args.append(
                            (
                                RESULTS_PATH / "monte_carlo" / f"{s}_{n}element_{i}_{a}dB",
                                RESULTS_PATH / f"{s}" / f"{n}_element" / f"{i}_imu" / f"{a}_dB",
                                sv_pos,
                                rng,
                            )
                        )
        for b in tqdm(
            p.imap(process_result, args),
            total=len(conf["mc_params"]["scenarios"])
            * len(conf["mc_params"]["n_ant"])
            * len(conf["mc_params"]["imu_models"])
            * len(conf["mc_params"]["attenuation"]),
            desc=f"[\u001b[31;1mcharlizard\u001b[0m] Compiling results ... ",
            ascii=".>#",
            bar_format="{desc}{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{rate_fmt}]",
        ):
            pass

    toc(t0, f"[\u001b[31;1mcharlizard\u001b[0m] GNSS+ Monte Carlo results compiled in")
