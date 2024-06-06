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
import h5py
from tqdm import tqdm
from multiprocessing import freeze_support, pool, cpu_count
from log_utils import *
from yaml_parser import *

import navtools as nt
from navsim.configuration import *
from navsim.simulations.measurement import MeasurementSimulation
from navsim.simulations.ins import INSSimulation

from simulate_rcvr import PROJECT_PATH, CONFIG_FILE, DATA_PATH, RESULTS_PATH


def generate_config(yml: dict):
    # generate navsim configurations
    t = TimeConfiguration(
        duration=yml["time"]["duration"],
        fsim=yml["time"]["fsim"],
        year=yml["date"]["year"],
        month=yml["date"]["month"],
        day=yml["date"]["day"],
        hour=yml["date"]["hour"],
        minute=yml["date"]["minute"],
        second=yml["date"]["second"],
    )
    c = ConstellationsConfiguration(
        emitters={"gps": SignalConfiguration("gps", 0.0, "bpsk")},
        mask_angle=yml["constellations"]["mask_angle"],
    )
    e = ErrorConfiguration(
        ionosphere=yml["errors"]["ionosphere"],
        troposphere=yml["errors"]["troposphere"],
        rx_clock=yml["errors"]["rx_clock"],
    )
    i = IMUConfiguration(
        model=yml["imu"]["model"],
        osr=yml["imu"]["osr"],
        mobility=yml["imu"]["mobility"],
    )
    conf = SimulationConfiguration(
        time=t,
        constellations=c,
        errors=e,
        imu=i,
    )
    return conf


def save_chunk(x):
    conf, p, m, k0, dt, pos, vel, blk_idx = x
    meas_sim = MeasurementSimulation(conf, disable_progress=True)
    meas_sim.generate_truth(pos, vel)
    meas_sim.simulate()

    tmp_tow = np.zeros(p)
    tmp_id = np.zeros((p, m), dtype=dt)
    tmp_cn0 = np.zeros((p, m))
    tmp_pos = np.zeros((p, m, 3))
    tmp_vel = np.zeros((p, m, 3))
    tmp_aoa = np.zeros((p, m, 2))
    for j in range(p):
        tmp_id[j, :] = np.array([k for k in meas_sim.emitter_states.truth[j].keys()], dtype=dt)
        tmp_pos[j, :, :] = np.array([v.pos for v in meas_sim.emitter_states.truth[j].values()], dtype=float)
        tmp_vel[j, :, :] = np.array([v.vel for v in meas_sim.emitter_states.truth[j].values()], dtype=float)
        tmp_aoa[j, :, :] = np.array([[v.az, v.el] for v in meas_sim.emitter_states.truth[j].values()], dtype=float)
        tmp_cn0[j, :] = np.array([v.cn0 for v in meas_sim.observables[j].values()], dtype=float)
        tmp_tow[j] = float(meas_sim.emitter_states.truth[j][k0].gps_time.tow)
        # pbar.update()

    return tmp_tow, tmp_id, tmp_cn0, tmp_pos, tmp_vel, tmp_aoa, blk_idx


def generate_trajectory(scenario: str, conf: SimulationConfiguration, yml: dict):
    nt.io.ensure_exist(RESULTS_PATH / "truth_data")
    dump_filename = RESULTS_PATH / "truth_data" / f"{scenario}2.h5"
    data_filename = DATA_PATH / f"{scenario}.csv"

    # open simulators
    ins_sim = INSSimulation(conf, use_config_fsim=True)
    conf.time.duration = 2 / conf.time.fsim
    meas_sim = MeasurementSimulation(conf, disable_progress=True)

    # generate trajectory with ins simulator
    ins_sim._INSSimulation__init_pva = np.genfromtxt(data_filename, delimiter=",", skip_header=1, max_rows=1)
    ins_sim._INSSimulation__motion_def = np.genfromtxt(data_filename, delimiter=",", skip_header=3)
    ins_sim.simulate()

    # generate satellite data with meas sim
    meas_sim.generate_truth(ins_sim.ecef_position[:1, :], ins_sim.ecef_velocity[:1, :])
    meas_sim.simulate()

    # save data to file
    with h5py.File(dump_filename, "w") as h5:
        # opts = {"compression": "gzip", "compression_opts": 9}
        opts = {}
        n = ins_sim.geodetic_position.shape[0]
        m = len(meas_sim.emitter_states.truth[0])
        p = 4090
        q = int(n / p)
        dt = f"S{len(list(meas_sim.emitter_states.truth[0].keys())[0])}"
        k0 = list(meas_sim.emitter_states.truth[0].keys())[0]

        h5.create_dataset(name="N", data=ins_sim.time.size)
        h5.create_dataset(name="time", data=ins_sim.time)
        h5.create_dataset(name="rcvr_lla", data=ins_sim.geodetic_position, chunks=(p, 3), **opts)
        h5.create_dataset(name="rcvr_pos", data=ins_sim.ecef_position, chunks=(p, 3), **opts)
        h5.create_dataset(name="rcvr_vel", data=ins_sim.ecef_velocity, chunks=(p, 3), **opts)
        h5.create_dataset(name="rcvr_att", data=ins_sim.euler_angles, chunks=(p, 3), **opts)
        h5.create_dataset(name="imu_acc", data=ins_sim.true_specific_force, chunks=(p, 3), **opts)
        h5.create_dataset(name="imu_gyr", data=ins_sim.true_angular_velocity, chunks=(p, 3), **opts)
        h5.create_dataset(name="gps_tow", shape=n, dtype=float, chunks=p, **opts)
        h5.create_dataset(name="sv_id", shape=(n, m), dtype=dt, chunks=(p, m), **opts)
        h5.create_dataset(name="sv_cn0", shape=(n, m), dtype=float, chunks=(p, m), **opts)
        h5.create_dataset(name="sv_pos", shape=(n, m, 3), dtype=float, chunks=(p, m, 3), **opts)
        h5.create_dataset(name="sv_vel", shape=(n, m, 3), dtype=float, chunks=(p, m, 3), **opts)
        h5.create_dataset(name="sv_aoa", shape=(n, m, 2), dtype=float, chunks=(p, m, 2), **opts)

        del meas_sim
        conf.time.duration = p / conf.time.fsim

        with pool.Pool(processes=cpu_count()) as mp:
            args = [
                (
                    conf,
                    p,
                    m,
                    k0,
                    dt,
                    ins_sim.ecef_position[blk_idx : (blk_idx + p), :],
                    ins_sim.ecef_velocity[blk_idx : (blk_idx + p), :],
                    np.arange(blk_idx, (blk_idx + p)),
                )
                for blk_idx in range(0, n, p)
            ]
            for tmp_tow, tmp_id, tmp_cn0, tmp_pos, tmp_vel, tmp_aoa, blk_idx in tqdm(
                mp.imap(save_chunk, args),
                total=q,
                desc="[\u001b[31;1mcharlizard\u001b[0m] writing to h5 file ",
                ascii=".>#",
                bar_format="{desc}{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{rate_fmt}]",
                ncols=120,
            ):
                h5["sv_id"][blk_idx, :] = tmp_id
                h5["sv_pos"][blk_idx, :, :] = tmp_pos
                h5["sv_vel"][blk_idx, :, :] = tmp_vel
                h5["sv_aoa"][blk_idx, :, :] = tmp_aoa
                h5["sv_cn0"][blk_idx, :] = tmp_cn0
                h5["gps_tow"][blk_idx] = tmp_tow


if __name__ == "__main__":
    freeze_support()
    t0 = tic("[\u001b[31;1mcharlizard\u001b[0m] starting ... ", use_logger=False)
    yp = YamlParser(CONFIG_FILE)
    yml = yp.Yaml2Dict()
    conf = generate_config(yml)
    generate_trajectory("imu_devall_drive", conf, yml)
    toc(t0, "[\u001b[31;1mcharlizard\u001b[0m] done! ", use_logger=False)

    # dump_filename = RESULTS_PATH / "truth_data" / "test.h5"
    # # data = dict(np.load(dump_filename, allow_pickle=True))
    # # data = nt.io.loadhdf5(dump_filename)
    # hf = nt.io.hdf5Slicer(dump_filename, ["time", "gps_week", "gps_tow", "sv_id", "sv_pos", "sv_vel", "sv_aoa", "sv_cn0"])
    # data = hf.load_slice(np.arange(10))
    # print()
