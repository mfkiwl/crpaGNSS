import numpy as np
import navtools as nt
from tqdm import tqdm
from multiprocessing import pool, cpu_count, freeze_support
from logutils import tic, toc
from yamlparser import YamlParser

from simulate_rcvr import CONFIG_FILE, RESULTS_PATH, run_rcvr

if __name__ == "__main__":
    freeze_support()
    print(f"[\u001b[31;1mcharlizard\u001b[0m] Starting GNSS+ Monte Carlo ...")
    t0 = tic()

    # parse yaml config file
    yp = YamlParser(CONFIG_FILE)
    conf = yp.Yaml2Dict()

    # initialize monte carlo params
    params = {
        "init_pva_err": conf["mc_params"]["init_pva_err_std"],
        "f_sim": conf["time"]["fsim"],
        "f_imu": conf["time"]["fimu"],
        "f_rcvr": conf["time"]["frcvr"],
        "clock_model": conf["errors"]["rx_clock"],
    }

    # folder path: scenario / n_ant / imu_model / attenuation / run.npz
    for scenario in conf["mc_params"]["scenarios"]:
        t_scenario = tic()
        params["scenario"] = scenario

        for n_ant in conf["mc_params"]["n_ant"]:
            params["n_ant"] = n_ant

            for imu_model in conf["mc_params"]["imu_models"]:
                params["imu_model"] = imu_model

                for attenuation in conf["mc_params"]["attenuation"]:
                    params["attenuation"] = attenuation
                    dump_path = (
                        RESULTS_PATH / f"{scenario}" / f"{n_ant}_element" / f"{imu_model}_imu" / f"{attenuation}_dB"
                    )
                    prompt_str = (
                        f"[\u001b[31;1mcharlizard\u001b[0m] Running {scenario.upper()} - {n_ant} ELEMENT - "
                        + f"{imu_model.upper()} IMU - {attenuation} dB "
                    )

                    # run in parallel
                    with pool.Pool(processes=cpu_count()) as p:
                        args = [(params, i, True, False, True, dump_path) for i in range(conf["mc_params"]["n_runs"])]
                        for b in tqdm(
                            p.imap(run_rcvr, args),
                            total=conf["mc_params"]["n_runs"],
                            desc=prompt_str,
                            ascii=".>#",
                            bar_format="{desc}{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{rate_fmt}]",
                            # ncols=120,
                        ):
                            pass

        toc(t_scenario, f"[\u001b[31;1mcharlizard\u001b[0m] {scenario.upper()} scenario finished in")
    toc(t0, f"[\u001b[31;1mcharlizard\u001b[0m] GNSS+ Monte Carlo finished in")
