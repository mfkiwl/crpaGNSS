"""
|=================================== plot_monte_carlo_results.py ==================================|
|                                                                                                  |
|   Property of Daniel Sturdivant. Unauthorized copying of this file via any medium would be       |
|   super sad and unfortunate for me. Proprietary and confidential.                                |
|                                                                                                  |
| ------------------------------------------------------------------------------------------------ |
|                                                                                                  |
|   @file     scripts/plot_monte_carlo_results.py                                                  |
|   @brief    PLot results from monte carlo sims in 'monte_carlo_sim_pacific_pnt'.                 |
|   @author   Daniel Sturdivant <sturdivant20@gmail.com>                                           |
|   @date     March 2024                                                                           |
|                                                                                                  |
|==================================================================================================|
"""

import os
import numpy as np
import pandas as pd
from scipy.linalg import norm
from pathlib import Path

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator

import navtools as nt

N_RUNS = 100

PROJECT_PATH = Path(__file__).parents[2]
RESULTS_PATH = PROJECT_PATH / "results" / "pacific_pnt"
FIGURES_PATH = RESULTS_PATH / "figures"
SCENARIOS = ["leo", "buoy", "leo_and_buoy", "imu"]
# SCENARIOS = ["buoy"]
nt.io.ensure_exist(FIGURES_PATH)

bold_font = {"fontweight": "bold", "fontsize": 20}
COLORS = ["#100c08", "#324ab2", "#b47249", "#a52a2a", "#454d32"]


def load_data(path: str):
    filename = path / "mc_results.npz"
    return dict(np.load(filename))


def move_figure(f, x, y):
    """Move figure's upper left corner to pixel (x, y)"""
    backend = matplotlib.get_backend()
    if backend == "TkAgg":
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == "WXAgg":
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK, You can also use window.setGeometry
        f.canvas.manager.window.move(x, y)


def plot_individual(show: bool = False, save: bool = True):
    for scenario in SCENARIOS:
        # retrieve data from 'npz' file
        data = load_data(RESULTS_PATH / scenario)
        t = data["time"]
        title = scenario.replace("_", " ").replace(" and ", "+").upper()

        # plot position rmse values on one plot
        f_p_rmse, ax_p_rmse = plt.subplots(**{"figsize": (10, 5)})
        pos_norm = norm(data["position_rmse"], axis=1)
        d = data["position_rmse"]
        sns.lineplot(x=t, y=d[:, 0], ax=ax_p_rmse, label="East", marker="s", markevery=100, markersize=8)
        sns.lineplot(x=t, y=d[:, 1], ax=ax_p_rmse, label="North", marker="o", markevery=100)
        sns.lineplot(x=t, y=d[:, 2], ax=ax_p_rmse, label="Up", marker=">", markevery=100)
        sns.lineplot(x=t, y=pos_norm, ax=ax_p_rmse, label="Norm", linestyle="--")
        ax_p_rmse.set(xlabel="Time [s]", ylabel="RMSE [m]", yscale="log")
        ax_p_rmse.set_title(f"{title} Position RMSE", **bold_font)
        ax_p_rmse.legend()
        f_p_rmse.tight_layout()

        # plot velocity rmse values on one plot
        f_v_rmse, ax_v_rmse = plt.subplots(**{"figsize": (10, 5)})
        vel_norm = norm(data["velocity_rmse"], axis=1)
        d = data["velocity_rmse"]
        sns.lineplot(x=t, y=d[:, 0], ax=ax_v_rmse, label="East", marker="s", markevery=100, markersize=8)
        sns.lineplot(x=t, y=d[:, 1], ax=ax_v_rmse, label="North", marker="o", markevery=100)
        sns.lineplot(x=t, y=d[:, 2], ax=ax_v_rmse, label="Up", marker=">", markevery=100)
        sns.lineplot(x=t, y=vel_norm, ax=ax_v_rmse, label="Norm", linestyle="--")
        ax_v_rmse.set(xlabel="Time [s]", ylabel="RMSE [m/s]", yscale="log")
        ax_v_rmse.set_title(f"{title} Velocity RMSE", **bold_font)
        ax_v_rmse.legend()
        f_v_rmse.tight_layout()

        # --------------------------------------------------------------------------------------------------------------
        f_p_std, ax_p_std = plt.subplots(nrows=3, ncols=1, **{"figsize": (10, 12)})
        f_v_std, ax_v_std = plt.subplots(nrows=3, ncols=1, **{"figsize": (10, 12)})
        f_pall, ax_pall = plt.subplots(**{"figsize": (10, 5)})
        f_vall, ax_vall = plt.subplots(**{"figsize": (10, 5)})
        f_pall2, ax_pall2 = plt.subplots(**{"figsize": (10, 5)})
        f_vall2, ax_vall2 = plt.subplots(**{"figsize": (10, 5)})

        # plot 3 sigma position standard deviation plots
        for i, file in enumerate(os.listdir(RESULTS_PATH / scenario)):
            if "run" not in file:
                continue
            run = dict(np.load(RESULTS_PATH / scenario / file))
            sns.lineplot(
                x=t, y=run["position_error"][:, 0], ax=ax_p_std[0], color="#a2e3b8", label="_nolegend_", alpha=0.6
            )
            sns.lineplot(
                x=t, y=run["position_error"][:, 1], ax=ax_p_std[1], color="#a2e3b8", label="_nolegend_", alpha=0.6
            )
            sns.lineplot(
                x=t, y=run["position_error"][:, 2], ax=ax_p_std[2], color="#a2e3b8", label="_nolegend_", alpha=0.6
            )
            sns.lineplot(
                x=t, y=run["velocity_error"][:, 0], ax=ax_v_std[0], color="#a2e3b8", label="_nolegend_", alpha=0.6
            )
            sns.lineplot(
                x=t, y=run["velocity_error"][:, 1], ax=ax_v_std[1], color="#a2e3b8", label="_nolegend_", alpha=0.6
            )
            sns.lineplot(
                x=t, y=run["velocity_error"][:, 2], ax=ax_v_std[2], color="#a2e3b8", label="_nolegend_", alpha=0.6
            )
            sns.lineplot(
                x=t, y=norm(run["position_error"], axis=1), ax=ax_pall, color="#a2e3b8", label="_nolegend_", alpha=0.6
            )
            sns.lineplot(
                x=t, y=norm(run["velocity_error"], axis=1), ax=ax_vall, color="#a2e3b8", label="_nolegend_", alpha=0.6
            )
            sns.lineplot(
                x=t, y=run["position_error"].mean(axis=1), ax=ax_pall2, color="#a2e3b8", label="_nolegend_", alpha=0.6
            )
            sns.lineplot(
                x=t, y=run["velocity_error"].mean(axis=1), ax=ax_vall2, color="#a2e3b8", label="_nolegend_", alpha=0.6
            )

        d0 = data["position_error_mean"] + 3 * data["position_mc_std"]
        d1 = data["position_error_mean"] - 3 * data["position_mc_std"]
        sns.lineplot(x=t, y=3 * data["position_filter_std"][:, 0], ax=ax_p_std[0], color="#a52a2a", label="EKF")
        sns.lineplot(x=t, y=-3 * data["position_filter_std"][:, 0], ax=ax_p_std[0], color="#a52a2a", label="_nolegend_")
        sns.lineplot(x=t, y=d0[:, 0], ax=ax_p_std[0], color="#324ab2", label="MC")
        sns.lineplot(x=t, y=d1[:, 0], ax=ax_p_std[0], color="#324ab2", label="_nolegend_")

        sns.lineplot(x=t, y=3 * data["position_filter_std"][:, 1], ax=ax_p_std[1], color="#a52a2a")
        sns.lineplot(x=t, y=-3 * data["position_filter_std"][:, 1], ax=ax_p_std[1], color="#a52a2a")
        sns.lineplot(x=t, y=d0[:, 1], ax=ax_p_std[1], color="#324ab2")
        sns.lineplot(x=t, y=d1[:, 1], ax=ax_p_std[1], color="#324ab2")

        sns.lineplot(x=t, y=3 * data["position_filter_std"][:, 2], ax=ax_p_std[2], color="#a52a2a")
        sns.lineplot(x=t, y=-3 * data["position_filter_std"][:, 2], ax=ax_p_std[2], color="#a52a2a")
        sns.lineplot(x=t, y=d0[:, 2], ax=ax_p_std[2], color="#324ab2")
        sns.lineplot(x=t, y=d1[:, 2], ax=ax_p_std[2], color="#324ab2")

        m = norm(data["position_filter_std"], axis=1)
        sns.lineplot(x=t, y=3 * m, ax=ax_pall, color="#a52a2a", label="EKF")
        sns.lineplot(x=t, y=3 * norm(data["position_mc_std"], axis=1), ax=ax_pall, color="#324ab2", label="MC")

        m = data["position_filter_std"].mean(axis=1)
        sns.lineplot(x=t, y=3 * m, ax=ax_pall2, color="#a52a2a", label="EKF")
        sns.lineplot(x=t, y=-3 * m, ax=ax_pall2, color="#a52a2a", label="_nolabel_")
        sns.lineplot(x=t, y=d0.mean(axis=1), ax=ax_pall2, color="#324ab2", label="MC")
        sns.lineplot(x=t, y=d1.mean(axis=1), ax=ax_pall2, color="#324ab2", label="_nolabel_")

        ax_p_std[0].legend()
        ax_p_std[0].set_title(f"{title} EKF vs. MC Position 3σ Estimates", **bold_font)
        ax_p_std[0].set_ylabel("East [m]")
        ax_p_std[1].set_ylabel("North [m]")
        ax_p_std[2].set(xlabel="Time [s]", ylabel="Up [m]")
        f_p_std.tight_layout()
        f_p_std.subplots_adjust(hspace=0.05)

        ax_pall.legend()
        ax_pall.set_title(f"{title} EKF vs. MC Position Norm 3σ Estimates", **bold_font)
        ax_pall.set(xlabel="Time [s]", ylabel="Error [m]")
        f_pall.tight_layout()
        f_pall.subplots_adjust(hspace=0.05)

        ax_pall2.legend()
        ax_pall2.set_title(f"{title} EKF vs. MC Position Mean 3σ Estimates", **bold_font)
        ax_pall2.set(xlabel="Time [s]", ylabel="Error [m]")
        f_pall2.tight_layout()
        f_pall2.subplots_adjust(hspace=0.05)

        # plot 3 sigma velocity standard deviation plots
        d0 = data["velocity_error_mean"] + 3 * data["velocity_mc_std"]
        d1 = data["velocity_error_mean"] - 3 * data["velocity_mc_std"]
        sns.lineplot(x=t, y=3 * data["velocity_filter_std"][:, 0], ax=ax_v_std[0], color="#a52a2a", label="EKF")
        sns.lineplot(x=t, y=-3 * data["velocity_filter_std"][:, 0], ax=ax_v_std[0], color="#a52a2a", label="_nolegend_")
        sns.lineplot(x=t, y=d0[:, 0], ax=ax_v_std[0], color="#324ab2", label="MC")
        sns.lineplot(x=t, y=d1[:, 0], ax=ax_v_std[0], color="#324ab2", label="_nolegend_")

        sns.lineplot(x=t, y=3 * data["velocity_filter_std"][:, 1], ax=ax_v_std[1], color="#a52a2a")
        sns.lineplot(x=t, y=-3 * data["velocity_filter_std"][:, 1], ax=ax_v_std[1], color="#a52a2a")
        sns.lineplot(x=t, y=d0[:, 1], ax=ax_v_std[1], color="#324ab2")
        sns.lineplot(x=t, y=d1[:, 1], ax=ax_v_std[1], color="#324ab2")

        sns.lineplot(x=t, y=3 * data["velocity_filter_std"][:, 2], ax=ax_v_std[2], color="#a52a2a")
        sns.lineplot(x=t, y=-3 * data["velocity_filter_std"][:, 2], ax=ax_v_std[2], color="#a52a2a")
        sns.lineplot(x=t, y=d0[:, 2], ax=ax_v_std[2], color="#324ab2")
        sns.lineplot(x=t, y=d1[:, 2], ax=ax_v_std[2], color="#324ab2")

        m = norm(data["velocity_filter_std"], axis=1)
        sns.lineplot(x=t, y=3 * m, ax=ax_vall, color="#a52a2a", label="EKF")
        sns.lineplot(x=t, y=3 * norm(data["velocity_mc_std"], axis=1), ax=ax_vall, color="#324ab2", label="MC")

        m = data["velocity_filter_std"].mean(axis=1)
        sns.lineplot(x=t, y=3 * m, ax=ax_vall2, color="#a52a2a", label="EKF")
        sns.lineplot(x=t, y=-3 * m, ax=ax_vall2, color="#a52a2a", label="_nolabel_")
        sns.lineplot(x=t, y=d0.mean(axis=1), ax=ax_vall2, color="#324ab2", label="MC")
        sns.lineplot(x=t, y=d1.mean(axis=1), ax=ax_vall2, color="#324ab2", label="_nolabel_")

        ax_v_std[0].legend()
        ax_v_std[0].set_title(f"{title} EKF vs. MC Velocity 3σ Estimates", **bold_font)
        ax_v_std[0].set_ylabel("East [m/2]")
        ax_v_std[1].set_ylabel("North [m/2]")
        ax_v_std[2].set(xlabel="Time [s]", ylabel="Up [m/2]")
        f_v_std.tight_layout()
        f_v_std.subplots_adjust(hspace=0.05)

        ax_vall.legend()
        ax_vall.set_title(f"{title} EKF vs. MC Velocity Norm 3σ Estimates", **bold_font)
        ax_vall.set(xlabel="Time [s]", ylabel="Error [m/s]")
        f_vall.tight_layout()
        f_vall.subplots_adjust(hspace=0.05)

        ax_vall2.legend()
        ax_vall2.set_title(f"{title} EKF vs. MC Velocity Mean 3σ Estimates", **bold_font)
        ax_vall2.set(xlabel="Time [s]", ylabel="Error [m/s]")
        f_vall2.tight_layout()
        f_vall2.subplots_adjust(hspace=0.05)

        # --------------------------------------------------------------------------------------------------------------
        # plot gdop/vdop/hdop and number of emitters
        f_dop, ax_dop = plt.subplots(nrows=2, ncols=1, **{"figsize": (10, 8)})
        d = data["dop"]
        sns.lineplot(x=t, y=d[:, 0], ax=ax_dop[0], label="GDOP", marker="s", markevery=100, markersize=8)
        sns.lineplot(x=t, y=d[:, 2], ax=ax_dop[0], label="HDOP", marker="o", markevery=100)
        sns.lineplot(x=t, y=d[:, 3], ax=ax_dop[0], label="VDOP", marker=">", markevery=100)
        sns.lineplot(x=t, y=np.round(data["dop"][:, 5]), ax=ax_dop[1], linestyle="--")
        ax_dop[0].set_ylabel("Magnitude")
        ax_dop[0].set_title(f"{title} Dilution of Precision", **bold_font)
        ax_dop[0].legend()
        ax_dop[0].yaxis.set_major_locator(MaxNLocator(integer=True))
        ax_dop[1].set(xlabel="Time [s]", ylabel="Amount of Visible Emitters")
        ax_dop[1].yaxis.set_major_locator(MaxNLocator(integer=True))
        f_dop.tight_layout()
        f_dop.subplots_adjust(hspace=0.05)

        # --------------------------------------------------------------------------------------------------------------
        # plot 3 sigma position standard deviation plots
        f_std, ax_std = plt.subplots(**{"figsize": (10, 5)})
        filt_norm = norm(data["position_filter_std"], axis=1)
        mc_norm = norm(data["position_mc_std"], axis=1)
        sns.lineplot(x=t, y=filt_norm, ax=ax_std, label="EKF")
        sns.lineplot(x=t, y=mc_norm, ax=ax_std, label="MC")
        ax_std.set_title(f"{title} EKF vs. MC Position Standard Deviation", **bold_font)
        ax_std.set(xlabel="Time [s]", ylabel="σ [m]")

        f_std.tight_layout()
        f_std.subplots_adjust(hspace=0.05)

        # plot position mean and standard deviation plots
        f_blake, ax_blake = plt.subplots(nrows=2, ncols=1, **{"figsize": (10, 8)})
        sns.lineplot(x=t, y=data["position_error_mean"][:, 0], ax=ax_blake[0], label="E")
        sns.lineplot(x=t, y=data["position_error_mean"][:, 1], ax=ax_blake[0], label="N")
        sns.lineplot(x=t, y=data["position_error_mean"][:, 2], ax=ax_blake[0], label="U")
        sns.lineplot(x=t, y=data["velocity_error_mean"][:, 0], ax=ax_blake[1], label="E")
        sns.lineplot(x=t, y=data["velocity_error_mean"][:, 1], ax=ax_blake[1], label="N")
        sns.lineplot(x=t, y=data["velocity_error_mean"][:, 2], ax=ax_blake[1], label="U")
        ax_blake[0].set_title(f"{title} MC error mean", **bold_font)
        ax_blake[0].legend()
        ax_blake[0].set(ylabel="Position [m]")
        ax_blake[1].set(xlabel="Time [s]", ylabel="Velocity [m/s]")

        f_std.tight_layout()
        f_std.subplots_adjust(hspace=0.05)

        # show plots if desired (pixels set based on my lab laptop)
        if show:
            move_figure(f_p_rmse, 350, 850)
            move_figure(f_v_rmse, 400, 900)
            move_figure(f_p_std, 350, 850)
            move_figure(f_v_std, 400, 900)
            move_figure(f_dop, 300, 800)
            move_figure(f_blake, 300, 800)
            plt.show()

        # save plots if desired
        if save:
            nt.io.ensure_exist(FIGURES_PATH / f"{scenario}")
            f_p_rmse.savefig(FIGURES_PATH / f"{scenario}" / f"{scenario}_rmse_position.jpeg")
            f_v_rmse.savefig(FIGURES_PATH / f"{scenario}" / f"{scenario}_rmse_velocity.jpeg")
            f_p_std.savefig(FIGURES_PATH / f"{scenario}" / f"{scenario}_3_sigma_position.jpeg")
            f_v_std.savefig(FIGURES_PATH / f"{scenario}" / f"{scenario}_3_sigma_velocity.jpeg")
            f_dop.savefig(FIGURES_PATH / f"{scenario}" / f"{scenario}_dop.jpeg")
            f_std.savefig(FIGURES_PATH / f"{scenario}" / f"{scenario}_standard_deviation.jpeg")
            f_blake.savefig(FIGURES_PATH / f"{scenario}" / f"{scenario}_blake.jpeg")
            f_pall.savefig(FIGURES_PATH / f"{scenario}" / f"{scenario}_3_sigma_position_norm.jpeg")
            f_pall2.savefig(FIGURES_PATH / f"{scenario}" / f"{scenario}_3_sigma_position_mean.jpeg")
            f_vall.savefig(FIGURES_PATH / f"{scenario}" / f"{scenario}_3_sigma_velocity_norm.jpeg")
            f_vall2.savefig(FIGURES_PATH / f"{scenario}" / f"{scenario}_3_sigma_velocity_mean.jpeg")

            pp = PdfPages(FIGURES_PATH / f"_{scenario}_individual.pdf")
            pp.savefig(f_p_rmse)
            pp.savefig(f_dop)
            pp.savefig(f_v_rmse)
            pp.savefig(f_pall2)
            pp.savefig(f_vall2)
            pp.close()

        plt.close("all")


def plot_comparison(show: bool = False, save: bool = True):
    f_p, ax_p = plt.subplots(**{"figsize": (10, 5)})
    f_v, ax_v = plt.subplots(**{"figsize": (10, 5)})
    f_p_imu, ax_p_imu = plt.subplots(**{"figsize": (10, 5)})
    f_v_imu, ax_v_imu = plt.subplots(**{"figsize": (10, 5)})
    f_dop, ax_dop = plt.subplots(nrows=4, ncols=1, **{"figsize": (10, 12)})

    mkrs = ["s", "o", ">", ""]
    mkr_size = [8, 10, 10, 10]
    l_style = ["-", "-", "-", "--"]
    for scenario, mkr, mkrs, ls in zip(SCENARIOS, mkrs, mkr_size, l_style):
        # retrieve data from 'npz' file
        data = load_data(RESULTS_PATH / scenario)
        t = data["time"]
        title = scenario.replace("_", " ").replace(" and ", "+").upper()

        # plot rmse norm
        d = data["dop"]
        d0 = norm(data["position_rmse"], axis=1)
        d1 = norm(data["velocity_rmse"], axis=1)
        sns.lineplot(x=t, y=d0, ax=ax_p_imu, label=f"{title}", marker=mkr, linestyle=ls, markersize=mkrs, markevery=100)
        sns.lineplot(x=t, y=d1, ax=ax_v_imu, label=f"{title}", marker=mkr, linestyle=ls, markersize=mkrs, markevery=100)
        if scenario.casefold() != "imu":
            sns.lineplot(x=t, y=d0, ax=ax_p, label=f"{title}", marker=mkr, linestyle=ls, markersize=mkrs, markevery=100)
            sns.lineplot(x=t, y=d1, ax=ax_v, label=f"{title}", marker=mkr, linestyle=ls, markersize=mkrs, markevery=100)
            sns.lineplot(
                x=t, y=d[:, 0], ax=ax_dop[0], label=f"{title}", marker=mkr, linestyle=ls, markersize=mkrs, markevery=100
            )
            sns.lineplot(x=t, y=d[:, 2], ax=ax_dop[1], marker=mkr, linestyle=ls, markersize=mkrs, markevery=100)
            sns.lineplot(x=t, y=d[:, 3], ax=ax_dop[2], marker=mkr, linestyle=ls, markersize=mkrs, markevery=100)
            sns.lineplot(
                x=t, y=np.round(d[:, 5]), ax=ax_dop[3], marker=mkr, linestyle=ls, markersize=mkrs, markevery=100
            )

    ax_p.set(xlabel="Time [s]", ylabel="RMSE [m]", yscale="log")
    ax_p.set_title(f"Position RMSE", **bold_font)
    ax_p.legend()
    f_p.tight_layout()
    f_p.subplots_adjust(hspace=0.05)

    ax_p_imu.set(xlabel="Time [s]", ylabel="RMSE [m]", yscale="log")
    ax_p_imu.set_title(f"Position RMSE including IMU", **bold_font)
    ax_p_imu.legend()
    f_p_imu.tight_layout()
    f_p_imu.subplots_adjust(hspace=0.05)

    ax_v.set(xlabel="Time [s]", ylabel="RMSE [m/s]", yscale="log")
    ax_v.set_title(f"Velocity RMSE", **bold_font)
    ax_v.legend()
    f_v.tight_layout()
    f_v.subplots_adjust(hspace=0.05)

    ax_v_imu.set(xlabel="Time [s]", ylabel="RMSE [m/s]", yscale="log")
    ax_v_imu.set_title(f"Velocity RMSE including IMU", **bold_font)
    ax_v_imu.legend()
    ax_v_imu.set_yscale("log")
    f_v_imu.tight_layout()
    f_v_imu.subplots_adjust(hspace=0.05)

    ax_dop[0].set(ylabel="GDOP", ylim=[0, 22])
    ax_dop[0].set_title(f"Dilution of Precision", **bold_font)
    ax_dop[0].legend()
    ax_dop[0].yaxis.set_major_locator(MaxNLocator(integer=True))
    ax_dop[1].set(ylabel="HDOP", ylim=[0, 5])
    ax_dop[1].yaxis.set_major_locator(MaxNLocator(integer=True))
    # ax_dop[1].legend().set_visible(False)
    ax_dop[2].set(ylabel="VDOP", ylim=[0, 22])
    ax_dop[2].yaxis.set_major_locator(MaxNLocator(integer=True))
    # ax_dop[2].legend().set_visible(False)
    ax_dop[3].set(xlabel="Time [s]", ylabel="Amount of Visible Emitters")
    ax_dop[3].yaxis.set_major_locator(MaxNLocator(integer=True))
    # ax_dop[3].legend().set_visible(False)
    f_dop.tight_layout()
    f_dop.subplots_adjust(hspace=0.05)

    # show plots if desired (pixels set based on my lab laptop)
    if show:
        move_figure(f_p_imu, 350, 850)
        move_figure(f_v_imu, 400, 900)
        move_figure(f_p, 450, 950)
        move_figure(f_v, 500, 1000)
        move_figure(f_dop, 300, 800)
        plt.show()

    # save plots if desired
    if save:
        nt.io.ensure_exist(FIGURES_PATH / f"comparisons")
        f_p_imu.savefig(FIGURES_PATH / "comparisons" / f"comparison_rmse_position.jpeg")
        f_v_imu.savefig(FIGURES_PATH / "comparisons" / f"comparison_rmse_velocity.jpeg")
        f_p.savefig(FIGURES_PATH / "comparisons" / f"comparison_rmse_position_no_imu.jpeg")
        f_v.savefig(FIGURES_PATH / "comparisons" / f"comparison_rmse_velocity_no_imu.jpeg")
        f_dop.savefig(FIGURES_PATH / "comparisons" / f"comparison_dop.jpeg")

        pp = PdfPages(FIGURES_PATH / f"_comparison.pdf")
        pp.savefig(f_p)
        pp.savefig(f_dop)
        pp.savefig(f_v)
        pp.savefig(f_p_imu)
        pp.savefig(f_v_imu)
        pp.close()

    plt.close("all")


def plot_trajectory(show: bool = False, save: bool = True):
    import navsim as ns
    import navtools as nt

    # create artificial config
    config = ns.SimulationConfiguration(
        time=ns.TimeConfiguration(200, 0.1, 2024, 1, 1, 12, 1, 0),
        constellations=ns.ConstellationsConfiguration(
            mask_angle=7.5,
            emitters={
                "iridium-next": ns.SignalConfiguration(signal="iridium"),
                # "orbcomm": ns.SignalConfiguration(signal="orbcomm"),
                "buoy": ns.SignalConfiguration(signal="buoy"),
            },
        ),
        errors=ns.ErrorConfiguration(),
        imu=ns.IMUConfiguration(model="perfect"),
    )
    ins_sim = ns.simulations.INSSimulation(config, False)
    motion_def_file_path = PROJECT_PATH / "data" / "imu_pacific_pnt.csv"

    # simulate artificial config
    ins_sim._INSSimulation__init_pva = np.genfromtxt(motion_def_file_path, delimiter=",", skip_header=1, max_rows=1)
    ins_sim._INSSimulation__motion_def = np.genfromtxt(motion_def_file_path, delimiter=",", skip_header=3)
    ins_sim.simulate()

    f_update = int(100 * 10)
    meas_sim = ns.simulations.MeasurementSimulation(config, False)
    meas_sim.generate_truth(ins_sim.ecef_position[0, :], ins_sim.ecef_velocity[0, :])
    meas_sim.simulate()

    # rotate emitters into enu frame
    print(ins_sim.geodetic_position[0, :])
    C_e_n = nt.ecef2enuDcm(ins_sim.geodetic_position[0, :] * np.array([np.pi / 180, np.pi / 180, 1.0]))
    ecef0 = ins_sim.ecef_position[0, :]
    e_keys = []
    e_pos = []
    # for es in meas_sim.emitter_states.truth:
    es = meas_sim.emitter_states.truth[0]
    for k, e in es.items():
        if k in e_keys:
            i = e_keys.index(k)
            e_pos[i].append(C_e_n @ (e.pos - ecef0))
        else:
            e_keys.append(k)
            e_pos.append([C_e_n @ (e.pos - ecef0)])
    print()

    # plot trajectory only
    f_traj, ax_traj = plt.subplots(**{"figsize": (10, 8)})
    ax_traj.plot(
        ins_sim.tangent_position[::100, 0],
        ins_sim.tangent_position[::100, 1],
        # ax=ax_traj,
        label="Trajectory",
        # marker="$\u1F6EA$",
        marker="$\u2708$",
        markersize=25,
        markevery=100,
        color="#b47249",
    )

    # plot combined trajectory and emitters
    f2, ax2 = plt.subplots(**{"figsize": (10, 8)})
    key_cnt = [0, 0]
    for k, e in zip(e_keys, e_pos):
        p = np.array(e)
        if "iridium" in k.casefold():
            key_cnt[0] += 1
            # mkr = "$\u1F6F0$"
            mkr = "$\u22C8$"
            clr = "#a52a2a"
            if key_cnt[0] > 1:
                lbl = "_nolegend_"
            else:
                lbl = "LEO"

            # sns.lineplot(
            #     x=p[:, 0],
            #     y=p[:, 1],
            #     ax=ax2,
            #     label=lbl,
            #     marker=mkr,
            #     markersize=18,
            #     markevery=10,
            #     color=clr,
            # )
            ax2.plot(
                p[0, 0],
                p[0, 1],
                label=lbl,
                marker=mkr,
                markersize=18,
                color=clr,
            )
        elif "buoy" in k.casefold():
            key_cnt[1] += 1
            # mkr = "$\u1F6DF$"
            mkr = "$\u265F$"
            clr = "#324ab2"
            if key_cnt[1] > 1:
                lbl = "_nolegend_"
            else:
                lbl = "BUOY"

            # sns.lineplot(
            #     x=p[:, 0],
            #     y=p[:, 1],
            #     ax=ax2,
            #     label=lbl,
            #     marker=mkr,
            #     markersize=18,
            #     markevery=10,
            #     color=clr,
            # )
            ax2.plot(
                p[0, 0],
                p[0, 1],
                label=lbl,
                marker=mkr,
                markersize=18,
                color=clr,
            )
    # sns.lineplot(
    #     x=ins_sim.tangent_position[::100, 0],
    #     y=ins_sim.tangent_position[::100, 1],
    #     x=ins_sim.tangent_position[0, 0],
    #     y=ins_sim.tangent_position[0, 1],
    #     ax=ax2,
    #     label="Trajectory",
    #     # marker="$\u1F6EA$",
    #     marker="$\u2708$",
    #     markersize=25,
    #     markevery=100,
    #     color="#b47249",
    # )
    ax2.plot(
        ins_sim.tangent_position[0, 0],
        ins_sim.tangent_position[0, 1],
        label="Trajectory",
        marker="$\u2708$",
        markersize=20,
        color="#b47249",
    )

    ax_traj.set(xlabel="East [m]", ylabel="North [m]", aspect="equal")
    ax2.set(xlabel="East [m]", ylabel="North [m]", aspect="equal")
    ax2.legend()
    f_traj.tight_layout()
    f_traj.subplots_adjust(hspace=0.05)
    f2.tight_layout()
    f2.subplots_adjust(hspace=0.05)

    # show plots if desired (pixels set based on my lab laptop)
    if show:
        move_figure(f_traj, 350, 850)
        move_figure(f2, 400, 900)
        plt.show()

    # save plots if desired
    if save:
        nt.io.ensure_exist(FIGURES_PATH / f"trajectory")
        f_traj.savefig(FIGURES_PATH / "trajectory" / f"trajectory.jpeg")
        f2.savefig(FIGURES_PATH / "trajectory" / f"trajectory_with_emitters.jpeg")

        pp = PdfPages(FIGURES_PATH / f"_trajectory.pdf")
        pp.savefig(f_traj)
        pp.savefig(f2)
        pp.close()


if __name__ == "__main__":
    # set seaborn variables and turn on grid
    sns.set_theme(
        font="Times New Roman",
        context="talk",
        palette=sns.color_palette(COLORS),
        style="ticks",
        rc={"axes.grid": True},
    )
    # Set1, Dark2, copper

    # print("[charlizard] plotting individual results ")
    # plot_individual(show=False, save=True)

    print("[charlizard] plotting comparison of results ")
    plot_comparison(show=False, save=True)

    # print("[charlizard] plotting 2d trajectory ")
    # plot_trajectory(show=False, save=True)
