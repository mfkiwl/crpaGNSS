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

import numpy as np
import pandas as pd
from scipy.linalg import norm
from pathlib import Path

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import navtools as nt


PROJECT_PATH = Path(__file__).parents[2]
RESULTS_PATH = PROJECT_PATH / "results" / "pacific_pnt"
FIGURES_PATH = RESULTS_PATH / "figures"
# SCENARIOS = ["leo", "buoy", "leo_and_buoy", "imu"]
SCENARIOS = ["imu"]
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

        # plot position rmse values on one plot
        f_p_rmse, ax_p_rmse = plt.subplots(**{"figsize": (10, 5)})
        pos_norm = norm(data["position_rmse"], axis=1)
        d = data["position_rmse"]
        sns.lineplot(x=t, y=d[:, 0], ax=ax_p_rmse, label="East", marker="s", markevery=100, markersize=8)
        sns.lineplot(x=t, y=d[:, 1], ax=ax_p_rmse, label="North", marker="o", markevery=100)
        sns.lineplot(x=t, y=d[:, 2], ax=ax_p_rmse, label="Up", marker=">", markevery=100)
        sns.lineplot(x=t, y=pos_norm, ax=ax_p_rmse, label="Norm", linestyle="--")
        ax_p_rmse.set(xlabel="Time [s]", ylabel="RMSE [m]")
        ax_p_rmse.set_title(f"{scenario.replace('_', ' ').replace(' and ', '+').upper()} Position RMSE", **bold_font)
        ax_p_rmse.legend(loc="upper left")
        ax_p_rmse.set_axisbelow(True)
        f_p_rmse.tight_layout()

        # plot velocity rmse values on one plot
        f_v_rmse, ax_v_rmse = plt.subplots(**{"figsize": (10, 5)})
        vel_norm = norm(data["velocity_rmse"], axis=1)
        d = data["velocity_rmse"]
        sns.lineplot(x=t, y=d[:, 0], ax=ax_v_rmse, label="East", marker="s", markevery=100, markersize=8)
        sns.lineplot(x=t, y=d[:, 1], ax=ax_v_rmse, label="North", marker="o", markevery=100)
        sns.lineplot(x=t, y=d[:, 2], ax=ax_v_rmse, label="Up", marker=">", markevery=100)
        sns.lineplot(x=t, y=vel_norm, ax=ax_v_rmse, label="Norm", linestyle="--")
        ax_v_rmse.set(xlabel="Time [s]", ylabel="RMSE [m/s]")
        ax_v_rmse.set_title(f"{scenario.replace('_', ' ').replace(' and ', '+').upper()} Velocity RMSE", **bold_font)
        ax_v_rmse.legend(loc="upper left")
        ax_v_rmse.set_axisbelow(True)
        f_v_rmse.tight_layout()

        # plot attitude rmse values on one plot
        f_a_rmse, ax_a_rmse = plt.subplots(**{"figsize": (10, 5)})
        att_norm = norm(data["attitude_rmse"], axis=1)
        d = data["attitude_rmse"]
        sns.lineplot(x=t, y=d[:, 0], ax=ax_a_rmse, label="East", marker="s", markevery=100, markersize=8)
        sns.lineplot(x=t, y=d[:, 1], ax=ax_a_rmse, label="North", marker="o", markevery=100)
        sns.lineplot(x=t, y=d[:, 2], ax=ax_a_rmse, label="Up", marker=">", markevery=100)
        sns.lineplot(x=t, y=att_norm, ax=ax_a_rmse, label="Norm", linestyle="--")
        ax_a_rmse.set(xlabel="Time [s]", ylabel="RMSE [°]")
        ax_a_rmse.set_title(f"{scenario.replace('_', ' ').replace(' and ', '+').upper()} Attitude RMSE", **bold_font)
        ax_a_rmse.legend(loc="upper left")
        f_a_rmse.tight_layout()

        # plot clock rmse values on one plot
        f_c_rmse, ax_c_rmse = plt.subplots(nrows=2, ncols=1, **{"figsize": (10, 7)})
        sns.lineplot(x=t, y=data["clock_rmse"][:, 0], ax=ax_c_rmse[0], label="Bias")
        sns.lineplot(x=t, y=data["clock_rmse"][:, 1], ax=ax_c_rmse[1], label="Drift")
        ax_c_rmse[0].set(xlabel="Time [s]", ylabel="RMSE [m]")
        ax_c_rmse[1].set(xlabel="Time [s]", ylabel="RMSE [m/s]")
        ax_c_rmse[0].set_title(f"{scenario.replace('_', ' ').replace(' and ', '+').upper()} Clock RMSE", **bold_font)
        ax_c_rmse[0].legend(loc="upper left")
        ax_c_rmse[1].legend(loc="upper left")
        f_c_rmse.tight_layout()
        f_c_rmse.subplots_adjust(hspace=0.05)

        # plot 3 sigma position standard deviation plots
        f_p_std, ax_p_std = plt.subplots(nrows=3, ncols=1, **{"figsize": (10, 12)})
        sns.lineplot(x=t, y=3 * data["position_filter_std"][:, 0], ax=ax_p_std[0], color="r", label="EKF")
        sns.lineplot(x=t, y=-3 * data["position_filter_std"][:, 0], ax=ax_p_std[0], color="r", label="_nolegend_")
        sns.lineplot(
            x=t,
            y=data["position_error_mean"][:, 0] + 3 * data["position_mc_std"][:, 0],
            ax=ax_p_std[0],
            color="b",
            label="MC",
        )
        sns.lineplot(
            x=t,
            y=data["position_error_mean"][:, 0] - 3 * data["position_mc_std"][:, 0],
            ax=ax_p_std[0],
            color="b",
            label="_nolegend_",
        )
        ax_p_std[0].legend(loc="upper center", ncols=2)
        ax_p_std[0].set_title(
            f"{scenario.replace('_', ' ').replace(' and ', '+').upper()} EKF vs. MC Position 3σ Estimates", **bold_font
        )
        ax_p_std[0].set_ylabel("East [m]")

        sns.lineplot(x=t, y=3 * data["position_filter_std"][:, 1], ax=ax_p_std[1], color="r")
        sns.lineplot(x=t, y=-3 * data["position_filter_std"][:, 1], ax=ax_p_std[1], color="r")
        sns.lineplot(
            x=t, y=data["position_error_mean"][:, 1] + 3 * data["position_mc_std"][:, 1], ax=ax_p_std[1], color="b"
        )
        sns.lineplot(
            x=t, y=data["position_error_mean"][:, 1] - 3 * data["position_mc_std"][:, 1], ax=ax_p_std[1], color="b"
        )
        ax_p_std[1].set_ylabel("North [m]")

        sns.lineplot(x=t, y=3 * data["position_filter_std"][:, 2], ax=ax_p_std[2], color="r")
        sns.lineplot(x=t, y=-3 * data["position_filter_std"][:, 2], ax=ax_p_std[2], color="r")
        sns.lineplot(
            x=t, y=data["position_error_mean"][:, 2] + 3 * data["position_mc_std"][:, 2], ax=ax_p_std[2], color="b"
        )
        sns.lineplot(
            x=t, y=data["position_error_mean"][:, 2] - 3 * data["position_mc_std"][:, 2], ax=ax_p_std[2], color="b"
        )
        ax_p_std[2].set(xlabel="Time [s]", ylabel="Up [m]")

        f_p_std.tight_layout()
        f_p_std.subplots_adjust(hspace=0.05)

        # plot 3 sigma velocity standard deviation plots
        f_v_std, ax_v_std = plt.subplots(nrows=3, ncols=1, **{"figsize": (10, 12)})
        sns.lineplot(x=t, y=3 * data["velocity_filter_std"][:, 0], ax=ax_v_std[0], color="r", label="EKF")
        sns.lineplot(x=t, y=-3 * data["velocity_filter_std"][:, 0], ax=ax_v_std[0], color="r", label="_nolegend_")
        sns.lineplot(x=t, y=3 * data["velocity_mc_std"][:, 0], ax=ax_v_std[0], color="b", label="MC")
        sns.lineplot(x=t, y=-3 * data["velocity_mc_std"][:, 0], ax=ax_v_std[0], color="b", label="_nolegend_")
        ax_v_std[0].legend(loc="upper center", ncols=2)
        ax_v_std[0].set_title(
            f"{scenario.replace('_', ' ').replace(' and ', '+').upper()} EKF vs. MC Velocity 3σ Estimates", **bold_font
        )
        ax_v_std[0].set_ylabel("East [m/s]")

        sns.lineplot(x=t, y=3 * data["velocity_filter_std"][:, 1], ax=ax_v_std[1], color="r")
        sns.lineplot(x=t, y=-3 * data["velocity_filter_std"][:, 1], ax=ax_v_std[1], color="r")
        sns.lineplot(x=t, y=3 * data["velocity_mc_std"][:, 1], ax=ax_v_std[1], color="b")
        sns.lineplot(x=t, y=-3 * data["velocity_mc_std"][:, 1], ax=ax_v_std[1], color="b")
        ax_v_std[1].set_ylabel("North [m/s]")

        sns.lineplot(x=t, y=3 * data["velocity_filter_std"][:, 2], ax=ax_v_std[2], color="r")
        sns.lineplot(x=t, y=-3 * data["velocity_filter_std"][:, 2], ax=ax_v_std[2], color="r")
        sns.lineplot(x=t, y=3 * data["velocity_mc_std"][:, 2], ax=ax_v_std[2], color="b")
        sns.lineplot(x=t, y=-3 * data["velocity_mc_std"][:, 2], ax=ax_v_std[2], color="b")
        ax_v_std[2].set(xlabel="Time [s]", ylabel="Up [m/s]")

        f_v_std.tight_layout()
        f_v_std.subplots_adjust(hspace=0.05)

        # plot 3 sigma attitude standard deviation plots
        f_a_std, ax_a_std = plt.subplots(nrows=3, ncols=1, **{"figsize": (10, 12)})
        sns.lineplot(x=t, y=3 * data["attitude_filter_std"][:, 0], ax=ax_a_std[0], color="r", label="EKF")
        sns.lineplot(x=t, y=-3 * data["attitude_filter_std"][:, 0], ax=ax_a_std[0], color="r", label="_nolegend_")
        sns.lineplot(x=t, y=3 * data["attitude_mc_std"][:, 0], ax=ax_a_std[0], color="b", label="MC")
        sns.lineplot(x=t, y=-3 * data["attitude_mc_std"][:, 0], ax=ax_a_std[0], color="b", label="_nolegend_")
        ax_a_std[0].legend(loc="upper center", ncols=2)
        ax_a_std[0].set_title(
            f"{scenario.replace('_', ' ').replace(' and ', '+').upper()} EKF vs. MC Attitude 3σ Estimates", **bold_font
        )
        ax_a_std[0].set_ylabel("Roll [°]")

        sns.lineplot(x=t, y=3 * data["attitude_filter_std"][:, 1], ax=ax_a_std[1], color="r")
        sns.lineplot(x=t, y=-3 * data["attitude_filter_std"][:, 1], ax=ax_a_std[1], color="r")
        sns.lineplot(x=t, y=3 * data["attitude_mc_std"][:, 1], ax=ax_a_std[1], color="b")
        sns.lineplot(x=t, y=-3 * data["attitude_mc_std"][:, 1], ax=ax_a_std[1], color="b")
        ax_a_std[1].set_ylabel("Pitch [°]")

        sns.lineplot(x=t, y=3 * data["attitude_filter_std"][:, 2], ax=ax_a_std[2], color="r")
        sns.lineplot(x=t, y=-3 * data["attitude_filter_std"][:, 2], ax=ax_a_std[2], color="r")
        sns.lineplot(x=t, y=3 * data["attitude_mc_std"][:, 2], ax=ax_a_std[2], color="b")
        sns.lineplot(x=t, y=-3 * data["attitude_mc_std"][:, 2], ax=ax_a_std[2], color="b")
        ax_a_std[2].set(xlabel="Time [s]", ylabel="Yaw [°]")

        f_a_std.tight_layout()
        f_a_std.subplots_adjust(hspace=0.05)

        # plot 3 sigma clock standard deviation plots
        f_c_std, ax_c_std = plt.subplots(nrows=2, ncols=1, **{"figsize": (10, 7)})
        sns.lineplot(x=t, y=3 * data["clock_filter_std"][:, 0], ax=ax_c_std[0], color="r", label="EKF")
        sns.lineplot(x=t, y=-3 * data["clock_filter_std"][:, 0], ax=ax_c_std[0], color="r", label="_nolegend_")
        sns.lineplot(x=t, y=3 * data["clock_mc_std"][:, 0], ax=ax_c_std[0], color="b", label="MC")
        sns.lineplot(x=t, y=-3 * data["clock_mc_std"][:, 0], ax=ax_c_std[0], color="b", label="_nolegend_")
        ax_c_std[0].legend(loc="upper center", ncols=2)
        ax_c_std[0].set_title(
            f"{scenario.replace('_', ' ').replace(' and ', '+').upper()} EKF vs. MC Clock 3σ Estimates", **bold_font
        )
        ax_c_std[0].set_ylabel("Bias [m]")

        sns.lineplot(x=t, y=3 * data["clock_filter_std"][:, 1], ax=ax_c_std[1], color="r")
        sns.lineplot(x=t, y=-3 * data["clock_filter_std"][:, 1], ax=ax_c_std[1], color="r")
        sns.lineplot(x=t, y=3 * data["clock_mc_std"][:, 1], ax=ax_c_std[1], color="b")
        sns.lineplot(x=t, y=-3 * data["clock_mc_std"][:, 1], ax=ax_c_std[1], color="b")
        ax_c_std[1].set_ylabel("Drift [m/s]")

        f_c_std.tight_layout()
        f_c_std.subplots_adjust(hspace=0.05)

        # plot gdop/vdop/hdop and number of emitters
        f_dop, ax_dop = plt.subplots(nrows=2, ncols=1, **{"figsize": (10, 8)})
        d = data["dop"]
        sns.lineplot(x=t, y=d[:, 0], ax=ax_dop[0], label="GDOP", marker="s", markevery=100, markersize=8)
        sns.lineplot(x=t, y=d[:, 2], ax=ax_dop[0], label="HDOP", marker="o", markevery=100)
        sns.lineplot(x=t, y=d[:, 3], ax=ax_dop[0], label="VDOP", marker=">", markevery=100)
        sns.lineplot(x=t, y=data["dop"][:, 5], ax=ax_dop[1], linestyle="--")
        ax_dop[0].set_ylabel("Magnitude")
        ax_dop[0].set_title(
            f"{scenario.replace('_', ' ').replace(' and ', '+').upper()} Dilution of Precision", **bold_font
        )
        ax_dop[0].legend(loc="upper left")
        ax_dop[1].set(xlabel="Time [s]", ylabel="Amount of Visible Emitters")
        # ax_dop[1].legend(loc="upper left")

        f_dop.tight_layout()
        f_dop.subplots_adjust(hspace=0.05)

        # plot 3 sigma position standard deviation plots
        f_std, ax_std = plt.subplots(**{"figsize": (10, 5)})
        filt_norm = norm(data["position_filter_std"], axis=1)
        mc_norm = norm(data["position_mc_std"], axis=1)
        sns.lineplot(x=t, y=filt_norm, ax=ax_std, label="EKF")
        sns.lineplot(x=t, y=mc_norm, ax=ax_std, label="MC")
        # ax_std.legend(loc="upper center", ncols=2)
        ax_std.set_title(
            f"{scenario.replace('_', ' ').replace(' and ', '+').upper()} EKF vs. MC Position Standard Deviation",
            **bold_font,
        )
        ax_std.set(xlabel="Time [s]", ylabel="σ [m]")

        f_std.tight_layout()
        f_std.subplots_adjust(hspace=0.05)

        # plot position mean and standard deviation plots
        f_blake, ax_blake = plt.subplots(nrows=2, ncols=1, **{"figsize": (10, 8)})
        # filt_std_norm = norm(data["position_filter_std"], axis=1)
        # filt_mean_norm = np.zeros(filt_std_norm.size)
        # mc_std_norm = norm(data["position_mc_std"], axis=1)
        # mc_mean_norm = norm(data["position_error_mean"], axis=1)
        sns.lineplot(x=t, y=data["position_error_mean"][:, 0], ax=ax_blake[0], label="EKF std")
        sns.lineplot(x=t, y=data["position_error_mean"][:, 1], ax=ax_blake[0], label="_nolegend_")
        sns.lineplot(x=t, y=data["position_error_mean"][:, 2], ax=ax_blake[0], label="EKF mean")
        sns.lineplot(x=t, y=data["velocity_error_mean"][:, 0], ax=ax_blake[1], label="E")
        sns.lineplot(x=t, y=data["velocity_error_mean"][:, 1], ax=ax_blake[1], label="N")
        sns.lineplot(x=t, y=data["velocity_error_mean"][:, 2], ax=ax_blake[1], label="U")
        ax_blake[0].set_title(
            f"{scenario.replace('_', ' ').replace(' and ', '+').upper()} MC error mean",
            **bold_font,
        )
        ax_blake[0].legend()
        ax_blake[1].legend()
        ax_blake[0].set(ylabel="Position [m]")
        ax_blake[1].set(xlabel="Time [s]", ylabel="Velocity [m/s]")

        f_std.tight_layout()
        f_std.subplots_adjust(hspace=0.05)

        # show plots if desired (pixels set based on my lab laptop)
        if show:
            move_figure(f_p_rmse, 350, 850)
            move_figure(f_v_rmse, 400, 900)
            move_figure(f_a_rmse, 450, 950)
            move_figure(f_c_rmse, 500, 1000)
            move_figure(f_p_std, 350, 850)
            move_figure(f_v_std, 400, 900)
            move_figure(f_a_std, 450, 950)
            move_figure(f_c_std, 500, 1000)
            move_figure(f_dop, 300, 800)
            plt.show()

        # save plots if desired
        if save:
            f_p_rmse.savefig(FIGURES_PATH / f"{scenario}_rmse_position.jpeg")
            f_v_rmse.savefig(FIGURES_PATH / f"{scenario}_rmse_velocity.jpeg")
            f_a_rmse.savefig(FIGURES_PATH / f"{scenario}_rmse_attitude.jpeg")
            f_c_rmse.savefig(FIGURES_PATH / f"{scenario}_rmse_clock.jpeg")
            f_p_std.savefig(FIGURES_PATH / f"{scenario}_3_sigma_position.jpeg")
            f_v_std.savefig(FIGURES_PATH / f"{scenario}_3_sigma_velocity.jpeg")
            f_a_std.savefig(FIGURES_PATH / f"{scenario}_3_sigma_attitude.jpeg")
            f_c_std.savefig(FIGURES_PATH / f"{scenario}_3_sigma_clock.jpeg")
            f_dop.savefig(FIGURES_PATH / f"{scenario}_dop.jpeg")
            f_std.savefig(FIGURES_PATH / f"{scenario}_standard_deviation.jpeg")
            f_blake.savefig(FIGURES_PATH / f"{scenario}_blake.jpeg")

            pp = PdfPages(FIGURES_PATH / f"_{scenario}_individual.pdf")
            pp.savefig(f_p_rmse)
            pp.savefig(f_dop)
            pp.savefig(f_v_rmse)
            pp.savefig(f_std)
            pp.close()

        plt.close("all")


def plot_comparison(show: bool = False, save: bool = True):
    f_p, ax_p = plt.subplots(**{"figsize": (10, 5)})
    f_v, ax_v = plt.subplots(**{"figsize": (10, 5)})
    f_p_imu, ax_p_imu = plt.subplots(**{"figsize": (10, 5)})
    f_v_imu, ax_v_imu = plt.subplots(**{"figsize": (10, 5)})
    f_dop, ax_dop = plt.subplots(nrows=2, ncols=1, **{"figsize": (10, 8)})

    mkrs = ["s", "o", ">", ""]
    mkr_size = [8, 10, 10, 10]
    l_style = ["-", "-", "-", "--"]
    for scenario, mkr, mkrs, ls in zip(SCENARIOS, mkrs, mkr_size, l_style):
        # retrieve data from 'npz' file
        data = load_data(RESULTS_PATH / scenario)
        t = data["time"]

        # plot position rmse norm
        d = norm(data["position_rmse"], axis=1)
        sns.lineplot(
            x=t,
            y=d,
            ax=ax_p_imu,
            label=f"{scenario.replace('_', ' ').replace(' and ', '+').upper()}",
            marker=mkr,
            linestyle=ls,
            markersize=mkrs,
            markevery=100,
        )
        if scenario.casefold() != "imu":
            sns.lineplot(
                x=t,
                y=d,
                ax=ax_p,
                label=f"{scenario.replace('_', ' ').replace(' and ', '+').upper()}",
                marker=mkr,
                linestyle=ls,
                markersize=mkrs,
                markevery=100,
            )

        # plot velocity rmse norm
        d = norm(data["velocity_rmse"], axis=1)
        sns.lineplot(
            x=t,
            y=d,
            ax=ax_v_imu,
            label=f"{scenario.replace('_', ' ').replace(' and ', '+').upper()}",
            marker=mkr,
            linestyle=ls,
            markersize=mkrs,
            markevery=100,
        )
        if scenario.casefold() != "imu":
            sns.lineplot(
                x=t,
                y=d,
                ax=ax_v,
                label=f"{scenario.replace('_', ' ').replace(' and ', '+').upper()}",
                marker=mkr,
                linestyle=ls,
                markersize=mkrs,
                markevery=100,
            )

        # plot GDOP
        if scenario.casefold() != "imu":
            d = data["dop"]
            sns.lineplot(
                x=t,
                y=d[:, 0],
                ax=ax_dop[0],
                label=f"{scenario.replace('_', ' ').replace(' and ', '+').upper()}",
                marker=mkr,
                linestyle=ls,
                markersize=mkrs,
                markevery=100,
            )
            sns.lineplot(
                x=t,
                y=d[:, 5],
                ax=ax_dop[1],
                # label=f"{scenario.replace('_', ' ').title()}",
                marker=mkr,
                linestyle=ls,
                markersize=mkrs,
                markevery=100,
            )

    ax_p.set(xlabel="Time [s]", ylabel="RMSE [m]")
    ax_p.set_title(f"Position RMSE", **bold_font)
    ax_p.legend(loc="upper left")
    f_p.tight_layout()
    f_p.subplots_adjust(hspace=0.05)

    ax_p_imu.set(xlabel="Time [s]", ylabel="RMSE [m]")
    ax_p_imu.set_title(f"Position RMSE including IMU", **bold_font)
    ax_p_imu.legend(loc="upper left")
    f_p_imu.tight_layout()
    f_p_imu.subplots_adjust(hspace=0.05)

    ax_v.set(xlabel="Time [s]", ylabel="RMSE [m/s]")
    ax_v.set_title(f"Velocity RMSE", **bold_font)
    ax_v.legend(loc="upper left")
    f_v.tight_layout()
    f_v.subplots_adjust(hspace=0.05)

    ax_v_imu.set(xlabel="Time [s]", ylabel="RMSE [m/s]")
    ax_v_imu.set_title(f"Velocity RMSE including IMU", **bold_font)
    ax_v_imu.legend(loc="upper left")
    f_v_imu.tight_layout()
    f_v_imu.subplots_adjust(hspace=0.05)

    ax_dop[0].set(ylabel="Magnitude", ylim=[0, 17])
    ax_dop[0].set_title(f"Geometric Dilution of Precision", **bold_font)
    ax_dop[0].legend(loc="upper left")
    ax_dop[1].set(xlabel="Time [s]", ylabel="Amount of Visible Emitters")
    # ax_dop[1].legend(loc="upper left")
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
        f_p_imu.savefig(FIGURES_PATH / f"comparison_rmse_position.jpeg")
        f_v_imu.savefig(FIGURES_PATH / f"comparison_rmse_velocity.jpeg")
        f_p.savefig(FIGURES_PATH / f"comparison_rmse_position_no_imu.jpeg")
        f_v.savefig(FIGURES_PATH / f"comparison_rmse_velocity_no_imu.jpeg")
        f_dop.savefig(FIGURES_PATH / f"comparison_dop.jpeg")

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
        time=ns.TimeConfiguration(1000, 0.1, 2024, 1, 1, 0, 0, 0),
        constellations=ns.ConstellationsConfiguration(
            {
                "iridium-next": ns.SignalConfiguration(signal="iridium"),
                # "orbcomm": ns.SignalConfiguration(signal="orbcomm"),
                "buoy": ns.SignalConfiguration(signal="buoy"),
            }
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
    meas_sim.generate_truth(ins_sim.ecef_position[::f_update, :], ins_sim.ecef_velocity[::f_update, :])
    meas_sim.simulate()

    # rotate emitters into enu frame
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
    sns.lineplot(
        x=ins_sim.tangent_position[::100, 0],
        y=ins_sim.tangent_position[::100, 1],
        ax=ax_traj,
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

    ax_traj.set(xlabel="East [m]", ylabel="North [m]")
    ax2.set(xlabel="East [m]", ylabel="North [m]")
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
        f_traj.savefig(FIGURES_PATH / f"trajectory.jpeg")
        f2.savefig(FIGURES_PATH / f"trajectory_with_emitters.jpeg")

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

    print("[charlizard] plotting individual results ")
    plot_individual(show=True, save=False)

    # print("[charlizard] plotting comparison of results ")
    # plot_comparison(show=False, save=True)

    # print("[charlizard] plotting 2d trajectory ")
    # plot_trajectory(show=False, save=True)
