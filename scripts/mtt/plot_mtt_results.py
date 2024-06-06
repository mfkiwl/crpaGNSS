"""
|======================================= plot_mtt_results.py ======================================|
|                                                                                                  |
|   Property of Daniel Sturdivant. Unauthorized copying of this file via any medium would be       |
|   super sad and unfortunate for me. Proprietary and confidential.                                |
|                                                                                                  |
| ------------------------------------------------------------------------------------------------ |
|                                                                                                  |
|   @file     scripts/mtt/plot_mtt_results.py                                                      |
|   @brief    Plot multi-target tracking results.                                                  |
|   @author   Daniel Sturdivant <sturdivant20@gmail.com>                                           |
|   @date     May 2024                                                                             |
|                                                                                                  |
|==================================================================================================|
"""

import os
import numpy as np
from pathlib import Path
import navtools as nt
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator

PROJECT_PATH = Path(__file__).parents[2]
RESULTS_PATH = PROJECT_PATH / "results" / "mtt"
FIGURES_PATH = RESULTS_PATH / "figures"
SCENARIO = ["multi_emitter_dynamic", "multi_emitter_static"]
COLORS = [
    "#100c08",
    "#324ab2",
    "#c5961d",
    "#a52a2a",
    "#454d32",
    "#a2e3b8",
    "#c8c8c8",
]  # black, blue, orange-gold, red, olive, mint, gray
N_RUNS = 100
SAVE = True
bold_font = {"fontweight": "bold", "fontsize": 20}


def load_data(filename: str):
    return dict(np.load(filename, allow_pickle=True))


def generate_mc_results(scenario: str):
    if os.path.isfile(RESULTS_PATH / scenario / f"mc_results.npz"):
        mc_results = load_data(RESULTS_PATH / scenario / f"mc_results.npz")

    else:
        # initialize data
        data = load_data(RESULTS_PATH / scenario / f"run1.npz")

        e_x = np.empty(data["time"].size, dtype=object)
        e_y = np.empty(data["time"].size, dtype=object)
        v_x = np.empty(data["time"].size, dtype=object)
        v_y = np.empty(data["time"].size, dtype=object)
        N = np.empty(data["time"].size, dtype=object)
        for i in range(e_x.size):
            e_x[i] = []
            e_y[i] = []
            v_x[i] = []
            v_y[i] = []
            N[i] = []

        for n in range(N_RUNS):
            # get data
            data = load_data(RESULTS_PATH / scenario / f"run{n}.npz")

            # loop through each time point in data
            for i in range(data["time"].size):
                for j in range(min([data["P_est"][i].shape[2], data["pos_err"][i].shape[0]])):
                    e_x[i].append(data["pos_err"][i][j, 0])
                    e_y[i].append(data["pos_err"][i][j, 1])
                    v_x[i].append(data["P_est"][i][0, 0, j])
                    v_y[i].append(data["P_est"][i][1, 1, j])
                N[i].append(data["n_est"][i])

        mc_results = {
            "mean_x": np.empty(data["time"].size, dtype=float),
            "mean_y": np.empty(data["time"].size, dtype=float),
            "rmse_x": np.empty(data["time"].size, dtype=float),
            "rmse_y": np.empty(data["time"].size, dtype=float),
            "mc_std_x": np.empty(data["time"].size, dtype=float),
            "mc_std_y": np.empty(data["time"].size, dtype=float),
            "an_std_x": np.empty(data["time"].size, dtype=float),
            "an_std_y": np.empty(data["time"].size, dtype=float),
            "mean_n": np.empty(data["time"].size, dtype=float),
            "e_x": e_x,
            "e_y": e_y,
            "time": data["time"],
        }
        for i in range(data["time"].size):
            mc_results["mean_x"][i] = np.array(e_x[i]).mean()
            mc_results["mean_y"][i] = np.array(e_y[i]).mean()
            mc_results["rmse_x"][i] = np.sqrt((np.array(e_x[i]) ** 2).mean())
            mc_results["rmse_y"][i] = np.sqrt((np.array(e_y[i]) ** 2).mean())
            mc_results["mc_std_x"][i] = np.array(e_x[i]).std()
            mc_results["mc_std_y"][i] = np.array(e_y[i]).std()
            mc_results["an_std_x"][i] = np.sqrt(np.array(v_x[i]).mean())
            mc_results["an_std_y"][i] = np.sqrt(np.array(v_y[i]).mean())
            mc_results["mean_n"][i] = np.array(N[i]).mean()

        np.savez_compressed(RESULTS_PATH / scenario / f"mc_results.npz", **mc_results)
    return mc_results


def plot_2d_position(data: dict, scenario: str, save: bool):
    f, ax = plt.subplots(**{"figsize": (8, 8)})

    # plot receivers
    sns.scatterplot(x=data["x_rcvr"][0, :], y=data["x_rcvr"][1, :], s=100, ax=ax, color=COLORS[0], label="Receivers")

    # plot estimates
    for i in range(data["x_est"].size):
        if data["x_est"][i].size > 0:
            if i == 0:
                ax.scatter(data["x_est"][i][0, :], data["x_est"][i][1, :], s=30, color=COLORS[5], marker="o", label="Estimate")
            else:
                ax.scatter(data["x_est"][i][0, :], data["x_est"][i][1, :], s=30, color=COLORS[5], marker="o", label="_nolabel_")

    # plot emitter paths
    for i in range(data["t_birth"].size):
        x = []
        y = []
        for j in range(data["t_birth"][i], data["t_death"][i]):
            if data["n_true"][j] < (i + 1):
                x.append(data["x_true"][j][0, i - (i - data["n_true"][j] + 1)])
                y.append(data["x_true"][j][1, i - (i - data["n_true"][j] + 1)])
            else:
                x.append(data["x_true"][j][0, i])
                y.append(data["x_true"][j][1, i])
        if i == 0:
            sns.lineplot(x=x, y=y, ax=ax, color=COLORS[1], linewidth=3, label="Emitters", estimator=None, sort=False)
        else:
            sns.lineplot(x=x, y=y, ax=ax, color=COLORS[1], linewidth=3, label="_nolegend_", estimator=None, sort=False)

    handles, labels = ax.get_legend_handles_labels()
    order = [0, 2, 1]
    ax.legend([handles[i] for i in order], [labels[i] for i in order])

    ax.set_title(f"2D East-North Position", **bold_font)
    ax.grid(visible=True, which="both", axis="both")
    ax.set(xlabel="East [m]", ylabel="North [m]", aspect="equal")
    f.tight_layout()

    if save:
        nt.io.ensure_exist(FIGURES_PATH / f"{scenario}")
        f.savefig(FIGURES_PATH / f"{scenario}" / f"{scenario}_2d_position.jpeg")
    # plt.show()
    return f, ax


def plot_position_vs_time(data: dict, scenario: str, save: bool):
    f, ax = plt.subplots(nrows=2, ncols=1, **{"figsize": (8, 8)})

    # plot estimates
    for i in range(data["x_est"].size):
        if data["x_est"][i].size > 0:
            t = np.repeat(data["time"][i], data["n_est"][i])
            if i == 0:
                ax[0].scatter(t, data["x_est"][i][0, :], s=30, color=COLORS[5], marker="o", label="Estimate")
                ax[1].scatter(t, data["x_est"][i][1, :], s=30, color=COLORS[5], marker="o", label="Estimate")
            else:
                ax[0].scatter(t, data["x_est"][i][0, :], s=30, color=COLORS[5], marker="o", label="_nolabel_")
                ax[1].scatter(t, data["x_est"][i][1, :], s=30, color=COLORS[5], marker="o", label="_nolabel_")

    # plot emitter paths
    for i in range(data["t_birth"].size):
        x = []
        y = []
        for j in range(data["t_birth"][i], data["t_death"][i]):
            if data["n_true"][j] < (i + 1):
                x.append(data["x_true"][j][0, i - (i - data["n_true"][j] + 1)])
                y.append(data["x_true"][j][1, i - (i - data["n_true"][j] + 1)])
            else:
                x.append(data["x_true"][j][0, i])
                y.append(data["x_true"][j][1, i])
        t = data["time"][data["t_birth"][i] : data["t_death"][i]]
        if i == 0:
            sns.lineplot(x=t, y=x, ax=ax[0], color=COLORS[1], linewidth=3, label="Truth", estimator=None, sort=False)
            sns.lineplot(x=t, y=y, ax=ax[1], color=COLORS[1], linewidth=3, label="Truth", estimator=None, sort=False)
        else:
            sns.lineplot(x=t, y=x, ax=ax[0], color=COLORS[1], linewidth=3, label="_nolegend_", estimator=None, sort=False)
            sns.lineplot(x=t, y=y, ax=ax[1], color=COLORS[1], linewidth=3, label="_nolegend_", estimator=None, sort=False)

    handles, labels = ax[0].get_legend_handles_labels()
    order = [1, 0]
    ax[0].legend([handles[i] for i in order], [labels[i] for i in order])

    ax[0].set_title(f"Position vs. Time", **bold_font)
    ax[0].grid(visible=True, which="both", axis="both")
    ax[1].grid(visible=True, which="both", axis="both")
    ax[0].set(ylabel="East [m]")
    ax[1].set(xlabel="Time [s]", ylabel="North [m]")
    ax[1].get_legend().remove()
    f.tight_layout()

    if save:
        nt.io.ensure_exist(FIGURES_PATH / f"{scenario}")
        f.savefig(FIGURES_PATH / f"{scenario}" / f"{scenario}_position_v_time.jpeg")
    # plt.show()
    return f, ax


def plot_velocity_vs_time(data: dict, scenario: str, save: bool):
    f, ax = plt.subplots(nrows=2, ncols=1, **{"figsize": (8, 8)})

    # plot estimates
    for i in range(data["x_est"].size):
        if data["x_est"][i].size > 0:
            t = np.repeat(data["time"][i], data["n_est"][i])
            if i == 0:
                ax[0].scatter(t, data["x_est"][i][2, :], s=30, color=COLORS[5], marker="o", label="Estimate")
                ax[1].scatter(t, data["x_est"][i][3, :], s=30, color=COLORS[5], marker="o", label="Estimate")
            else:
                ax[0].scatter(t, data["x_est"][i][2, :], s=30, color=COLORS[5], marker="o", label="_nolabel_")
                ax[1].scatter(t, data["x_est"][i][3, :], s=30, color=COLORS[5], marker="o", label="_nolabel_")

    # plot emitter paths
    for i in range(data["t_birth"].size):
        x = []
        y = []
        for j in range(data["t_birth"][i], data["t_death"][i]):
            if data["n_true"][j] < (i + 1):
                x.append(data["x_true"][j][2, i - (i - data["n_true"][j] + 1)])
                y.append(data["x_true"][j][3, i - (i - data["n_true"][j] + 1)])
            else:
                x.append(data["x_true"][j][2, i])
                y.append(data["x_true"][j][3, i])
        t = data["time"][data["t_birth"][i] : data["t_death"][i]]
        if i == 0:
            sns.lineplot(x=t, y=x, ax=ax[0], color=COLORS[1], linewidth=3, label="Truth", estimator=None, sort=False)
            sns.lineplot(x=t, y=y, ax=ax[1], color=COLORS[1], linewidth=3, label="Truth", estimator=None, sort=False)
        else:
            sns.lineplot(x=t, y=x, ax=ax[0], color=COLORS[1], linewidth=3, label="_nolegend_", estimator=None, sort=False)
            sns.lineplot(x=t, y=y, ax=ax[1], color=COLORS[1], linewidth=3, label="_nolegend_", estimator=None, sort=False)

    handles, labels = ax[0].get_legend_handles_labels()
    order = [1, 0]
    ax[0].legend([handles[i] for i in order], [labels[i] for i in order])

    ax[0].set_title(f"Velocity vs. Time", **bold_font)
    ax[0].grid(visible=True, which="both", axis="both")
    ax[1].grid(visible=True, which="both", axis="both")
    ax[0].set(ylabel="East [m/s]")
    ax[1].set(xlabel="Time [s]", ylabel="North [m/s]")
    ax[1].get_legend().remove()
    f.tight_layout()

    if save:
        nt.io.ensure_exist(FIGURES_PATH / f"{scenario}")
        f.savefig(FIGURES_PATH / f"{scenario}" / f"{scenario}_velocity_v_time.jpeg")
    # plt.show()
    return f, ax


def plot_cardinality_vs_time(data: dict, scenario: str, save: bool):
    f, ax = plt.subplots(**{"figsize": (8, 5)})
    sns.lineplot(x=data["time"], y=data["n_true"], ax=ax, color=COLORS[1], linewidth=3, label="Truth")
    sns.scatterplot(x=data["time"], y=data["n_est"], ax=ax, s=50, color=COLORS[5], marker="o", edgecolor=None, label="Estimate")

    ax.set_title(f"Cardinality vs. Time", **bold_font)
    ax.grid(visible=True, which="both", axis="both")
    ax.set(xlabel="Time [s]", ylabel="Number of Targets")
    ax.legend()

    if save:
        nt.io.ensure_exist(FIGURES_PATH / f"{scenario}")
        f.savefig(FIGURES_PATH / f"{scenario}" / f"{scenario}_cardinality_v_time.jpeg")
    # plt.show()
    return f, ax


def plot_all_meas_vs_time(data: dict, scenario: str, save: bool):
    f, ax = plt.subplots(**{"figsize": (8, 5)})
    for i in range(data["time"].size):
        t = np.tile(data["time"][i], data["y_meas"][i].shape).ravel()
        if i == 0:
            sns.scatterplot(
                x=t,
                y=data["y_meas"][i].ravel(),
                ax=ax,
                s=20,
                color=COLORS[5],
                marker="x",
                linewidth=2.25,
                edgecolor=None,
                label="Estimate",
            )
        else:
            sns.scatterplot(
                x=t,
                y=data["y_meas"][i].ravel(),
                ax=ax,
                s=20,
                color=COLORS[5],
                marker="x",
                linewidth=2.25,
                edgecolor=None,
                label="_nolabel_",
            )

    # plot emitter paths
    for i in range(data["t_birth"].size):
        y1 = []
        y2 = []
        y3 = []
        y4 = []
        for j in range(data["t_birth"][i], data["t_death"][i]):
            if data["n_true"][j] < (i + 1):
                y1.append(data["y_true"][j][0, i - (i - data["n_true"][j] + 1)])
                y2.append(data["y_true"][j][1, i - (i - data["n_true"][j] + 1)])
                y3.append(data["y_true"][j][2, i - (i - data["n_true"][j] + 1)])
                y4.append(data["y_true"][j][3, i - (i - data["n_true"][j] + 1)])
            else:
                y1.append(data["y_true"][j][0, i])
                y2.append(data["y_true"][j][1, i])
                y3.append(data["y_true"][j][2, i])
                y4.append(data["y_true"][j][3, i])
        t = data["time"][data["t_birth"][i] : data["t_death"][i]]
        if i == 0:
            sns.lineplot(x=t, y=y1, ax=ax, color=COLORS[1], linewidth=3, label="Truth", estimator=None, sort=False)
        else:
            sns.lineplot(x=t, y=y1, ax=ax, color=COLORS[1], linewidth=3, label="_nolegend_", estimator=None, sort=False)
        sns.lineplot(x=t, y=y2, ax=ax, color=COLORS[1], linewidth=3, label="_nolegend_", estimator=None, sort=False)
        sns.lineplot(x=t, y=y3, ax=ax, color=COLORS[1], linewidth=3, label="_nolegend_", estimator=None, sort=False)
        sns.lineplot(x=t, y=y4, ax=ax, color=COLORS[1], linewidth=3, label="_nolegend_", estimator=None, sort=False)

    ax.set_title(f"Measurements and True Measurements", **bold_font)
    ax.grid(visible=True, which="both", axis="both")
    ax.set(xlabel="Time [s]", ylabel="Angle [rad]")
    ax.legend()

    if save:
        nt.io.ensure_exist(FIGURES_PATH / f"{scenario}")
        f.savefig(FIGURES_PATH / f"{scenario}" / f"{scenario}_all_meas_v_time.jpeg")
    # plt.show()
    return f, ax


def plot_mc_results(scenario: str, save: bool):
    data = generate_mc_results(scenario)

    f, ax = plt.subplots(nrows=2, ncols=1, **{"figsize": (8, 8)})
    for i in range(data["time"].size):
        t = np.repeat(data["time"][i], len(data["e_x"][i]))
        sns.scatterplot(
            x=t,
            y=np.array(data["e_x"][i]),
            ax=ax[0],
            color=COLORS[5],
            marker=".",
            s=25,
            alpha=0.6,
            edgecolor=None,
            label="_nolegend_",
        )
        sns.scatterplot(
            x=t,
            y=np.array(data["e_y"][i]),
            ax=ax[1],
            color=COLORS[5],
            marker=".",
            s=25,
            alpha=0.6,
            edgecolor=None,
            label="_nolegend_",
        )
    sns.lineplot(x=data["time"], y=data["mean_x"], ax=ax[0], color=COLORS[0], linewidth=3, label="MC μ")
    sns.lineplot(x=data["time"], y=data["mean_x"] + 3 * data["mc_std_x"], ax=ax[0], color=COLORS[1], linewidth=3, label="MC 3σ")
    sns.lineplot(
        x=data["time"], y=data["mean_x"] - 3 * data["mc_std_x"], ax=ax[0], color=COLORS[1], linewidth=3, label="_nolabel_"
    )
    sns.lineplot(x=data["time"], y=3 * data["an_std_x"], ax=ax[0], color=COLORS[3], linewidth=3, label="Analytical 3σ")
    sns.lineplot(x=data["time"], y=-3 * data["an_std_x"], ax=ax[0], color=COLORS[3], linewidth=3, label="_nolabel_")

    sns.lineplot(x=data["time"], y=data["mean_y"], ax=ax[1], color=COLORS[0], linewidth=3, label="MC μ")
    sns.lineplot(x=data["time"], y=data["mean_y"] + 3 * data["mc_std_y"], ax=ax[1], color=COLORS[1], linewidth=3, label="MC 3σ")
    sns.lineplot(
        x=data["time"], y=data["mean_y"] - 3 * data["mc_std_y"], ax=ax[1], color=COLORS[1], linewidth=3, label="_nolabel_"
    )
    sns.lineplot(x=data["time"], y=3 * data["an_std_y"], ax=ax[1], color=COLORS[3], linewidth=3, label="Analytical 3σ")
    sns.lineplot(x=data["time"], y=-3 * data["an_std_y"], ax=ax[1], color=COLORS[3], linewidth=3, label="_nolabel_")

    ax[0].set_title(f"Position Statistics", **bold_font)
    ax[0].grid(visible=True, which="both", axis="both")
    ax[1].grid(visible=True, which="both", axis="both")
    ax[0].set(ylabel="East Error [m]")
    ax[1].set(xlabel="Time [s]", ylabel="North Error [m]")
    ax[0].legend()
    ax[1].get_legend().remove()
    f.tight_layout()

    f1, ax1 = plt.subplots(**{"figsize": (8, 5)})
    sns.lineplot(x=data["time"], y=data["rmse_x"], ax=ax1, color=COLORS[1], linewidth=3, label="East")
    sns.lineplot(x=data["time"], y=data["rmse_y"], ax=ax1, color=COLORS[3], linewidth=3, label="North")
    ax1.set_title(f"Position RMSE vs. Time", **bold_font)
    ax1.grid(visible=True, which="both", axis="both")
    ax1.set(xlabel="Time [s]", ylabel="RMSE [m]")
    ax1.legend()
    f1.tight_layout()

    f2, ax2 = plt.subplots(**{"figsize": (8, 5)})
    data2 = load_data(RESULTS_PATH / scenario / "run1.npz")
    sns.lineplot(x=data["time"], y=data2["n_true"], ax=ax2, color=COLORS[3], linewidth=3, label="Truth")
    sns.lineplot(x=data["time"], y=data["mean_n"], ax=ax2, color=COLORS[1], linewidth=3, label="Estimate")
    ax2.set_title(f"Mean Number of Targets vs. Time", **bold_font)
    ax2.grid(visible=True, which="both", axis="both")
    ax2.set(xlabel="Time [s]", ylabel="Number of Target")
    ax2.legend()
    f2.tight_layout()

    if save:
        nt.io.ensure_exist(FIGURES_PATH / f"{scenario}")
        f.savefig(FIGURES_PATH / f"{scenario}" / f"{scenario}_stats_v_time.jpeg")
        f1.savefig(FIGURES_PATH / f"{scenario}" / f"{scenario}_rmse_v_time.jpeg")
        f2.savefig(FIGURES_PATH / f"{scenario}" / f"{scenario}_mean_cardinality_v_time.jpeg")

    return f, ax, f1, ax1, f2, ax2


if __name__ == "__main__":
    sns.set_theme(
        font="Times New Roman",
        context="talk",
        palette=sns.color_palette(COLORS),
        style="ticks",
        rc={"axes.grid": True},
    )

    for j in range(len(SCENARIO)):
        data = load_data(RESULTS_PATH / SCENARIO[j] / "run1.npz")
        f1, ax1 = plot_2d_position(data, SCENARIO[j], SAVE)
        f2, ax2 = plot_position_vs_time(data, SCENARIO[j], SAVE)
        f3, ax3 = plot_velocity_vs_time(data, SCENARIO[j], SAVE)
        f4, ax4 = plot_cardinality_vs_time(data, SCENARIO[j], SAVE)
        f5, ax5 = plot_all_meas_vs_time(data, SCENARIO[j], SAVE)
        f6, ax6, f7, ax7, f8, ax8 = plot_mc_results(SCENARIO[j], SAVE)

        pp = PdfPages(FIGURES_PATH / f"{SCENARIO[j]}" / f"_{SCENARIO[j]}.pdf")
        pp.savefig(f1)
        pp.savefig(f2)
        pp.savefig(f3)
        pp.savefig(f4)
        pp.savefig(f5)
        pp.savefig(f6)
        pp.savefig(f7)
        pp.savefig(f8)
        pp.close()

    # plt.show()
