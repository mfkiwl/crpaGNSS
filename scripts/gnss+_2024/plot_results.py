import os
import numpy as np
import navtools as nt
from scipy.linalg import norm

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator

from charlizard.plotting.geoplot import geoplot
from charlizard.plotting.skyplot import skyplot
from charlizard.plotting.plot_window import plotWindow
from simulate_rcvr import RESULTS_PATH

bold_font = {"fontweight": "bold", "fontsize": 20}
COLORS = [
    "#100c08",
    "#324ab2",
    "#c5961d",
    "#a52a2a",
    "#a2e3b8",
    "#454d32",
    "#c8c8c8",
]  # black, blue, orange-gold, red, olive, mint, gray


#! === plot_1_to_4_elements ===
def plot_1_to_4_elements(
    x: np.ndarray,
    one_element: np.ndarray,
    two_element: np.ndarray,
    three_element: np.ndarray,
    four_element: np.ndarray,
    fig: plt.Figure = None,
    ax: plt.Axes = None,
    colors: list = None,
    **kwargs,
):
    if fig is None and ax is None:
        fig, ax = plt.subplots()
    if colors is None:
        colors = COLORS[1:5]
    ax.plot(x, one_element, color=colors[0], **kwargs)
    ax.plot(x, two_element, color=colors[1], **kwargs)
    ax.plot(x, three_element, color=colors[2], **kwargs)
    ax.plot(x, four_element, color=colors[3], **kwargs)
    return fig, ax


if __name__ == "__main__":
    # set seaborn variables and turn on grid
    sns.set_theme(
        font="Times New Roman",
        # context="talk",
        context="notebook",
        palette=sns.color_palette(COLORS),
        style="ticks",
        rc={"axes.grid": True},
    )
    pw = plotWindow()

    scenarios = ["static"]
    attenuation = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
    imu_models = ["consumer", "industrial"]
    linestyles = ["--", "-"]
    markers = ["o", "x"]
    colors = [[COLORS[1], "#e18634", COLORS[3], "#00b300"], ["#9683ec", COLORS[2], "#db2026", COLORS[4]]]

    mc_path = RESULTS_PATH / "monte_carlo"
    # RESULTS_PATH / f"static" / f"1_element" / f"consumer_imu" / f"0.0_dB" / f"run0.npz"
    data_truth = nt.io.loadhdf5(RESULTS_PATH / "truth_data" / "static.h5")

    # # * --- generate skyplot ---
    # f0, ax0 = skyplot(
    #     az=data_truth["sv_aoa"][::3000, :, 0].T,
    #     el=data_truth["sv_aoa"][::3000, :, 1].T,
    #     name=data_truth["sv_id"][0, :].astype(str),
    #     deg=True,
    #     color=COLORS[5],
    #     edgecolors=COLORS[4],
    #     s=100,
    # )
    # pw.addPlot("skyplot", f0)

    # # * --- generate geoplot for dynamic path ---
    # f00, ax00 = geoplot(
    #     lon=data_truth["rcvr_lla"][::1000, 1],
    #     lat=data_truth["rcvr_lla"][::1000, 0],
    #     figsize=(10, 8),
    #     plot_init_pos=True,
    #     tiles="satellite",
    #     **{"color": COLORS[3], "s": 30, "label": "Truth"},
    # )
    # pw.addPlot("geoplot", f00)

    # * --- generates single run plots ---
    f1, ax1 = plt.subplots(figsize=(8, 8))
    f2, ax2 = plt.subplots(figsize=(8, 8))
    f3, ax3 = plt.subplots(figsize=(8, 8))
    time = np.arange(150)
    for i, imu in enumerate(imu_models):
        data_1element = np.load(RESULTS_PATH / "single_runs" / f"static_1element_{imu}_20.0dB.npz")
        data_2element = np.load(RESULTS_PATH / "single_runs" / f"static_2element_{imu}_20.0dB.npz")
        data_3element = np.load(RESULTS_PATH / "single_runs" / f"static_3element_{imu}_20.0dB.npz")
        data_4element = np.load(RESULTS_PATH / "single_runs" / f"static_4element_{imu}_20.0dB.npz")
        f1, ax1 = plot_1_to_4_elements(
            x=time,
            one_element=data_1element["cn0"][::50, :].mean(axis=1) - 2,
            two_element=data_2element["cn0"][::50, :].mean(axis=1) - 2,
            three_element=data_3element["cn0"][::50, :].mean(axis=1) - 2,
            four_element=data_4element["cn0"][::50, :].mean(axis=1) - 2,
            fig=f1,
            ax=ax1,
            colors=colors[i],
            linestyle=linestyles[i],
            marker=markers[i],
            markersize=4,
            markevery=10,
            markeredgecolor=None,
        )
        f2, ax2 = plot_1_to_4_elements(
            x=time,
            one_element=data_1element["attitude_error"][::50, 2],
            two_element=data_2element["attitude_error"][::50, 2],
            three_element=data_3element["attitude_error"][::50, 2],
            four_element=data_4element["attitude_error"][::50, 2],
            fig=f2,
            ax=ax2,
            colors=colors[i],
            linestyle=linestyles[i],
            marker=markers[i],
            markersize=4,
            markevery=10,
            markeredgecolor=None,
        )
        f3, ax3 = plot_1_to_4_elements(
            x=time,
            one_element=np.linalg.norm(data_1element["position_error"][::50, :], axis=1),
            two_element=np.linalg.norm(data_2element["position_error"][::50, :], axis=1),
            three_element=np.linalg.norm(data_3element["position_error"][::50, :], axis=1),
            four_element=np.linalg.norm(data_4element["position_error"][::50, :], axis=1),
            fig=f3,
            ax=ax3,
            colors=colors[i],
            linestyle=linestyles[i],
            marker=markers[i],
            markersize=4,
            markevery=10,
            markeredgecolor=None,
        )
    ax1.axhline(y=data_1element["cn0"][-1000::50, :].mean() - 2, color=COLORS[0], linestyle=":", label="_nolabel_")
    ax1.axhline(y=data_2element["cn0"][-1000::50, :].mean() - 2, color=COLORS[0], linestyle=":", label="_nolabel_")
    ax1.axhline(y=data_3element["cn0"][-1000::50, :].mean() - 2, color=COLORS[0], linestyle=":", label="_nolabel_")
    ax1.axhline(y=data_4element["cn0"][-1000::50, :].mean() - 2, color=COLORS[0], linestyle=":", label="_nolabel_")
    ax1.set(xlabel="Time [s]", ylabel=r"$\mathdefault{C/N_0}$ [dB-Hz]")
    ax2.set(xlabel="Time [s]", ylabel=r"Yaw Error [$^\circ$]", ylim=[-25, 15])
    ax3.set(xlabel="Time [s]", ylabel="Position Error [m]")
    # ax1.legend(["1 element", "_nolabel_", "2 element", "_nolabel_", "3 element", "_nolabel_", "4 element", "_nolabel_"])
    # ax2.legend(["1 element", "_nolabel_", "2 element", "_nolabel_", "3 element", "_nolabel_", "4 element", "_nolabel_"])
    # ax3.legend(["1 element", "_nolabel_", "2 element", "_nolabel_", "3 element", "_nolabel_", "4 element", "_nolabel_"])
    pw.addPlot("cn0", f1)
    pw.addPlot("yaw", f2)
    pw.addPlot("pos", f3)

    # * --- generate monte carlo plots ---
    f4, ax4 = plt.subplots()
    f5, ax5 = plt.subplots()
    f6, ax6 = plt.subplots()
    prob_trk_1element = np.zeros(len(attenuation))
    prob_trk_2element = np.zeros(len(attenuation))
    prob_trk_3element = np.zeros(len(attenuation))
    prob_trk_4element = np.zeros(len(attenuation))
    att_rmse_1element = np.zeros(len(attenuation))
    att_rmse_2element = np.zeros(len(attenuation))
    att_rmse_3element = np.zeros(len(attenuation))
    att_rmse_4element = np.zeros(len(attenuation))
    pos_rmse_1element = np.zeros(len(attenuation))
    pos_rmse_2element = np.zeros(len(attenuation))
    pos_rmse_3element = np.zeros(len(attenuation))
    pos_rmse_4element = np.zeros(len(attenuation))
    for i, imu in enumerate(imu_models):
        for j, a in enumerate(attenuation):
            prob_trk_1element[j] = np.load(RESULTS_PATH / "monte_carlo" / f"static_1element_{imu}_{a}dB.npz")["prob_tracking"]
            prob_trk_2element[j] = np.load(RESULTS_PATH / "monte_carlo" / f"static_2element_{imu}_{a}dB.npz")["prob_tracking"]
            prob_trk_3element[j] = np.load(RESULTS_PATH / "monte_carlo" / f"static_3element_{imu}_{a}dB.npz")["prob_tracking"]
            prob_trk_4element[j] = np.load(RESULTS_PATH / "monte_carlo" / f"static_4element_{imu}_{a}dB.npz")["prob_tracking"]
            # fmt: off
            att_rmse_1element[j] = np.load(RESULTS_PATH / "monte_carlo" / f"static_1element_{imu}_{a}dB.npz")["final_attitude_rmse"]
            att_rmse_2element[j] = np.load(RESULTS_PATH / "monte_carlo" / f"static_2element_{imu}_{a}dB.npz")["final_attitude_rmse"]
            att_rmse_3element[j] = np.load(RESULTS_PATH / "monte_carlo" / f"static_3element_{imu}_{a}dB.npz")["final_attitude_rmse"]
            att_rmse_4element[j] = np.load(RESULTS_PATH / "monte_carlo" / f"static_4element_{imu}_{a}dB.npz")["final_attitude_rmse"]
            pos_rmse_1element[j] = np.load(RESULTS_PATH / "monte_carlo" / f"static_1element_{imu}_{a}dB.npz")["final_position_rmse"]
            pos_rmse_2element[j] = np.load(RESULTS_PATH / "monte_carlo" / f"static_2element_{imu}_{a}dB.npz")["final_position_rmse"]
            pos_rmse_3element[j] = np.load(RESULTS_PATH / "monte_carlo" / f"static_3element_{imu}_{a}dB.npz")["final_position_rmse"]
            pos_rmse_4element[j] = np.load(RESULTS_PATH / "monte_carlo" / f"static_4element_{imu}_{a}dB.npz")["final_position_rmse"]
            # fmt: on
        f4, ax4 = plot_1_to_4_elements(
            x=45 - np.array(attenuation[3:]),
            # x=attenuation[3:],
            one_element=prob_trk_1element[3:],
            two_element=prob_trk_2element[3:],
            three_element=prob_trk_3element[3:],
            four_element=prob_trk_4element[3:],
            fig=f4,
            ax=ax4,
            colors=colors[i],
            linestyle=linestyles[i],
            marker=markers[i],
            markersize=4,
            markeredgecolor=None,
        )
        f5, ax5 = plot_1_to_4_elements(
            x=45 - np.array(attenuation[3:]),
            # x=attenuation[3:],
            one_element=att_rmse_1element[3:],
            two_element=att_rmse_2element[3:],
            three_element=att_rmse_3element[3:],
            four_element=att_rmse_4element[3:],
            fig=f5,
            ax=ax5,
            colors=colors[i],
            linestyle=linestyles[i],
            marker=markers[i],
            markersize=4,
            markeredgecolor=None,
        )
        f6, ax6 = plot_1_to_4_elements(
            x=45 - np.array(attenuation[3:]),
            # x=attenuation[3:],
            one_element=pos_rmse_1element[3:],
            two_element=pos_rmse_2element[3:],
            three_element=pos_rmse_3element[3:],
            four_element=pos_rmse_4element[3:],
            fig=f6,
            ax=ax6,
            colors=colors[i],
            linestyle=linestyles[i],
            marker=markers[i],
            markersize=4,
            markeredgecolor=None,
        )
    ax4.set(xlabel=r"Nominal $\mathdefault{C/N_0}$ [dB-Hz]", ylabel="Probability of Tracking")
    ax5.set(xlabel=r"Nominal $\mathdefault{C/N_0}$ [dB-Hz]", ylabel="Attitude RMSE [deg]", yscale="log")
    ax6.set(xlabel=r"Nominal $\mathdefault{C/N_0}$ [dB-Hz]", ylabel="Position RMSE [m]", yscale="log")
    pw.addPlot("static_mc_prob_track", f4)
    pw.addPlot("static_mc_att_rmse", f5)
    pw.addPlot("static_mc_pos_rmse", f6)

    f7, ax7 = plt.subplots()
    f8, ax8 = plt.subplots()
    f9, ax9 = plt.subplots()
    for i, imu in enumerate(imu_models):
        for j, a in enumerate(attenuation):
            prob_trk_1element[j] = np.load(RESULTS_PATH / "monte_carlo" / f"dynamic_1element_{imu}_{a}dB.npz")["prob_tracking"]
            prob_trk_2element[j] = np.load(RESULTS_PATH / "monte_carlo" / f"dynamic_2element_{imu}_{a}dB.npz")["prob_tracking"]
            prob_trk_3element[j] = np.load(RESULTS_PATH / "monte_carlo" / f"dynamic_3element_{imu}_{a}dB.npz")["prob_tracking"]
            prob_trk_4element[j] = np.load(RESULTS_PATH / "monte_carlo" / f"dynamic_4element_{imu}_{a}dB.npz")["prob_tracking"]
            # fmt: off
            att_rmse_1element[j] = np.load(RESULTS_PATH / "monte_carlo" / f"dynamic_1element_{imu}_{a}dB.npz")["final_attitude_rmse"]
            att_rmse_2element[j] = np.load(RESULTS_PATH / "monte_carlo" / f"dynamic_2element_{imu}_{a}dB.npz")["final_attitude_rmse"]
            att_rmse_3element[j] = np.load(RESULTS_PATH / "monte_carlo" / f"dynamic_3element_{imu}_{a}dB.npz")["final_attitude_rmse"]
            att_rmse_4element[j] = np.load(RESULTS_PATH / "monte_carlo" / f"dynamic_4element_{imu}_{a}dB.npz")["final_attitude_rmse"]
            pos_rmse_1element[j] = np.load(RESULTS_PATH / "monte_carlo" / f"dynamic_1element_{imu}_{a}dB.npz")["final_position_rmse"]
            pos_rmse_2element[j] = np.load(RESULTS_PATH / "monte_carlo" / f"dynamic_2element_{imu}_{a}dB.npz")["final_position_rmse"]
            pos_rmse_3element[j] = np.load(RESULTS_PATH / "monte_carlo" / f"dynamic_3element_{imu}_{a}dB.npz")["final_position_rmse"]
            pos_rmse_4element[j] = np.load(RESULTS_PATH / "monte_carlo" / f"dynamic_4element_{imu}_{a}dB.npz")["final_position_rmse"]
            # fmt: on
        f7, ax7 = plot_1_to_4_elements(
            x=45 - np.array(attenuation[3:]),
            # x=attenuation[3:],
            one_element=prob_trk_1element[3:],
            two_element=prob_trk_2element[3:],
            three_element=prob_trk_3element[3:],
            four_element=prob_trk_4element[3:],
            fig=f7,
            ax=ax7,
            colors=colors[i],
            linestyle=linestyles[i],
            marker=markers[i],
            markersize=4,
            markeredgecolor=None,
        )
        f8, ax8 = plot_1_to_4_elements(
            x=45 - np.array(attenuation[3:]),
            # x=attenuation[3:],
            one_element=att_rmse_1element[3:],
            two_element=att_rmse_2element[3:],
            three_element=att_rmse_3element[3:],
            four_element=att_rmse_4element[3:],
            fig=f8,
            ax=ax8,
            colors=colors[i],
            linestyle=linestyles[i],
            marker=markers[i],
            markersize=4,
            markeredgecolor=None,
        )
        f9, ax9 = plot_1_to_4_elements(
            x=45 - np.array(attenuation[3:]),
            # x=attenuation[3:],
            one_element=pos_rmse_1element[3:],
            two_element=pos_rmse_2element[3:],
            three_element=pos_rmse_3element[3:],
            four_element=pos_rmse_4element[3:],
            fig=f9,
            ax=ax9,
            colors=colors[i],
            linestyle=linestyles[i],
            marker=markers[i],
            markersize=4,
            markeredgecolor=None,
        )
    ax7.set(xlabel=r"Nominal $\mathdefault{C/N_0}$ [dB-Hz]", ylabel="Probability of Tracking")
    ax8.set(xlabel=r"Nominal $\mathdefault{C/N_0}$ [dB-Hz]", ylabel="Attitude RMSE [deg]", yscale="log")
    ax9.set(xlabel=r"Nominal $\mathdefault{C/N_0}$ [dB-Hz]", ylabel="Position RMSE [m]", yscale="log")
    pw.addPlot("dynamic_mc_prob_track", f7)
    pw.addPlot("dynamic_mc_att_rmse", f8)
    pw.addPlot("dynamic_mc_pos_rmse", f9)

    # ---
    # plt.show()
    pw.show()
