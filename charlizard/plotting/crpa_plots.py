"""
|========================================== crpa_plots.py =========================================|
|                                                                                                  |
|   Property of Daniel Sturdivant. Unauthorized copying of this file via any medium would be       |
|   super sad and unfortunate for me. Proprietary and confidential.                                |
|                                                                                                  |
| ------------------------------------------------------------------------------------------------ |
|                                                                                                  |
|   @file     scripts/crpa_plots.py                                                                |
|   @brief    Basic CRPA visualization tools.                                                      |
|   @author   Daniel Sturdivant <sturdivant20@gmail.com>                                           |
|   @date     May 2024                                                                             |
|                                                                                                  |
|==================================================================================================|
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import navtools as nt
from pathlib import Path
import seaborn as sns
from multiprocessing import pool, cpu_count
from tqdm import tqdm

PROJECT_PATH = Path(__file__).parents[2]
RESULTS_PATH = PROJECT_PATH / "results"
FIGURES_PATH = RESULTS_PATH / "CRPA"

bold_font = {"fontweight": "bold", "fontsize": 20}
sns.set_theme(
    font="Times New Roman",
    context="talk",
    # palette=sns.color_palette(COLORS),
    style="ticks",
    rc={"axes.grid": True},
)


def plot_3d_array_factor(
    ant_xyz: np.ndarray,
    weights: np.ndarray,
    title: str,
    N_ANT: int,
    WAVELENGTH: float,
):
    el = np.linspace(0, np.pi / 2, 90)  # 0-90 deg
    az = np.linspace(0, 2 * np.pi, 360)  # 0-360 deg

    pattern_3d = np.zeros((90 * 360, 3))
    norm_gain = np.zeros(90 * 360)
    l = 0
    for i in range(el.size):
        for j in range(az.size):
            # test unit vector
            u = np.array([np.sin(az[j]) * np.cos(el[i]), np.cos(az[j]) * np.cos(el[i]), np.sin(el[i])])

            # sum power across all elements
            p = 0
            for k in range(N_ANT):
                p += weights[k] * np.exp(1j * 2 * np.pi * (ant_xyz[:, k] @ u) / WAVELENGTH)

            # calculate gain and direction
            # for some reason cos and sin of azimuth must be switched
            gain_xyz = np.array([np.cos(az[j]) * np.cos(el[i]), np.sin(az[j]) * np.cos(el[i]), np.sin(el[i])])
            norm_gain[l] = np.linalg.norm(p)
            pattern_3d[l, :] = norm_gain[l] * gain_xyz
            l += 1

    # plot
    f = plt.figure()
    ax = f.add_subplot(111)  # , projection='3d')
    im = ax.scatter(pattern_3d[:, 0], pattern_3d[:, 1], pattern_3d[:, 2], c=norm_gain, cmap="gnuplot")  # cmap='turbo')
    f.colorbar(im, ax=ax)
    ax.grid(visible=True, which="both")
    ax.set_xlabel("X (boresight)")
    ax.set_ylabel("Y (horizontal)")
    # ax.set_zlabel('Z (vertical)')
    f.suptitle(f"{title}, Gain Pattern (Amplitude) of Panel In Cartesian Coordinates")

    return f, ax


def plot_2d_polar_pattern(
    ant_xyz: np.ndarray,
    weights: np.ndarray,
    title: str,
    N_ANT: int,
    WAVELENGTH: float,
):
    az = np.linspace(0, 2 * np.pi, 360)
    el = 0.0
    gain_pattern = np.zeros(az.size)
    for i in range(az.size):
        # unit vector to test
        u = np.array([np.sin(az[i]) * np.cos(el), np.cos(az[i]) * np.cos(el), np.sin(el)])

        # sum power across all elements
        p = 0
        for k in range(N_ANT):
            p += weights[k] * np.exp(-1j * 2 * np.pi * (u @ ant_xyz[:, k]) / WAVELENGTH)

        # calculate gain
        gain_pattern[i] = np.abs(p)
    # gain_pattern = 10 * np.log10(gain_pattern)

    # plot
    f = plt.figure()
    ax = f.add_subplot(111, projection="polar", theta_offset=np.pi / 2, theta_direction=-1)
    ax.plot(az, gain_pattern)
    f.suptitle(f"{title}, CRPA Gain Pattern at an Elevation of 0 degrees")

    return f, ax


def gain_pattern_plot(ant, w, n, l):
    az = np.linspace(0, np.pi * 2, 360)
    pattern = np.zeros(360)
    for i in range(360):
        u = np.array([np.sin(az[i]), np.cos(az[i]), 0])
        p = 0
        for k in range(n):
            p += w[k] * np.exp(-1j * 2 * np.pi * (u @ ant[:, k]) / l)
        pattern[i] = np.abs(p)
    return az, pattern


def gain_pattern_plot_3d(x):
    ant, w, n, l, idx = x

    el = np.linspace(0, np.pi / 2, 90)  # 0-90 deg
    az = np.linspace(0, 2 * np.pi, 360)  # 0-360 deg
    pattern_3d = np.zeros((90 * 360, 3))
    norm_gain = np.zeros(90 * 360)
    m = 0
    for i in range(el.size):
        for j in range(az.size):
            u = np.array([np.sin(az[j]) * np.cos(el[i]), np.cos(az[j]) * np.cos(el[i]), np.sin(el[i])])
            p = 0
            for k in range(n):
                p += w[k] * np.exp(-1j * 2 * np.pi * (ant[:, k] @ u) / l)
            # gain_xyz = np.array([np.cos(az[j]) * np.cos(el[i]), np.sin(az[j]) * np.cos(el[i]), np.sin(el[i])])
            norm_gain[m] = np.linalg.norm(p)
            pattern_3d[m, :] = norm_gain[m] * u
            m += 1
    return norm_gain, pattern_3d, idx


def polar_pattern_animation(
    ant_xyz: np.ndarray,
    weights: np.ndarray,
    y_desired: np.ndarray,
    y_mixed: np.ndarray,
    title: str,
    N_ANT: int,
    WAVELENGTH: float,
    N_FRAMES: int,
    plot_3d: bool = True,
    show: bool = True,
    save: bool = False,
    filename: str = "2d_pattern_animation",
):

    # extract pattern for each timestep
    if plot_3d:
        pattern = np.zeros((N_FRAMES, 90 * 360, 3))
        gain = np.zeros((N_FRAMES, 90 * 360))
        with pool.Pool(processes=cpu_count()) as p:
            args = [(ant_xyz, weights[:, i], N_ANT, WAVELENGTH, i) for i in range(N_FRAMES)]
            for g, p, i in tqdm(
                p.imap(gain_pattern_plot_3d, args),
                total=N_FRAMES,
                desc="gain_pattern_plot_3d: ",
                ascii=".>#",
                bar_format="{desc}{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{rate_fmt}]",
                ncols=120,
            ):
                pattern[i, :, :] = p
                gain[i, :] = g
        # for i in range(N_FRAMES):
        #     gain[i, :], pattern[i, :, :] = gain_pattern_plot_3d(ant_xyz, weights[:, i], N_ANT, WAVELENGTH, i)
    else:
        pattern = np.zeros((N_FRAMES, 360))
        for i in range(N_FRAMES):
            az, pattern[i, :] = gain_pattern_plot(ant_xyz, weights[:, i], N_ANT, WAVELENGTH)

    # initialize plot
    f = plt.figure(figsize=(16, 9))
    if plot_3d:
        gs = f.add_gridspec(2, 2, width_ratios=[1.25, 1])
        ax0 = f.add_subplot(gs[:, 0])
        ax0.set(
            xlim=[
                pattern[:, :, :2].min() + 0.05 * pattern[:, :, :2].min(),
                pattern[:, :, :2].max() + 0.05 * pattern[:, :, :2].max(),
            ],
            ylim=[
                pattern[:, :, :2].min() + 0.05 * pattern[:, :, :2].min(),
                pattern[:, :, :2].max() + 0.05 * pattern[:, :, :2].max(),
            ],
        )
        ax0.set_aspect("equal", adjustable="box")
    else:
        gs = f.add_gridspec(2, 2)
        ax0 = f.add_subplot(gs[:, 0], projection="polar", theta_offset=np.pi / 2, theta_direction=-1)
        ax0.set(ylim=[0, pattern.max() + 0.05 * pattern.max()])
    ax1 = f.add_subplot(gs[0, 1])
    ax2 = f.add_subplot(gs[1, 1])

    f.suptitle(title, **bold_font)
    ax1.set(
        ylabel="Signal Amplitude",
        xlim=[0, N_FRAMES],
        ylim=[min([y_desired.min() - 0.5, y_mixed.min() - 0.5]), max([y_desired.max() + 0.5, y_mixed.max() + 0.5])],
    )
    ax2.set(
        xlabel="Iteration Number",
        ylabel="Weights",
        xlim=[0, N_FRAMES],
        ylim=[weights.real.min() - 0.5, weights.real.max() + 0.5],
    )

    # plot initial frame
    if plot_3d:
        pp = ax0.scatter(
            pattern[0, :, 0], pattern[0, :, 1], s=5, c=gain[0, :], cmap="gnuplot", vmin=gain.min(), vmax=gain.max()
        )
        f.colorbar(pp, ax=ax0)
    else:
        pp = ax0.plot(az, pattern[0, :])[0]
    dp = ax1.plot(0, y_desired[0], label="Desired")[0]
    mp = ax1.plot(0, y_mixed[0], label="Mixed")[0]
    wp = []
    for i in range(N_ANT):
        wp.append(ax2.plot(0, weights.real[i, 0], label=f"Weight {i}")[0])
    ax1.legend(loc="upper right")
    ax1.set_xticklabels([])
    ax2.legend(loc="upper right")
    f.tight_layout()
    f.subplots_adjust(hspace=0.05)

    def update(frame):
        t = np.arange(frame)
        if plot_3d:
            ax0.clear()
            ax0.set(
                xlim=[
                    pattern[:, :, :2].min() + 0.05 * pattern[:, :, :2].min(),
                    pattern[:, :, :2].max() + 0.05 * pattern[:, :, :2].max(),
                ],
                ylim=[
                    pattern[:, :, :2].min() + 0.05 * pattern[:, :, :2].min(),
                    pattern[:, :, :2].max() + 0.05 * pattern[:, :, :2].max(),
                ],
            )
            ax0.set_aspect("equal", adjustable="box")
            pp = ax0.scatter(
                pattern[frame, :, 0],
                pattern[frame, :, 1],
                s=5,
                c=gain[frame, :],
                cmap="gnuplot",
                vmin=gain.min(),
                vmax=gain.max(),
            )
            # alpha = (gain[frame, :] - gain.min()) / (gain.max() - gain.min())
            # pp.set_offsets((pattern[frame, :, 0], pattern[frame, :, 1]))
            # pp.set_alpha(alpha)
        else:
            ax0.clear()
            ax0.set(ylim=[0, pattern.max() + 0.05 * pattern.max()])
            pp = ax0.plot(az, pattern[frame, :])[0]
            # pp.set_data((az, pattern[frame, :]))
        dp.set_data((t, y_desired[:frame]))
        mp.set_data((t, y_mixed[:frame]))
        for i in range(N_ANT):
            wp[i].set_data((t, weights.real[i, :frame]))

    # create animation
    ani = FuncAnimation(fig=f, func=update, frames=N_FRAMES, interval=34)
    if show:
        plt.show()
    if save:
        nt.io.ensure_exist(FIGURES_PATH)
        ani.save(FIGURES_PATH / f"{filename}.gif")


def music_2d_plot(ant_xyz, P_music_dB: np.ndarray, D: int, resolution: float):
    az = np.arange(-180.0, 180.0, resolution)
    az_idx = P_music_dB.argsort()[::-1]

    f, ax = plt.subplots()
    for i in range(D):
        ax.vlines(x=az[az_idx[i]], ymin=0.0, ymax=P_music_dB[az_idx[i]], color="r", linestyle="--")
        plt.text(az[az_idx[i]], P_music_dB[az_idx[i]], f"Peak {i+1}: {az[az_idx[i]]} deg", fontsize=14)
    ax.plot(az, P_music_dB, "k", linewidth=2)
    ax.set(
        xlabel="Azimuth [deg]",
        ylabel="Power [dB]",
        title="2D MUSIC Power Plot",
        xticks=np.arange(-180, 180 + 45, 45),
    )

    return f, ax
