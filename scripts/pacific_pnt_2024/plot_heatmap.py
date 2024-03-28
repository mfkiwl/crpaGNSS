"""
|========================================= plot_heatmap.py ========================================|
|                                                                                                  |
|   Property of Daniel Sturdivant. Unauthorized copying of this file via any medium would be       |
|   super sad and unfortunate for me. Proprietary and confidential.                                |
|                                                                                                  |
| ------------------------------------------------------------------------------------------------ |
|                                                                                                  |
|   @file     scripts/plot_heatmap.py                                                              |
|   @brief    Plot heatmap of buoy positions and GDOPs for January 1, 2024 at 12:00 UTC.           |
|   @author   Daniel Sturdivant <sturdivant20@gmail.com>                                           |
|   @date     March 2024                                                                           |
|                                                                                                  |
|==================================================================================================|
"""

import numpy as np
from scipy.linalg import norm, inv, pinv
from multiprocessing import Pool, freeze_support, cpu_count
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.basemap import Basemap
import seaborn as sns

import navtools as nt

PROJECT_PATH = Path(__file__).parents[2]
RESULTS_PATH = PROJECT_PATH / "results" / "pacific_pnt"
FIGURES_PATH = RESULTS_PATH / "figures" / "heatmaps"
nt.io.ensure_exist(FIGURES_PATH)

SAVE = True
DEG_SPACING = 1
ALTITUDE = 11000
D2R = np.pi / 180
LLA_D2R = np.array([D2R, D2R, 1.0])

# create custom color map
cdict = {
    "red": (
        (0.0, 32 / 255, 32 / 255),
        (0.01, 64 / 255, 64 / 255),
        (1 / 5 / 1.5 * 1, 112 / 255, 112 / 255),
        (1 / 4 / 1.5 * 2, 230 / 255, 230 / 255),
        (1 / 4.5 / 1.5 * 3, 253 / 255, 253 / 255),
        (1 / 4 / 1.5 * 4, 244 / 255, 244 / 255),
        (1.0, 169 / 255, 169 / 255),
    ),
    "green": (
        (0.0, 42 / 255, 42 / 255),
        (0.01, 57 / 255, 57 / 255),
        (1 / 5 / 1.5 * 1, 198 / 255, 198 / 255),
        (1 / 4 / 1.5 * 2, 241 / 255, 241 / 255),
        (1 / 4.5 / 1.5 * 3, 219 / 255, 219 / 255),
        (1 / 4 / 1.5 * 4, 109 / 255, 109 / 255),
        (1.0, 23 / 255, 23 / 255),
    ),
    "blue": (
        (0.0, 68 / 255, 68 / 255),
        (0.01, 144 / 255, 144 / 255),
        (1 / 5 / 1.5 * 1, 162 / 255, 162 / 255),
        (1 / 4 / 1.5 * 2, 146 / 255, 146 / 255),
        (1 / 5 / 1.5 * 3, 127 / 255, 127 / 255),
        (1 / 4 / 1.5 * 4, 69 / 255, 69 / 255),
        (1.0, 69 / 255, 69 / 255),
    ),
}
cmap = colors.LinearSegmentedColormap("new_cmap", segmentdata=cdict)


def for_loop(*x):
    x = x[0]
    lats, lon, alt, buoy_ecefs, i = x

    n_buoys = np.zeros(lats.size, dtype=int)
    gdop = 1e4 * np.ones(lats.size, dtype=float)
    vdop = 1e4 * np.ones(lats.size, dtype=float)
    hdop = 1e4 * np.ones(lats.size, dtype=float)

    for j, lat in enumerate(lats):
        user_lla = np.array([lat, lon, alt]) * LLA_D2R
        user_pos = nt.lla2ecef(user_lla)

        # record buoy positions available
        buoy_pos = []
        for buoy_ecef in buoy_ecefs:
            buoy_p = buoy_ecef
            _, el, _ = nt.ecef2aer(user_pos, buoy_p)

            if (el / D2R) > 0.0:
                n_buoys[j] += 1
                buoy_pos.append(buoy_p)

        # calculate GDOP of available buoys
        if n_buoys[j] > 0:
            buoy_pos = np.array(buoy_pos)
            dr = user_pos - buoy_pos
            u = dr / norm(dr, axis=1)[:, None]
            H = np.column_stack((u, np.ones(u.shape[0])))
            C_e_n = nt.ecef2enuDcm(user_lla)
            try:
                DOP = inv(H.T @ H)
                DOP[:3, :3] = C_e_n @ DOP[:3, :3] @ C_e_n.T
                gdop[j] = np.sqrt(np.abs(DOP.trace()))
                vdop[j] = np.sqrt(np.abs(DOP[2, 2]))
                hdop[j] = np.sqrt(np.abs(DOP[:2, :2].trace()))
            except:
                continue

    return n_buoys, gdop, vdop, hdop, i


def scatter_plot(buoy_llas: np.ndarray, save: bool = True):
    """generate a scatter plot of buoy locations"""

    f, ax = plt.subplots(**{"figsize": (10, 5)})
    # m = Basemap(projection='robin',lat_0=0, lon_0=0, resolution='c')
    m = Basemap(
        ax=ax,
        projection="cyl",
        lat_0=0,
        lon_0=0,
        resolution="c",
        llcrnrlat=-90,
        urcrnrlat=90,
        llcrnrlon=-180,
        urcrnrlon=180,
    )
    x, y = m(buoy_llas[:, 1], buoy_llas[:, 0])

    # fill in continents and draw their border (maybe thicker borders??)
    m.fillcontinents(color="gray", lake_color="white")
    m.drawcoastlines(linewidth=1.0, color="k")
    m.drawmeridians(meridians=np.arange(-180, 200, 20), color="k", linewidth=0.5)
    m.drawparallels(circles=np.arange(-90, 90, 10), color="k", linewidth=0.5)

    sns.scatterplot(x=x, y=y, ax=ax, s=12)
    ax.set_aspect("equal", "box")
    f.tight_layout()
    if save:
        f.savefig(FIGURES_PATH / "buoy_scatterplot.jpeg", dpi=300)


def dop_plot(lats: np.ndarray, lons: np.ndarray, alt: float, dop: np.ndarray, label: str, save: bool = True):
    """generate a plot of buoy gdop"""

    f, ax = plt.subplots(**{"figsize": (12, 5), "edgecolor": "w"})
    # m = Basemap(projection="robin", lat_0=0, lon_0=0, resolution="c")
    m = Basemap(
        ax=ax,
        projection="cyl",
        lat_0=0,
        lon_0=0,
        resolution="c",
        llcrnrlat=-90,
        urcrnrlat=90,
        llcrnrlon=-180,
        urcrnrlon=180,
    )
    x, y = m(*np.meshgrid(lons, lats))

    # fill in continents and draw their border (maybe thicker borders??)
    m.fillcontinents(color="gray", lake_color="white")
    m.drawcoastlines(linewidth=1.0, color="w")
    m.drawmeridians(meridians=np.arange(-180, 200, 20), color="w", linewidth=0.5)
    m.drawparallels(circles=np.arange(-90, 90, 10), color="w", linewidth=0.5)

    # heatmap
    map = m.contourf(x=x, y=y, data=dop, cmap=cmap, levels=np.arange(-0.1, 100.1, 0.1), extend="both")
    map.cmap.set_over((150 / 255, 20 / 255, 59 / 255))
    map.cmap.set_under((68 / 255, 42 / 255, 32 / 255))
    map.set_clim(-0.1, 100.1)

    # legend / colorbar / colormap
    cbar = m.colorbar(map, label=label, ticks=[0, 5, 10, 25, 50, 75, 100])
    cbar.set_ticklabels(("0", "5", "10", "25", "50", "75", "100+"))

    ax.set_aspect("equal", "box")
    f.tight_layout()
    if save:
        f.savefig(FIGURES_PATH / f"buoy_{label.lower()}_{int(alt)}m.jpeg", dpi=300)


def num_buoys_plot(lats: np.ndarray, lons: np.ndarray, alt: float, n_buoys: np.ndarray, save: bool = True):
    """generate a plot of number of buoys"""

    f, ax = plt.subplots(**{"figsize": (12, 5)})
    # m = Basemap(projection='robin',lat_0=0, lon_0=0, resolution='c')
    m = Basemap(
        ax=ax,
        projection="cyl",
        lat_0=0,
        lon_0=0,
        resolution="c",
        llcrnrlat=-90,
        urcrnrlat=90,
        llcrnrlon=-180,
        urcrnrlon=180,
    )
    x, y = m(*np.meshgrid(lons, lats))

    # fill in continents and draw their border (maybe thicker borders??)
    m.fillcontinents(color="gray", lake_color="white")
    m.drawcoastlines(linewidth=1.0, color="w")
    m.drawmeridians(meridians=np.arange(-180, 200, 20), color="w", linewidth=0.5)
    m.drawparallels(circles=np.arange(-90, 90, 10), color="w", linewidth=0.5)

    # heatmap
    map = m.contourf(x=x, y=y, data=n_buoys, cmap=cmap, levels=np.arange(-0.1, 25.1, 0.1), extend="both")
    map.cmap.set_over((150 / 255, 20 / 255, 59 / 255))
    map.cmap.set_under((68 / 255, 42 / 255, 32 / 255))
    map.set_clim(-0.1, 25.1)

    # legend / colorbar / colormap
    cbar = m.colorbar(map, label="Amount of Visible Buoys", ticks=[0, 3, 7, 15, 25])
    cbar.set_ticklabels(("0", "3", "8", "15", "25+"))

    ax.set_aspect("equal", "box")
    f.tight_layout()
    if save:
        f.savefig(FIGURES_PATH / f"buoy_amount_{int(alt)}m.jpeg", dpi=300)


if __name__ == "__main__":
    freeze_support()

    # set seaborn variables and turn on grid
    sns.set_theme(
        font="Times New Roman",
        context="talk",
        palette="Set1",
        style="ticks",
        rc={"axes.grid": True},
    )

    # parse buoy data
    start_t = datetime(2024, 1, 1, 12, 0, 0)
    stop_t = datetime(2024, 1, 1, 12, 0, 1)
    bp = nt.parsers.buoy.BuoyParser(start_time=start_t, stop_time=stop_t)
    bp.grab_data()
    buoys = bp.grab_emitters()

    # extract buoy position data
    buoy_lla = np.zeros((buoys.shape[0], 3))
    buoy_ecef = np.zeros((buoys.shape[0], 3))
    for k, buoy in enumerate(buoys):
        buoy_ecef[k, :] = buoy.at(start_t)[0]
        buoy_lla[k, :] = nt.ecef2lla(buoy_ecef[k, :]) / LLA_D2R

    # clear memory
    del start_t, stop_t, bp, buoys

    # extract number of buoys and gdop data
    lats = np.arange(-90.0, 90.0 + DEG_SPACING, DEG_SPACING)
    lons = np.arange(-180.0, 180.0 + DEG_SPACING, DEG_SPACING)
    with Pool(processes=cpu_count()) as p:
        args = [(lats, lon, ALTITUDE, buoy_ecef, i) for i, lon in enumerate(lons)]

        n_buoys = np.zeros((lats.size, lons.size))
        gdop = np.zeros((lats.size, lons.size))
        vdop = np.zeros((lats.size, lons.size))
        hdop = np.zeros((lats.size, lons.size))
        for n, g, v, h, i in tqdm(
            p.imap(for_loop, args),
            total=lons.size,
            desc="[charlizard] calculting buoy availability ",
            ascii=".>#",
            bar_format="{desc}{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{rate_fmt}]",
            ncols=120,
        ):
            n_buoys[:, i] = n
            gdop[:, i] = g
            vdop[:, i] = v
            hdop[:, i] = h

    # scatter plot of buoy positions
    scatter_plot(buoy_lla, SAVE)
    dop_plot(lats, lons, ALTITUDE, gdop, "GDOP", SAVE)
    dop_plot(lats, lons, ALTITUDE, vdop, "VDOP", SAVE)
    dop_plot(lats, lons, ALTITUDE, hdop, "HDOP", SAVE)
    num_buoys_plot(lats, lons, ALTITUDE, n_buoys, SAVE)

    plt.show()
    print()
