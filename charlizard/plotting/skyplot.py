"""
|========================================== skyplot.py ============================================|
|                                                                                                  |
|  Property of Daniel Sturdivant. Unauthorized copying of this file via any medium would be        |
|  super sad and unfortunate for me. Proprietary and confidential.                                 |
|                                                                                                  |
|--------------------------------------------------------------------------------------------------| 
|                                                                                                  |
|  @file     charlizard/plotting/skyplot.py                                                        |
|  @brief    Sky plot.                                                                             |
|  @author   Daniel Sturdivant <sturdivant20@gmail.com>                                            | 
|  @date     December 2023                                                                         |
|                                                                                                  |
|==================================================================================================|
"""

import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Iterable
import matplotlib.patheffects as pe


# # --- SKYPLOT ---
# # creates a polar plot given azimuth and elevation
# def skyplot(az, el, *args):
#     # Inputs:
#     #   az        (np.ndarray)  size NxM array of azimuth angles in degrees
#     #   el        (np.ndarray)  size NxM array of elevation angles in degrees
#     #   name      (list)        size N string list of names (optional)
#     #   color     (str)         axes line color (optional)
#     #   ax        (axes)        figure plot axes (optional)
#     #
#     # Outputs:
#     #

#     # determine number of channels
#     n = az.shape[0]

#     # check for optional arguments 'name' and 'line_style'
#     if len(args) == 1:
#         name = args[0]
#         color = "green"
#         ax_ = None
#     elif len(args) == 2:
#         name, color = args
#         ax_ = None
#     elif len(args) == 3:
#         name, color, ax_ = args
#     else:
#         name = np.array(map(str, np.arange(n) + 1))
#         color = "green"
#         ax_ = None

#     # set up polar plot
#     if ax_ is None:
#         f = plt.figure()
#         ax = f.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
#     else:
#         ax = ax_
#     ax.set_theta_zero_location("N")
#     ax.set_theta_direction(-1)

#     if len(az.shape) > 1:
#         for ii in np.arange(az.shape[1]):
#             # plot satellite trajectory
#             ax.plot(np.deg2rad(az[:, ii]), 90.0 - el[:, ii], color=color)

#             # plot start point
#             ax.plot(np.deg2rad(az[0, ii]), 90.0 - el[0, ii], color=color, marker="*", markersize=7.0)

#             # plot name plate
#             ax.annotate(
#                 name[ii],
#                 xy=(np.deg2rad(az[-1, ii]), 90.0 - el[-1, ii]),
#                 bbox=dict(boxstyle="round", fc=color, alpha=0.5),
#                 horizontalalignment="center",
#                 verticalalignment="center",
#                 size=8,
#             )

#     else:
#         # plot only name plate
#         for ii in np.arange(az.shape[0]):
#             ax.annotate(
#                 name[ii],
#                 xy=(np.deg2rad(az[ii]), 90.0 - el[ii]),
#                 bbox=dict(boxstyle="round", fc=color, alpha=0.3),
#                 horizontalalignment="center",
#                 verticalalignment="center",
#             )

#     ax.set_yticks(range(0, 90 + 10, 10))
#     ax.set_yticklabels(["90", "", "", "60", "", "", "30", "", "", ""])
#     ax.grid(True)

#     return f, ax


def skyplot(
    az: np.ndarray,
    el: np.ndarray,
    name: str | list = None,
    deg: bool = True,
    fig: plt.Figure = None,
    ax: plt.Axes = None,
    **kwargs
):
    # if isinstance(plt.gca(), plt.PolarAxes):
    #     ax = plt.gca()
    # else:
    #     plt.close()
    #     fig = plt.gcf()
    #     ax = fig.add_subplot(projection="polar")
    if fig == None and ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(projection="polar")

    if deg:
        az = np.radians(az)
    else:
        el = np.degrees(el)

    # format polar axes
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlim(91, 1)

    degree_sign = "\N{DEGREE SIGN}"
    r_labels = [
        # "0" + degree_sign,
        "",
        "",
        "30" + degree_sign,
        "",
        "60" + degree_sign,
        "",
        "90" + degree_sign,
    ]
    ax.set_rgrids(range(1, 106, 15), r_labels, angle=22.5)

    ax.set_axisbelow(True)

    # plot
    ax.scatter(az, el, **kwargs)

    # annotate object names
    if name is not None:
        if not isinstance(name, Iterable):
            name = (name,)

        for obj, n in enumerate(name):
            ax.annotate(
                n,
                (az[obj, 0], el[obj, 0]),
                fontsize="x-small",
                path_effects=[pe.withStroke(linewidth=3, foreground="w")],
            )

    ax.figure.canvas.draw()

    return fig, ax
