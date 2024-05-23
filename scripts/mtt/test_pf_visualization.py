"""
|===================================== test_visualization.py ======================================|
|                                                                                                  |
|   Property of Daniel Sturdivant. Unauthorized copying of this file via any medium would be       |
|   super sad and unfortunate for me. Proprietary and confidential.                                |
|                                                                                                  |
| ------------------------------------------------------------------------------------------------ |
|                                                                                                  |
|   @file     scripts/test_visualization.py                                                        |
|   @brief    Test script for plotting TDoA and AoA measurements on a heat map. This script has    | 
|             no relationship to the rest of the package.                                          |
|   @author   Daniel Sturdivant <sturdivant20@gmail.com>                                           |
|   @date     December 2023                                                                        |
|                                                                                                  |
|==================================================================================================|
"""

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import freeze_support, pool
from tqdm import tqdm

R2D = 180 / np.pi
TWO_PI = 2 * np.pi


def combinations(v):
    n = v.size
    c0 = np.zeros(2 ** (n - 1) - 1, dtype=np.int32)
    c1 = np.zeros(2 ** (n - 1) - 1, dtype=np.int32)
    a = 0
    for i in range(n):
        for j in range(i + 1, n):
            c0[a] = v[i]
            c1[a] = v[j]
            a += 1
    return c0, c1


def likelihood(x, mu, sigma):
    return np.exp(-((x - mu) ** 2) / (2 * sigma * sigma)) / (np.sqrt(TWO_PI) * sigma)


def for_loop(*args):
    args = args[0]
    x, y, res, i, aoa, rdoa, receivers = args

    dN_ij = x - receivers[:, 1]
    for j in range(y.size):
        dE_ij = y[j] - receivers[:, 0]
        R_ij = np.sqrt(dE_ij**2 + dN_ij**2)
        aoa_ij = np.arctan2(dE_ij, dN_ij) * R2D
        rdoa_ij = R_ij[c0] - R_ij[c1]
        for k in range(aoa.size):
            res[j] += likelihood(aoa[k] - aoa_ij[k % m], 0.0, aoa_std)
        for k in range(rdoa.size):
            res[j] += likelihood(rdoa[k] - rdoa_ij[k % m], 0.0, rdoa_std)
    return res, i


# --------------------------------------------------------------------------------------------------#
if __name__ == "__main__":
    freeze_support()

    # * --- emitter and receiver locations (ENU) ---
    emitters = np.array([[-10.1, 99.9], [999, 999]], dtype=np.float64)
    receivers = np.array([[850, -250], [400, 950], [-900, -600]], dtype=np.float64)

    # truth measurements
    n = emitters.shape[0]
    m = receivers.shape[0]
    c0, c1 = combinations(np.arange(m, dtype=np.int32))
    aoa = np.zeros(n * m, dtype=np.float64)  #! degrees
    rdoa = np.zeros(n * m, dtype=np.float64)  #! meters
    for i in range(n):
        dE = emitters[i, 0] - receivers[:, 0]
        dN = emitters[i, 1] - receivers[:, 1]
        R = np.sqrt(dE * dE + dN * dN)
        aoa[i * m : (i + 1) * m] = np.arctan2(dE, dN) * R2D
        rdoa[i * m : (i + 1) * m] = R[c0] - R[c1]

    # error model
    aoa_std = 5  #! degrees
    rdoa_std = 40e-9 * 299792458  #! meters

    # * --- particle loactions ---
    x = np.arange(-1000, 1000, 0.5)
    y = np.arange(-1000, 1000, 0.5)
    L = x.size + y.size

    res = np.zeros((x.size, y.size), dtype=np.float64)
    with pool.Pool(processes=14) as p:
        args = [(x[i], y, res[i, :], i, aoa, rdoa, receivers) for i in range(x.size)]
        for r, j in tqdm(p.imap(for_loop, args), total=x.size, ascii=".>#", bar_format="{l_bar}{bar:50}{r_bar}"):
            res[j, :] = r

    # surface plot
    X, Y = np.meshgrid(x, y)
    f = plt.figure()
    ax = f.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, res, cmap="jet")
    plt.show()

print()
