"""
|========================================== crpa_music.py =========================================|
|                                                                                                  |
|   Property of Daniel Sturdivant. Unauthorized copying of this file via any medium would be       |
|   super sad and unfortunate for me. Proprietary and confidential.                                |
|                                                                                                  |
| ------------------------------------------------------------------------------------------------ |
|                                                                                                  |
|   @file     scripts/crpa_music.py                                                                |
|   @brief    Basic CRPA visualization tools.                                                      |
|   @author   Daniel Sturdivant <sturdivant20@gmail.com>                                           |
|   @date     May 2024                                                                             |
|                                                                                                  |
|==================================================================================================|
"""

import numpy as np
from scipy.linalg import inv, norm, eig
import navtools as nt
import matplotlib.pyplot as plt
from charlizard.plotting.crpa_plots import music_2d_plot

LIGHT_SPEED = 299792458.0  #! [m/s]
FREQUENCY = 1575.42e6  #! [Hz]
CHIP_FREQ = 1.023e6  #! [chips/s]
WAVELENGTH = LIGHT_SPEED / FREQUENCY  #! [m]
CHIP_WIDTH = LIGHT_SPEED / CHIP_FREQ  #! [m/chip]

R2D = 180 / np.pi
LLA_R2D = np.array([R2D, R2D, 1], dtype=float)


##### *SETUP* #####
CN0 = 50
TRUE_AZ = np.array([30.0, -44.0])
RESOLUTION = 0.5

# ant_xyz = (
#     np.array(
#         [
#             [0, 0, 0],
#             [WAVELENGTH, 0, 0],
#             [0, -WAVELENGTH, 0],
#             [WAVELENGTH, -WAVELENGTH, 0],
#         ],
#         dtype=float,
#     )
#     / 2.0
# )
# true_u = np.array([np.sin(TRUE_AZ / R2D), np.cos(TRUE_AZ / R2D), np.zeros(2)])
# spatial_phase = np.array([ant_xyz @ true_u[:, 0], ant_xyz @ true_u[:, 1]]).T

# ##### *CORRELATORS* #####
# # t = np.linspace(0, 0.001, 5000)
# # f = 1.0
# # X = np.exp(1j * 2 * np.pi * (f * t[:, None, None] + np.tile(spatial_phase, (5000, 1, 1))))
# # X = (X[:, :, 0] * X[:, :, 1]).T
# # # X = X[:, :, 0].T
# # X += np.random.randn(X.shape[0], X.shape[1])

# X = np.zeros(spatial_phase.shape[0], dtype=complex)
# phase_err = 0.2 * np.random.randn(2)[None, :] - spatial_phase / WAVELENGTH
# chip_err = 0.02 * np.random.randn(2)[None, :] - spatial_phase / CHIP_WIDTH
# freq_err = 1.0 * np.random.randn(2)[None, :] + spatial_phase / WAVELENGTH * 0.02

# A = np.sqrt(2.0 * 10 ** (CN0 / 10) * 0.02)
# R = 1.0 - np.abs(chip_err)
# # R[R <= 0.0] = 0.0
# F = np.sinc(np.pi * freq_err * 0.02)
# P = np.exp(1j * 2.0 * np.pi * phase_err)
# tmp = A * R * F * P
# tmp = tmp[:, 1]
# X.real = tmp.real + np.random.randn(X.size)
# X.imag = tmp.imag + np.random.randn(X.size)

# ##### *MUSIC* #####
# M = X.shape[0]
# S = np.outer(X, X.conj()) / M  #! MxM covariance of X, Eq. 2
# # S = X @ X.conj().T / M
# e, v = eig(S)
# idx = np.abs(e) < (e.max() / 10)
# N = int(np.sum(idx))
# D = M - N  #! number of incident signals, Eq. 5
# U = v[:, idx]  #! null space of S
# az = np.arange(-180.0, 180.0, RESOLUTION)
# P_music = np.zeros(az.size, dtype=float)
# for i in range(az.size):
#     u = np.array([np.sin(az[i] / R2D), np.cos(az[i] / R2D), 0.0])
#     a = np.exp(-1j * 2.0 * np.pi * (ant_xyz @ u) / WAVELENGTH)
#     P_music[i] = 1.0 / np.abs((a.conj().T @ U @ U.conj().T @ a))  #! 1 / euclidean distance, Eq. 6
# P_music_dB = 10 * np.log10(P_music)

# ##### *PLOT* #####
# f, ax = music_2d_plot(ant_xyz, P_music_dB, D, RESOLUTION)
# plt.show()

sv_pos = np.array(
    [
        [13392805.9056497, -17121749.1891207, 14687559.3031646],
        [6558025.92118539, -25313733.9460521, -2308000.09537896],
        [20600200.9715807, 1611189.36004887, 16840991.8522564],
        [-20821967.3849201, -10248798.8462802, 12920961.314036],
        [-2448335.17445519, -15382265.0421222, 21511575.4267938],
        [-10294441.3786277, -21482990.7102644, 12067578.1476593],
        [-17309673.391087, -20246903.7228924, 1775667.14427449],
        [15227770.6795543, -8130788.44020159, 20525152.692794],
        [-2087486.0955587, -24668403.7414645, 9352392.52854125],
    ]
)
sv_vel = np.array(
    [
        [-221.056287643381, 1926.03974518703, 2428.7703846003],
        [285.050999297152, 412.141699137332, -3166.81857932004],
        [1552.20278533067, 1475.5304691025, -1993.34893533392],
        [-949.929972800083, -1375.24282607976, -2567.41888531236],
        [2620.51038720661, -784.798306884304, -276.308652992449],
        [1184.57768347481, 886.214757928628, 2694.25088782723],
        [329.26144599767, 35.7128933308745, 3176.42832842922],
        [-124.471987206527, 2503.56379837612, 1198.44988206422],
        [569.395249749801, -1095.9558367451, -2819.61004720292],
    ]
)

ant_pos_body = (
    np.array(
        [
            [0, 0, 0],
            [WAVELENGTH, 0, 0],
            [0, -WAVELENGTH, 0],
            [WAVELENGTH, -WAVELENGTH, 0],
        ],
        dtype=float,
    )
    / 2.0
)
ant_att_rpy = np.array([10, -5, 30], dtype=float)
user_ref_lla = np.array([32.586279, -85.494372, 194.83]) / LLA_R2D


# get ecef ant pos
ant_pos_ecef = np.zeros(ant_pos_body.shape)
C_b_n = nt.euler2dcm(ant_att_rpy / R2D, "enu").T
for i in range(ant_pos_ecef.shape[0]):
    ant_pos_ecef[i, :] = nt.enu2ecef(C_b_n @ ant_pos_body[i, :], user_ref_lla)


# generate prompt correlators relative to reference antenna
r = np.zeros((sv_vel.shape[0], ant_pos_ecef.shape[0]))
rdot = np.zeros((sv_vel.shape[0], ant_pos_ecef.shape[0]))
u = np.zeros((sv_vel.shape[0], 3, ant_pos_ecef.shape[0]))
true_az = np.zeros((sv_vel.shape[0], ant_pos_ecef.shape[0]))
true_el = np.zeros((sv_vel.shape[0], ant_pos_ecef.shape[0]))
for i in range(ant_pos_ecef.shape[0]):
    dr = ant_pos_ecef[i, :] - sv_pos
    dv = np.zeros(3) - sv_vel
    r[:, i] = norm(dr, axis=1)
    u[:, :, i] = dr / r[:, i][:, None]
    rdot[:, i] = np.sum(u[:, :, i] * dv, axis=1)
    for j in range(sv_vel.shape[0]):
        true_az[j, i], true_el[j, i], _ = nt.ecef2aer(sv_pos[j, :], ant_pos_ecef[i, :]) * R2D
true_rng = r[:, 0]
true_rng_rate = rdot[:, 0]

# correlators
chip_err = ((r - true_rng[:, None]) / CHIP_WIDTH).T
phase_err = ((r - true_rng[:, None]) / WAVELENGTH).T
freq_err = (-(rdot - true_rng_rate[:, None]) / WAVELENGTH).T
X = np.zeros((ant_pos_ecef.shape[0], sv_vel.shape[0]), dtype=complex)
for i in range(ant_pos_ecef.shape[0]):
    A = np.sqrt(2.0 * 10 ** (CN0 / 10) * 0.02)
    R = 1.0 - np.abs(chip_err[i])
    R[R <= 0.0] = 0.0
    F = np.sinc(np.pi * freq_err[i] * 0.02)
    P = np.exp(1j * 2.0 * np.pi * phase_err[i])
    X[i].real = A * R * F * P.real + np.random.randn()
    X[i].imag = A * R * F * P.imag + np.random.randn()

print(u.shape)
print(X[:, 0])
print(phase_err[:, 0])


##### *MUSIC* ##### (Schmidt - Multiple Emitter Location and Signal Parameter Estimation)
L = X.shape[1]
M = ant_pos_ecef.shape[0]  #! number of antennas
D = np.zeros(L)
az_est = np.zeros(L)
el_est = np.zeros(L)
for l in range(L):
    # 0) collect data, form S
    S = np.outer(X[:, l], X[:, l].conj()) / M  #! MxM covariance of X, Eq. 2

    # 1) calculate eigenstructure of S in metric of S0
    e, v = eig(S)
    idx = np.abs(e) < 1.0

    # 2) decide number of signals D
    N = np.sum(idx)  #! repeated min eigenvalue = 0, Eq. 5
    D = M - N  #! number of incident signals, Eq. 5

    # 3) evaluate P_music vs. arrival angle
    U = v[:, idx]
    a = np.zeros(M, dtype=complex)
    resolution = 10.0
    az_span = 180.0
    el_span = 90.0
    az_mean = 0.0
    el_mean = 0.0
    while resolution > 0.001:
        az = np.arange(az_mean - az_span, az_mean + az_span, resolution)
        el = np.arange(el_mean - el_span, el_mean + el_span, resolution)
        P_music = np.zeros((el.size, az.size), dtype=float)

        for i in range(az.size):
            for j in range(el.size):
                u = np.array(
                    [
                        np.sin(az[i] / R2D) * np.cos(el[j] / R2D),
                        np.cos(az[i] / R2D) * np.cos(el[j] / R2D),
                        np.sin(el[j] / R2D),
                    ],
                )
                for k in range(M):
                    a[k] = np.exp(-1j * 2.0 * np.pi * (ant_pos_body[k, :] @ u) / WAVELENGTH)
                P_music[j, i] = 1.0 / np.abs((a.conj().T @ U @ U.conj().T @ a))

        el_idx, az_idx = np.unravel_index(P_music.argmax(), P_music.shape)
        az_mean = az[az_idx]
        el_mean = el[el_idx]
        az_span = resolution / 2
        el_span = resolution / 2
        resolution /= 10

    az_est[l] = az_mean
    el_est[l] = el_mean

# 4) pick D peaks of P_music
# el_est = -el_est
print(f"CN0 = {CN0} dB\n")
print(f"true_az = {np.round(true_az[:,0],2)}")
print(f"est_az  = {az_est} \n")
print(f"true_el = {np.round(true_el[:,0],2)}")
print(f"est_el  = {el_est} \n")

# 5) calculate remaining parameters
R_inv = np.linalg.inv(1.0 / R2D * np.eye(L))
D_loc = np.array(
    [
        np.sin(az_est / R2D) * np.cos(el_est / R2D),
        np.cos(az_est / R2D) * np.cos(el_est / R2D),
        np.sin(el_est / R2D),
    ],
)
D_enu = np.array(
    [
        np.sin(true_az[:, 0] / R2D) * np.cos(true_el[:, 0] / R2D),
        np.cos(true_az[:, 0] / R2D) * np.cos(true_el[:, 0] / R2D),
        np.sin(true_el[:, 0] / R2D),
    ],
)

M_hat = D_loc @ R_inv @ D_enu.T @ np.linalg.inv(D_enu @ R_inv @ D_enu.T)
rpy = (
    np.array(
        [np.arctan(M_hat[2, 1] / M_hat[2, 2]), np.arcsin(M_hat[2, 0]), np.arctan2(M_hat[0, 0], M_hat[1, 0])],
    )
    * R2D
)

print(f"True Attitude = {np.round(ant_att_rpy,2)}")
print(f"Est. Attitude = {np.round(rpy,2)} \n")
