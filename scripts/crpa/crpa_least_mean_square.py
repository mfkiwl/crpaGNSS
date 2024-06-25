"""
|==================================== crpa_least_mean_square.py ===================================|
|                                                                                                  |
|   Property of Daniel Sturdivant. Unauthorized copying of this file via any medium would be       |
|   super sad and unfortunate for me. Proprietary and confidential.                                |
|                                                                                                  |
| ------------------------------------------------------------------------------------------------ |
|                                                                                                  |
|   @file     scripts/crpa_least_mean_square.py                                                    |
|   @brief    Basic CRPA visualization tools.                                                      |
|   @author   Daniel Sturdivant <sturdivant20@gmail.com>                                           |
|   @date     May 2024                                                                             |
|                                                                                                  |
|==================================================================================================|
"""

import numpy as np
import matplotlib.pyplot as plt
from charlizard.plotting.crpa_plots import polar_pattern_animation
from multiprocessing import freeze_support


# constants
TWO_PI = 2.0 * np.pi
LIGHT_SPEED = 299792458.0  #! [m/s]
T = 0.001
L = 301

# simulation parameters
DO_NULLS = True
FREQ = 60  #! [Hz]
WAVELENGTH = LIGHT_SPEED / FREQ  #! [m]
SAMP_FREQ = int(500.0 * FREQ)  #! [Hz]
SAMP_PER_MS = int(SAMP_FREQ * T)

# desired signal sources
SRC_AZ = np.deg2rad(335.0)
JAM_AZ = np.deg2rad(np.array([60.0, 220.0, 260.0]))
SRC_RANGE = 50.0 * WAVELENGTH
JAM_RANGE = np.array([5.0, 7.5, 10]) * WAVELENGTH
JAM_DOPPLER = np.array([-5.0, 2.5, 7.0]) * WAVELENGTH
REL_POWER = SRC_RANGE / JAM_RANGE


def band_limited_noise(f_min, f_max, n_samp, f_samp):
    freqs = np.abs(np.fft.fftfreq(n_samp, 1 / f_samp))
    f = np.zeros(n_samp)
    idx = np.where(np.logical_and(freqs >= f_min, freqs <= f_max))[0]
    f[idx] = 1

    f = np.array(f, dtype=complex)
    Np = (f.size - 1) // 2
    phase = np.random.rand(Np) * TWO_PI
    amp = np.cos(phase) + 1j * np.sin(phase)
    f[1 : Np + 1] *= amp
    f[-1 : -1 - Np : -1] = f[1 : Np + 1].conj()

    return np.fft.ifft(f)

if __name__ == '__main__':
    freeze_support()

    # linear antenna array and jammer locations
    src_xyz = SRC_RANGE * np.array([np.sin(SRC_AZ), np.cos(SRC_AZ), 0.0])
    jam_xyz = JAM_RANGE * np.array([np.sin(JAM_AZ), np.cos(JAM_AZ), np.zeros(JAM_AZ.shape)])
    ant_xyz = (
        np.array(
            [
                [0.0, WAVELENGTH, 0.0, WAVELENGTH],
                [0.0, 0.0, -WAVELENGTH, -WAVELENGTH],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        / 2
    )

    # simulated true sine wave and received sine waves
    t = np.arange(L * SAMP_PER_MS + 1) / SAMP_FREQ
    phase_src = np.linalg.norm(src_xyz - ant_xyz.T, axis=1) * (TWO_PI * FREQ / LIGHT_SPEED)
    phase_jam = np.zeros((4, jam_xyz.shape[1]))
    for i in range(jam_xyz.shape[1]):
        for j in range(4):
            phase_jam[j, i] = np.linalg.norm(jam_xyz[:, i] - ant_xyz[:, j]) * (TWO_PI * (FREQ + JAM_DOPPLER[i]) / LIGHT_SPEED)

    if DO_NULLS:
        y_desired = np.zeros(t.shape)
    else:
        y_desired = 2.0 * np.cos(TWO_PI * FREQ * t)
    sig_ant = np.cos(TWO_PI * FREQ * np.tile(t, (4, 1)) + phase_src[:, None])
    # bln = band_limited_noise(58, 62, 2 * t.size, SAMP_FREQ)
    # for i in range(4):
    #     idx1 =
    #     sig_ant[i, :] += bln[idx1:idx2]
    # for i in range(jam_xyz.shape[1]):
    #     sig_ant += REL_POWER[i] * np.cos(TWO_PI * (FREQ + JAM_DOPPLER[i]) * np.tile(t, (4, 1)) + phase_jam[:, i][:, None])
    # sig_ant += 0.5 * np.random.randn(sig_ant.shape[0], sig_ant.shape[1])

    # least mean squares parameters (initialize to weights of 0)
    mu = 1e-3
    W = np.zeros((4, L), dtype=float)
    W[0, 0] = 1
    y_mixed = np.zeros(y_desired.shape, dtype=float)

    # loop
    for k in range(L - 1):
        # grab sampled signal from each antenna
        X = sig_ant[:, SAMP_PER_MS * k : SAMP_PER_MS * (k + 1)]

        # mix sampled signals according to weights
        d_k = y_desired[SAMP_PER_MS * k : SAMP_PER_MS * (k + 1)]
        y_k = W[:, k].T @ X
        W[:, k + 1] = W[:, k] + mu * X @ (d_k - y_k)
        W[:, k + 1] = W[:, k + 1] * (W[0, k + 1] / np.abs(W[0, k + 1]))  # normalize

        # save output
        y_mixed[SAMP_PER_MS * k : SAMP_PER_MS * (k + 1)] = y_k

    # # plot signal result
    # f0, ax0 = plt.subplots()
    # ax0.plot(y_desired[: (L - 1) * SAMP_PER_MS : SAMP_PER_MS].real, label="Desired")
    # ax0.plot(y_mixed[: (L - 1) * SAMP_PER_MS : SAMP_PER_MS].real, label="Mixed")
    # # ax0.plot(sig_ant[0, : (L - 1) * SAMP_PER_MS : SAMP_PER_MS].real, label="Signal Ant. 1")
    # ax0.legend()
    # ax0.set(xlabel="Iteration Number", ylabel="Amplitude", title=f"Desired and Weighted Signal using mu={mu}")

    # # plot weights result
    # f1, ax1 = plt.subplots()
    # ax1.plot(W.T.real)
    # ax1.legend(["Weight 1", "Weight 2", "Weight 3", "Weight 4"])
    # ax1.set(xlabel="Iteration Number", ylabel="Weight Amplitude", title=f"Weight Iteration using \mu={mu}")

    # # scenario ENU
    # f2, ax2 = plt.subplots()
    # ax2.plot(ant_xyz[0, :], ant_xyz[1, :], "^")
    # ax2.plot(jam_xyz[0], jam_xyz[1], "s")
    # ax2.plot(src_xyz[0], src_xyz[1], "*")
    # ax2.set_aspect("equal", adjustable="box")

    # plt.show()

    # show
    polar_pattern_animation(
        ant_xyz=ant_xyz,
        weights=W,
        y_desired=y_desired[::SAMP_PER_MS].real,
        y_mixed=y_mixed[::SAMP_PER_MS].real,
        title=f"Least Mean Square Weighted Signal Output using mu={mu}",
        N_ANT=W.shape[0],
        WAVELENGTH=WAVELENGTH,
        N_FRAMES=L - 1,
        plot_3d=True,
        show=False,
        save=True,
        filename="test_crpa_lms_null_test_3d",
    )
