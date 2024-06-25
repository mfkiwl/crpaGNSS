"""
GNSS - Global Navigation Satellite Systems: GPS, Galileo, and more. 2007
"""

import numpy as np

SPEED_OF_LIGHT = 299792458.0  #! [m/s]
FREQUENCY = 1575.42e6  #! [Hz]
WAVELENGTH = SPEED_OF_LIGHT / FREQUENCY  #! [m]
R2D = 180 / np.pi
D2R = np.pi / 180

N_ANT = 4  #! number of elements
RADIAL_DIST = WAVELENGTH / 2  #! [m] linear separation
DEG_SEP = 360 / 4  #! [deg] angular separation

# --------------------------------------------------------------------------------------------------#
rpy_truth = np.array([1.0, -1.0, 35.0])  # rpy [rad]
sr, sp, sy = np.sin(rpy_truth * D2R)
cr, cp, cy = np.cos(rpy_truth * D2R)

# M_truth = np.array([[cr*cy - sr*sp*sy, cr*sy + sr*sp*cy, -sr*cp],                 #! Eq 13.6
#                     [          -cp*sy,            cp*cy,     sp],
#                     [sr*cy + cr*sp*sy, sr*sy - cr*sp*cy,  cr*cp]])
# r = R2D * np.arctan(-M_truth[0,2] / M_truth[2,2])                                 #! Eq 13.13
# # p = R2D * np.arctan(M_truth[1,2] / np.sqrt(M_truth[1,0]**2 + M_truth[1,1]**2))
# p = R2D * np.arcsin(M_truth[1,2])
# y = R2D * np.arctan(-M_truth[1,0] / M_truth[1,1])

M_truth = np.array(
    [
        [sy * cp, cr * cy + sr * sy * sp, -sr * cy + cr * sy * sp],
        [cy * cp, -cr * sy + sr * cy * sp, sr * sy + cr * cy * sp],
        [sp, -sr * cp, -cr * cp],
    ]
)
r = R2D * np.arctan(M_truth[2, 1] / M_truth[2, 2])
# p = R2D * np.arctan(M_truth[2,0] / np.sqrt(M_truth[0,0]**2 + M_truth[1,0]**2))
p = R2D * np.arcsin(M_truth[2, 0])
y = R2D * np.arctan(M_truth[0, 0] / M_truth[1, 0])

print(f"Roll = {r}, Pitch = {p} , yaw = {y}")

# --------------------------------------------------------------------------------------------------#
r2 = []
p2 = []
y2 = []

truth_user_enu = np.zeros(3)
truth_sv_enu = np.array(
    [[500, 1500, -2000, 3500, -3000], [-500, -3000, 1500, -2000, 3500], [6100, 5000, 5500, 4750, 4200]]
)

for _ in range(1000):
    rpy_noise = rpy_truth + 3 * np.random.randn(1)  #! std of 3 deg
    sr, sp, sy = np.sin(rpy_noise * D2R)
    cr, cp, cy = np.cos(rpy_noise * D2R)

    # M_noise = np.array([[cr*cy - sr*sp*sy, cr*sy + sr*sp*cy, -sr*cp],           #! Eq 13.6
    #                     [          -cp*sy,            cp*cy,     sp],
    #                     [sr*cy + cr*sp*sy, sr*sy - cr*sp*cy,  cr*cp]])
    M_noise = np.array(
        [
            [sy * cp, cr * cy + sr * sy * sp, -sr * cy + cr * sy * sp],
            [cy * cp, -cr * sy + sr * cy * sp, sr * sy + cr * cy * sp],
            [sp, -sr * cp, -cr * cp],
        ]
    )

    truth_user_body = M_noise @ truth_user_enu
    truth_sv_body = M_noise @ truth_sv_enu

    R_inv = np.linalg.inv(D2R * 3 * np.eye(5))
    D_enu = (truth_sv_enu - truth_user_enu[:, None]) / np.linalg.norm(truth_sv_enu - truth_user_enu[:, None], axis=0)
    D_loc = (truth_sv_body - truth_user_body[:, None]) / np.linalg.norm(
        truth_sv_body - truth_user_body[:, None], axis=0
    )

    M_hat = D_loc @ R_inv @ D_enu.T @ np.linalg.inv(D_enu @ R_inv @ D_enu.T)

    # r2.append(R2D * np.arctan(-M_hat[0,2] / M_hat[2,2]))                        #! Eq 13.13
    # # p2 = R2D * np.arctan(M_hat[1,2] / np.sqrt(M_hat[1,0]**2 + M_hat[1,1]**2))
    # p2.append(R2D * np.arcsin(M_hat[1,2]))
    # y2.append(R2D * np.arctan(-M_hat[1,0] / M_hat[1,1]))

    r2.append(R2D * np.arctan(M_hat[2, 1] / M_hat[2, 2]))
    # p2 = R2D * np.arctan(M_hat[2, 0] / np.sqrt(M_hat[0, 0] ** 2 + M_hat[1, 0] ** 2))
    p2.append(R2D * np.arcsin(M_hat[2, 0]))
    y2.append(R2D * np.arctan2(M_hat[0, 0], M_hat[1, 0]))

print(f"MEAN {{Roll = {np.mean(r2)}, Pitch = {np.mean(p2)} , yaw = {np.mean(y2)}}}")

# --------------------------------------------------------------------------------------------------#
SSE = np.trace((M_hat @ D_enu - D_loc) @ R_inv @ (M_hat @ D_enu - D_loc).T)
print(f"SSE = {SSE}")

print()
