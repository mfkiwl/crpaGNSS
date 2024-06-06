"""
|============================================= phd.py =============================================|
|                                                                                                  |
|   Property of Daniel Sturdivant. Unauthorized copying of this file via any medium would be       |
|   super sad and unfortunate for me. Proprietary and confidential.                                |
|                                                                                                  |
| ------------------------------------------------------------------------------------------------ |
|                                                                                                  |
|   @file     charlizard/algorithms/phd.py                                                         |
|   @brief    Probability Hypothesis Density Filter.                                               |
|   @author   Daniel Sturdivant <sturdivant20@gmail.com>                                           |
|   @date     May 2024                                                                             |
|                                                                                                  |
|==================================================================================================|
"""

# TODO: rewrite to incorperate general models instead of only aoa/tdoa

import numpy as np
from scipy.linalg import norm, inv, cholesky, block_diag, det
from scipy.stats.distributions import chi2
from dataclasses import dataclass
import navtools as nt

from charlizard.utils.combinations import combinations

TWO_PI = 2.0 * np.pi
R2D = 180 / np.pi
LIGHT_SPEED = 299792458.0


@dataclass
class PHDFilterConfig:
    T: float  #! integration period [s]
    order: int  #! 2: constant velocity, 3: constant acceleration
    vel_std: float  #! velocity process noise standard deviation [m/s]
    acc_std: float  #! acceleration process noise standard deviation [m/s^2]
    aoa_std: float  #! angle of arrival measurement noise standard deviation [deg]
    tdoa_std: float  #! time difference of arrival measurement noise standard deviation [ns]
    aoa_clutter: int  #! Poisson average rate of uniform clutter for AOA
    tdoa_clutter: int  #! Poisson average rate of uniform clutter for TDOA
    aoa_clutter_range: np.ndarray  #! uniform range for AOA clutter values [rad]
    tdoa_clutter_range: np.ndarray  #! uniform range for TDOA clutter values [m]
    aoa_update_rate: int  #! time between aoa measurement updates [s]
    tdoa_update_rate: int  #! time between aoa measurement updates [s]
    p_d: float  #! probability of detection
    p_s: float  #! probability of survival
    prune_threshold: float
    merge_threshold: float
    cap_threshold: int


@dataclass
class PHDFilterTruth:
    rcvr_pos: list  #! list of (N_rcvr x 2) east and north receiver positions
    rcvr_vel: list  #! list of (N_rcvr x 2) east and north receiver velocities
    emit_pos: list  #! list of (N_emit x 2) east and north emitter positions
    emit_vel: list  #! list of (N_emit x 2) east and north emitter velocities
    t_birth: list  #! list of emitter birth times (indexes)
    t_end: list  #! list of emitter death times (indexes)
    w_birth: list  #! list of birth weights/confidence
    N_time: int  #! total number of time points
    N_emit: list  #! list of the number of emitters at each time point
    N_rcvr: int  #! number of receivers


@dataclass
class PHDFilterEst:
    x: list  #! list of states at each epoch
    P: list  #! list of state covariances at each epoch
    w: list  #! list of weights at each epoch
    n: list  #! list of number of expected targets
    x_err: list


@dataclass
class PHDFilterBirth:
    w: float  #! weight of new birth
    P: np.ndarray  #! covariance of new birth
    x: np.ndarray  #! mean of new birth
    t_start: float  #! time when target is added
    t_end: float  #! time when target dies


class PHDFilter:
    def __init__(self, conf: PHDFilterConfig):
        # initialize filter
        self.T = conf.T
        self.aoa_std = conf.aoa_std / R2D
        self.tdoa_std = conf.tdoa_std * 1e-9 * LIGHT_SPEED
        self.p_d = conf.p_d
        self.p_s = conf.p_s
        self.aoa_update_rate = conf.aoa_update_rate
        self.tdoa_update_rate = conf.tdoa_update_rate
        self.prune_threshold = conf.prune_threshold
        self.merge_threshold = conf.merge_threshold
        self.cap_threshold = conf.cap_threshold

        # initialize clutter
        self.__init_clutter_param(conf)

        # initialize process model
        self.__init_process_model(conf)

    # * ##### gen_truth #####
    def gen_truth(self, truth: PHDFilterTruth):
        self.Truth = truth
        self.Truth.t_birth = np.array(self.Truth.t_birth)
        self.Truth.t_end = np.array(self.Truth.t_end)

    # * ##### gen_meas #####
    def gen_meas(self):
        self.Y = [None] * self.Truth.N_time
        self.Y_true = [None] * self.Truth.N_time

        for i in range(self.Truth.N_time):
            self.Y[i] = []
            self.Y_true[i] = []

            # truth target detection
            if self.Truth.N_emit[i] > 0:
                # determine if each target was truely detected
                # detect = np.random.rand(self.Truth.N_emit[i]) < self.p_d
                detect = np.ones(self.Truth.N_emit[i], dtype=bool)
                for j in range(detect.size):
                    if detect[j]:
                        # create true measurements
                        # if (i % self.tdoa_update_rate) == 0:  # tdoa is less frequent
                        #     if self.Truth.emit_pos[i].size > 2:
                        #         y, H, R, R_sqrt = self.__aoa_rdoa_model(self.Truth.emit_pos[i][j, :], self.Truth.rcvr_pos[i])
                        #     else:
                        #         y, H, R, R_sqrt = self.__aoa_rdoa_model(self.Truth.emit_pos[i], self.Truth.rcvr_pos[i])
                        # else:
                        if self.Truth.emit_pos[i].size > 2:
                            y, H, R, R_sqrt = self.__aoa_model(self.Truth.emit_pos[i][j, :], self.Truth.rcvr_pos[i])
                        else:
                            y, H, R, R_sqrt = self.__aoa_model(self.Truth.emit_pos[i], self.Truth.rcvr_pos[i])

                        # add true measurements
                        if j == 0:
                            self.Y[i] = np.array([y + R_sqrt @ np.random.randn(y.size)]).reshape(y.size, 1)
                            self.Y_true[i] = np.array([y]).reshape(y.size, 1)
                        else:
                            self.Y[i] = np.concatenate(
                                (self.Y[i], np.array([y + R_sqrt @ np.random.randn(y.size)]).reshape(y.size, 1)), axis=1
                            )
                            self.Y_true[i] = np.concatenate((self.Y_true[i], np.array([y]).reshape(y.size, 1)), axis=1)

            # determine number of clutter measurements
            # if (i % self.tdoa_update_rate) == 0:  # tdoa is less frequent
            #     N_clutter = np.random.poisson(self.lambda_comb)
            #     c0, _ = combinations(np.arange(self.Truth.N_rcvr))
            #     aoa_clutter = self.aoa_clutter_range * np.random.rand(self.Truth.N_rcvr, N_clutter) - self.aoa_clutter_range / 2
            #     tdoa_clutter = self.tdoa_clutter_range * np.random.rand(c0.size, N_clutter) - self.tdoa_clutter_range / 2
            #     y_clutter = np.vstack((aoa_clutter, tdoa_clutter))
            #     self.Y_type[i] = "comb"
            # else:
            N_clutter = np.random.poisson(self.lambda_aoa)
            y_clutter = nt.wrapTo2Pi(self.aoa_clutter_range * np.random.randn(self.Truth.N_rcvr, N_clutter))

            # add clutter measurements
            if len(self.Y[i]) > 0:
                self.Y[i] = np.concatenate((self.Y[i], y_clutter), axis=1)
            else:
                self.Y[i] = y_clutter

            # self.Y[i] = nt.wrapToPi(self.Y[i])
            # self.Y_true[i] = nt.wrapToPi(self.Y_true[i])

    # * ##### run_filter ###############################################################################################
    def run_filter(self, init_states, init_cov, init_weights):
        # output
        self.Est = PHDFilterEst(
            x=[None] * self.Truth.N_time,
            P=[None] * self.Truth.N_time,
            w=[None] * self.Truth.N_time,
            n=[0] * self.Truth.N_time,
            x_err=[None] * self.Truth.N_time,
        )

        # init
        x_update = init_states
        P_update = init_cov
        w_update = init_weights
        J_k = x_update.shape[1]
        N_states = x_update.shape[0]

        # --- run filter ---
        for i in range(self.Truth.N_time):

            # TODO: FIX BIRTHS TO NOT BE EXACT FROM TRUTH
            # --- births ---
            w_birth = np.empty(0)
            x_birth = np.empty(0)
            P_birth = np.empty(0)
            if (i % self.tdoa_update_rate) == 0:  # assuming other algorithm runs when tdoa is updated
                w_birth = 0.2 * np.ones(self.Truth.N_emit[i])
                P_birth = np.tile(np.array([self.default_P]).transpose(1, 2, 0), (1, 1, self.Truth.N_emit[i]))
                if self.order == 2:
                    x_birth = np.hstack((self.Truth.emit_pos[i], self.Truth.emit_vel[i])).T
                elif self.order == 3:
                    x_birth = np.hstack(
                        (self.Truth.emit_pos[i], self.Truth.emit_vel[i], np.zeros(self.Truth.emit_vel[i].shape))
                    ).T
                if self.Truth.N_emit[i] == 1:
                    x_birth = np.array([x_birth]).T

            # --- spawns ---
            # basing off covariance for now
            # N_spawns = 2
            # for j in range(w_update.size):
            #     if j == 0:
            #         w_spawn = 0.1 * np.ones(N_spawns)
            #         x_spawn = x_update[:, j][:, None] + P_update[:, :, j] @ np.random.randn(N_states, N_spawns)
            #         P_spawn = np.tile(np.array([P_update[:, :, j]]).transpose(1, 2, 0), (1, 1, N_spawns))
            #     else:
            #         w_spawn = np.concatenate((w_spawn, 0.1 * np.ones(N_spawns)))
            #         x_spawn = np.concatenate(
            #             (x_spawn, x_update[:, j][:, None] + P_update[:, :, j] @ np.random.randn(N_states, N_spawns)), axis=1
            #         )
            #         P_spawn = np.concatenate(
            #             (P_spawn, np.tile(np.array([P_update[:, :, j]]).transpose(1, 2, 0), (1, 1, N_spawns))), axis=2
            #         )

            # --- prediction for surviving targets ---
            x_predict = np.zeros((N_states, J_k))
            P_predict = np.zeros((N_states, N_states, J_k))
            w_predict = self.p_s * w_update
            for j in range(J_k):
                x_predict[:, j] = self.A @ x_update[:, j]
                P_predict[:, :, j] = self.A @ P_update[:, :, j] @ self.A.T + self.Q

            # --- union of births, spawns, and predictions ---
            # w_predict = np.concatenate((w_spawn, w_predict), axis=0)
            # x_predict = np.concatenate((x_spawn, x_predict), axis=1)
            # P_predict = np.concatenate((P_spawn, P_predict), axis=2)
            if w_birth.size > 0:
                w_predict = np.concatenate((w_birth, w_predict), axis=0)
                x_predict = np.concatenate((x_birth, x_predict), axis=1)
                P_predict = np.concatenate((P_birth, P_predict), axis=2)
            J_k = w_predict.size

            # --- chi^2 test (removes most faulty measurements) ---
            valid_idx = self.__chi2_test(x_predict, P_predict, J_k, i)

            # passing measurements
            Y = self.Y[i][:, valid_idx]
            n_meas, y_len = Y.shape

            # --- update of targets ---
            pdf_n = np.zeros((J_k, y_len))
            m_tmp = np.zeros((N_states, J_k, y_len))
            P_tmp = np.zeros((N_states, N_states, J_k))
            x_update = x_predict
            P_update = P_predict
            w_update = (1.0 - self.p_d) * w_predict
            for j in range(J_k):
                # get measurements and observation matrix
                # if (i % self.tdoa_update_rate) == 0:  # tdoa is less frequent
                #     y, H, R, _ = self.__aoa_rdoa_model(x_predict[:2, j], self.Truth.rcvr_pos[i])
                # else:
                y, H, R, _ = self.__aoa_model(x_predict[:2, j], self.Truth.rcvr_pos[i])

                # innovation covariance
                dy = Y - np.tile(y[:, None], (1, y_len))
                S = H @ P_predict[:, :, j] @ H.T + R
                inv_S = inv(S)

                # multivariate pdf
                dy_sq = np.squeeze(dy)
                det_S = det(S)
                if len(dy_sq.shape) > 1:
                    pdf_n[j, :] = (np.exp(-0.5 * dy_sq.T @ inv_S @ dy_sq) / np.sqrt(TWO_PI ** (n_meas / 2) * det_S)).diagonal()
                else:
                    pdf_n[j, :] = np.exp(-0.5 * dy_sq.T @ inv_S @ dy_sq) / np.sqrt(TWO_PI ** (n_meas / 2) * det_S)

                # kalman update
                K = P_predict[:, :, j] @ H.T @ inv_S
                P_tmp[:, :, j] = (self.I - K @ H) @ P_predict[:, :, j]
                m_tmp[:, j, :] = np.tile(x_predict[:, j][:, None], (1, y_len)) + K @ dy

            # --- combine measurement updates ---
            for k in range(y_len):
                w_tmp = self.p_d * w_predict * pdf_n[:, k]
                # if (i % self.tdoa_update_rate) == 0:  # tdoa is less frequent
                #     w_tmp /= self.k_comb + np.sum(w_tmp)
                # else:
                w_tmp /= self.k_aoa + np.sum(w_tmp)
                w_update = np.concatenate((w_update, w_tmp), axis=0)
                x_update = np.concatenate((x_update, m_tmp[:, :, k]), axis=1)
                P_update = np.concatenate((P_update, P_tmp), axis=2)
            # test = ~np.isnan(w_update)
            # w_update = w_update[test]
            # x_update = x_update[:, test]
            # P_update = P_update[:, :, test]

            # --- pruning test ---
            w_update, x_update, P_update = self.__pruning_test(w_update, x_update, P_update)

            # --- merging test ---
            w_update, x_update, P_update = self.__merging_test(w_update, x_update, P_update)

            # --- capping test ---
            w_update, x_update, P_update = self.__capping_test(w_update, x_update, P_update)

            # --- multiple target extraction ---
            self.Est.x[i] = []
            self.Est.P[i] = []
            self.Est.w[i] = []
            self.Est.n[i] = 0
            for j in range(w_update.size):
                if w_update[j] > 0.5:
                    for k in range(int(np.round(w_update[j]))):
                        self.Est.x[i].append(x_update[:, j])
                        self.Est.P[i].append(P_update[:, :, j])
                        self.Est.w[i].append(w_update[j])
                        self.Est.n[i] += 1
            J_k = self.Est.n[i]
            if J_k > 0:
                self.Est.x[i] = np.array(self.Est.x[i]).T
                self.Est.P[i] = np.array(self.Est.P[i]).transpose(1, 2, 0)
                self.Est.w[i] = np.array(self.Est.w[i])
                x_update = self.Est.x[i]
                P_update = self.Est.P[i]
                w_update = self.Est.w[i]
            else:
                J_k = w_update.size
                self.Est.x[i] = np.empty(0)
                self.Est.P[i] = np.empty(0)
                self.Est.w[i] = np.empty(0)

            # --- figure out error ---
            if self.Truth.emit_pos[i].size > 2:
                for j in range(min([self.Truth.emit_pos[i].shape[0], self.Est.x[i][:2, :].T.shape[0]])):
                    error = self.Truth.emit_pos[i][j, :] - self.Est.x[i][:2, :].T
                    idx1 = np.abs(error).argmin(axis=0)
                    idx2 = np.arange(2)
                    if j == 0:
                        self.Est.x_err[i] = np.array([error[idx1, idx2]])
                    else:
                        self.Est.x_err[i] = np.vstack((self.Est.x_err[i], error[idx1, idx2]))
            else:
                error = self.Truth.emit_pos[i] - self.Est.x[i][:2, :].T
                # idx1 = np.abs(error).argmin(axis=0)
                # idx2 = np.arange(2)
                # self.Est.x_err[i] = np.array([error[idx1, idx2]])
                self.Est.x_err[i] = error
                # self.Est.x_err[i] = np.array([error])

            # print()

    # * ################################################################################################################

    # * ##### init_clutter_param #####
    def __init_clutter_param(self, conf: PHDFilterConfig):
        # aoa clutter statistics
        self.lambda_aoa = conf.aoa_clutter  #! Poisson average rate of clutter
        self.aoa_clutter_range = conf.aoa_clutter_range[1] - conf.aoa_clutter_range[0]  #! uniform clutter
        self.k_aoa = self.lambda_aoa / self.aoa_clutter_range  #! uniform clutter density

        # tdoa clutter statistics
        self.lambda_tdoa = conf.tdoa_clutter
        self.tdoa_clutter_range = conf.tdoa_clutter_range[1] - conf.tdoa_clutter_range[0]
        self.k_tdoa = self.lambda_tdoa / self.tdoa_clutter_range

        # combined
        self.lambda_comb = conf.aoa_clutter + conf.tdoa_clutter
        self.k_comb = self.k_aoa * self.k_tdoa  #! combined clutter density

    # * ##### init_process_model #####
    def __init_process_model(self, conf: PHDFilterConfig):
        self.order = conf.order
        if conf.order == 2:
            t = self.T
            q = conf.vel_std
            self.A = np.array([[1.0, 0.0, t, 0.0], [0.0, 1.0, 0.0, t], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
            self.Q = np.array(
                [
                    [(q * t**3) / 3, 0.0, (q * t**2) / 2, 0.0],
                    [0.0, (q * t**3) / 3, 0.0, (q * t**2) / 2],
                    [(q * t**2) / 2, 0.0, q * t, 0.0],
                    [0.0, (q * t**2) / 2, 0.0, q * t],
                ]
            )
            self.I = np.eye(4)
            # self.default_P = np.diag([50.0, 50.0, 5.0, 5.0]) ** 2
            self.default_P = np.diag([25.0, 25.0, 1.0, 1.0]) ** 2
        elif conf.order == 3:
            t = self.T
            qv = conf.vel_std
            qa = conf.acc_std
            self.A = np.array(
                [
                    [1.0, 0.0, t, 0.0, 0.5 * t**2, 0.0],
                    [0.0, 1.0, 0.0, t, 0.0, 0.5 * t**2],
                    [0.0, 0.0, 1.0, 0.0, t, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, t],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ]
            )
            self.Q = np.array(
                [
                    [(qa * t**5) / 20 + (qv * t**3) / 3, 0.0, (qa * t**4) / 8 + (qv * t**2) / 2, 0.0, (qa * t**3) / 6, 0.0],
                    [0.0, (qa * t**5) / 20 + (qv * t**3) / 3, 0.0, (qa * t**4) / 8 + (qv * t**2) / 2, 0.0, (qa * t**3) / 6],
                    [(qa * t**4) / 8 + qv * (t**2) / 2, 0.0, (qa * t**3) / 3 + qv * t, 0.0, (qa * t**2) / 2, 0.0],
                    [0.0, (qa * t**4) / 8 + qv * (t**2) / 2, 0.0, (qa * t**3) / 3 + qv * t, 0.0, (qa * t**2) / 2],
                    [(qa * t**3) / 6, 0.0, (qa * t**2) / 2, 0.0, qa * t, 0.0],
                    [0.0, (qa * t**3) / 6, 0.0, (qa * t**2) / 2, 0.0, qa * t],
                ]
            )
            self.I = np.eye(6)
            # self.default_P = np.diag([50.0, 50.0, 5.0, 5.0, 1.0, 1.0]) ** 2
            self.default_P = np.diag([25.0, 25.0, 1.0, 1.0, 0.1, 0.1]) ** 2

    def __aoa_model(self, x, rcvr_pos):
        dE = x[0] - rcvr_pos[:, 0]
        dN = x[1] - rcvr_pos[:, 1]
        y = nt.wrapTo2Pi(np.arctan2(dE, dN))
        r2 = dE**2 + dN**2

        if self.order == 2:
            Z = np.zeros((y.size, 2))
        elif self.order == 3:
            Z = np.zeros((y.size, 4))
        H = np.column_stack((dN, -dE, Z)) / r2[:, None]
        R = np.diag(self.aoa_std**2 * np.ones(y.size))
        R_sqrt = np.diag(self.aoa_std * np.ones(y.size))
        return y, H, R, R_sqrt

    def __rdoa_model(self, x, rcvr_pos):
        idx = np.arange(rcvr_pos.shape[0])
        c0, c1 = combinations(idx)
        dR = x - rcvr_pos
        R = np.sqrt(dR[:, 0] ** 2 + dR[:, 1] ** 2)
        y = R[c0] - R[c1]

        if self.order == 2:
            Z = np.zeros((y.size, 2))
        elif self.order == 3:
            Z = np.zeros((y.size, 4))
        H = dR[c0, :] / R[c0, None] - dR[c1, :] / R[c1, None]
        H = np.column_stack((H, Z))
        R = np.diag(self.tdoa_std**2 * np.ones(y.size))
        R_sqrt = np.diag(self.tdoa_std * np.ones(y.size))
        return y, H, R, R_sqrt

    def __aoa_rdoa_model(self, x, rcvr_pos):
        y1, H1, R1, R1_sqrt = self.__aoa_model(x, rcvr_pos)
        y2, H2, R2, R2_sqrt = self.__rdoa_model(x, rcvr_pos)
        y = np.concatenate((y1, y2))
        H = np.concatenate((H1, H2))
        R = block_diag(R1, R2)
        R_sqrt = block_diag(R1_sqrt, R2_sqrt)
        return y, H, R, R_sqrt

    def __chi2_test(self, m_predict, P_predict, J_k, i):
        valid_idx = np.empty(0)
        z_len = self.Y[i].shape[1]
        gamma = chi2.ppf(0.997, df=z_len)  # 3 sigma bound
        for j in range(J_k):
            # if (i % self.tdoa_update_rate) == 0:  # tdoa is less frequent
            #     y, H, R, _ = self.__aoa_rdoa_model(m_predict[:2, j], self.Truth.rcvr_pos[i])
            # else:
            y, H, R, _ = self.__aoa_model(m_predict[:2, j], self.Truth.rcvr_pos[i])
            S = H @ P_predict[:, :, j] @ H.T + R
            inv_sqrt_S = inv(cholesky(S))
            dy = self.Y[i] - np.tile(y[:, None], (1, z_len))
            dist = np.sum((inv_sqrt_S.T @ dy) ** 2, axis=0)
            if j == 0:
                valid_idx = np.nonzero(dist < gamma)[0]
            else:
                valid_idx = np.union1d(valid_idx, np.nonzero(dist < gamma)[0])
        valid_idx = np.union1d(valid_idx, np.array([0], dtype=int))
        return valid_idx.astype(int)

    def __pruning_test(self, w_update, m_update, P_update):
        test = w_update > self.prune_threshold
        w_update = w_update[test]
        m_update = m_update[:, test]
        P_update = P_update[:, :, test]
        return w_update, m_update, P_update

    def __merging_test(self, w_update, m_update, P_update):
        N_state = m_update.shape[0]
        w_tmp = np.empty(0)
        m_tmp = np.empty(0)
        P_tmp = np.empty(0)
        idx = np.arange(w_update.size)
        el = 0
        while idx.size > 0:
            j = np.argmax(w_update)
            ij = np.empty(0, dtype=int)
            inv_P = inv(P_update[:, :, j])
            for i in idx:
                dm = m_update[:, i] - m_update[:, j]
                val = dm[None, :] @ inv_P @ dm[:, None]
                if val.squeeze() <= self.merge_threshold:
                    ij = np.append(ij, i)

            if el == 0:
                w_tmp = np.array([np.sum(w_update[ij])])
                m_tmp = np.array([np.sum(w_update[ij] * m_update[:, ij], axis=1) / w_tmp[el]]).T
                P_sum = np.zeros((N_state, N_state))
                for a in ij:
                    dm = m_tmp[:, el] - m_update[:, a]
                    P_sum += w_update[a] * P_update[:, :, a] + dm[:, None] * dm[None, :]
                P_tmp = np.array([P_sum / w_tmp[el]]).transpose(1, 2, 0)
            else:
                w_tmp = np.append(w_tmp, np.sum(w_update[ij]))
                m_tmp = np.concatenate((m_tmp, np.array([np.sum(w_update[ij] * m_update[:, ij], axis=1) / w_tmp[el]]).T), axis=1)
                P_sum = np.zeros((N_state, N_state))
                for a in ij:
                    dm = m_tmp[:, el] - m_update[:, a]
                    P_sum += w_update[a] * P_update[:, :, a] + dm[:, None] * dm[None, :]
                P_tmp = np.concatenate((P_tmp, np.array([P_sum / w_tmp[el]]).transpose(1, 2, 0)), axis=2)

            idx = np.setdiff1d(idx, ij)
            w_update[ij] = -1
            el += 1
        return w_tmp, m_tmp, P_tmp

    def __capping_test(self, w_update, m_update, P_update):
        if w_update.size > self.cap_threshold:
            idx = np.argsort(w_update)
            idx = idx[: self.cap_threshold]
            w_tmp = w_update[idx]
            w_update = w_tmp * sum(w_update) / sum(w_tmp)
            m_update = m_update[:, idx]
            P_update = P_update[:, :, idx]
        return w_update, m_update, P_update
