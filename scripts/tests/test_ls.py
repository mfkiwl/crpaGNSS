import numpy as np
from charlizard.estimators.ls import LeastSquares
from charlizard.utils.combinations import combinations

import matplotlib.pyplot as plt
import seaborn as sns

from sympy import plot_implicit, Point, Eq, Line, sqrt, Pow, symbols
from sympy.plotting.plot import MatplotlibBackend, Plot

# from sympy.abc import x, y


# def aoa_model(y, **params):
#     cosy = np.cos(y)
#     siny = np.sin(y)
#     H = np.column_stack((-cosy, siny))
#     b = -params["rcvr_pos"][:, 0] * cosy + params["rcvr_pos"][:, 1] * siny
#     return b, H


# def aoa_model(y, **params):
#   I = np.eye(2)
#   H = np.zeros((2,2), dtype=np.float64)
#   b = np.zeros(2, dtype=np.float64)
#   for i in range(y.size):
#     n = np.array([np.sin(y[i]), np.cos(y[i])], dtype=np.float64)
#     Inn = (I - np.outer(n, n))
#     H = H + Inn
#     b = b + Inn @ params['rcvr_pos'][i,:]
#   return b, H


def aoa_model(x, **params):
    dE = x[0] - params["rcvr_pos"][:, 0]
    dN = x[1] - params["rcvr_pos"][:, 1]
    y = np.arctan2(dE, dN)
    r2 = dE**2 + dN**2
    H = np.column_stack((dN, -dE)) / r2[:, None]
    return y, H


def rdoa_model(x, **params):
    idx = np.arange(params["rcvr_pos"].shape[0])
    c0, c1 = combinations(idx)
    dR = x - params["rcvr_pos"]
    R = np.sqrt(dR[:, 0] ** 2 + dR[:, 1] ** 2)
    y = R[c0] - R[c1]
    H = dR[c0, :] / R[c0, None] - dR[c1, :] / R[c1, None]
    return y, H


def aoa_rdoa_model(x, **params):
    y1, H1 = aoa_model(x, **params)
    y2, H2 = rdoa_model(x, **params)
    y = np.concatenate((y1, y2))
    H = np.concatenate((H1, H2))
    return y, H


if __name__ == "__main__":
    emitter = np.array([-10.1, 99.9], dtype=np.float64)
    params = {
        "rcvr_pos": np.array([[850, -250], [400, 950], [-900, -600], [-100, 100]], dtype=np.float64),
        "threshold": 1e-6,
    }
    rdoa_std = 50e-9 * 299792458  # 50 ns
    aoa_std = np.deg2rad(3.0)

    # * ##### SOLVE USING TDOA #####
    m0, H = rdoa_model(emitter, **params)
    m0 += np.random.randn(6) * rdoa_std

    solver = LeastSquares(rdoa_model)
    solver.set_initial_conditions(np.zeros(2, dtype=np.float64))
    solver.solve("iterative", params, m0)
    print("RDOA")
    print(solver.estimate)
    print(emitter - solver.estimate)
    print()

    # * ##### SOLVE USING AOA #####
    # y = np.arctan2(emitter[0]-params['rcvr_pos'][:,0], emitter[1]-params['rcvr_pos'][:,1])
    m1, _ = aoa_model(emitter, **params)
    m1 += np.random.randn(4) * aoa_std

    solver = LeastSquares(aoa_model)
    solver.set_initial_conditions(np.zeros(2, dtype=np.float64))
    solver.solve("iterative", params, m1)
    print("AOA")
    print(solver.estimate)
    print(emitter - solver.estimate)
    print()

    # * ##### SOLVE USING TDOA & AOA #####
    # m, _ = aoa_rdoa_model(emitter, **params)
    m = np.concatenate((m1, m0))
    solver = LeastSquares(aoa_rdoa_model)
    solver.set_initial_conditions(np.zeros(2, dtype=np.float64))
    solver.solve("iterative", params, m)
    print("AOA+RDOA")
    print(solver.estimate)
    print(emitter - solver.estimate)
    print()

    # * ##### PLOT #####
    aoa_lines = np.dstack(
        (
            params["rcvr_pos"],
            params["rcvr_pos"] + 1200 * np.array([np.sin(m[:4]), np.cos(m[:4])]).T,
        )
    )
    c0, c1 = combinations(np.arange(4))
    # print(f"c0 = {c0}, c1 = {c1}")

    x, y = symbols("x y")
    p = None
    for i in range(4):
        line = Line(Point(aoa_lines[i, 0, 0], aoa_lines[i, 1, 0]), Point(aoa_lines[i, 0, 1], aoa_lines[i, 1, 1]))
        p2 = plot_implicit(line.equation(x, y), (x, -1000, 1000), (y, -1000, 1000), show=False, line_color="b")
        if p:
            p.extend(p2)
        else:
            p = p2
    for i in range(6):
        x0 = params["rcvr_pos"][c0[i], 0]
        y0 = params["rcvr_pos"][c0[i], 1]
        x1 = params["rcvr_pos"][c1[i], 0]
        y1 = params["rcvr_pos"][c1[i], 1]
        r0 = sqrt(Pow(x0 - x, 2) + Pow(y0 - y, 2))
        r1 = sqrt(Pow(x1 - x, 2) + Pow(y1 - y, 2))
        p2 = plot_implicit(Eq(r0 - r1, m[4 + i]), (x, -1000, 1000), (y, -1000, 1000), show=False, line_color="c")
        if p:
            p.extend(p2)
        else:
            p = p2

    backend = MatplotlibBackend(p)
    backend.process_series()
    backend.fig.tight_layout()
    f = backend.fig
    ax = backend.ax[0]

    sns.scatterplot(x=params["rcvr_pos"][:, 0], y=params["rcvr_pos"][:, 1], ax=ax, color="k", s=50)
    sns.scatterplot(x=[emitter[0]], y=[emitter[1]], ax=ax, color="g", marker="*", s=300)
    sns.scatterplot(x=[solver.estimate[0]], y=[solver.estimate[1]], ax=ax, color="r", marker="*", s=300)
    plt.grid(visible=True, which="both", axis="both")
    p.show()
