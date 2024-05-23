import numpy as np
from numba import njit


@njit(cache=True)
def combinations(v):
    n = v.size
    c0 = np.zeros(2 ** (n - 1) - (n - 2), dtype=np.int32)
    c1 = np.zeros(2 ** (n - 1) - (n - 2), dtype=np.int32)
    a = 0
    for i in range(n):
        for j in range(i + 1, n):
            c0[a] = v[i]
            c1[a] = v[j]
            a += 1
    return c0, c1
