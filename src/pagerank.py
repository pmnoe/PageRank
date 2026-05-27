import numpy as np
import scipy.sparse as sp


def build_transition_matrix(edges, n):
    src, dst = edges["src"].values, edges["dst"].values
    M_tilde = sp.csc_matrix((np.ones(len(src)), (dst, src)), shape=(n, n))
    d = np.asarray(M_tilde.sum(axis=0), dtype=float).ravel()
    return M_tilde / d


def solve_direct(M, alpha, v):
    n = len(v)
    A = sp.eye(n, format="csc") - alpha * M
    return sp.linalg.splu(A).solve((1 - alpha) * v)


def solve_iterative(M, alpha, v, tol=1e-10, max_iter=500):
    p = np.ones(len(v)) / len(v)
    history = [p.copy()]
    for _ in range(max_iter):
        p_new = alpha * (M @ p) + (1 - alpha) * v
        history.append(p_new.copy())
        if np.linalg.norm(p_new - p, 1) < tol * (1 - alpha):
            break
        p = p_new
    return p_new, history
