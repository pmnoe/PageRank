import numpy as np
import scipy.sparse as sp


def page_rank_direct(M, alpha, v, pages):
    n = len(v)
    A = sp.eye(n, format="csc") - alpha * M
    b = (1 - alpha) * v
    LU_A = sp.linalg.splu(A)
    p = LU_A.solve(b)
    p_pagerank = pages.assign(pagerank=p).sort_values(by="pagerank", ascending=False)
    return p_pagerank


def page_rank_update(p_curr, M, alpha, v):
    p_next = alpha * M @ p_curr + (1 - alpha) * v
    return p_next


def get_pagerank_vector(M, alpha, v, pages, max_itter):
    n = len(v)
    p = np.ones(n) / n
    for it in range(max_itter):
        p = page_rank_update(p, M, alpha, v)
    p_pagerank = pages.assign(pagerank=p).sort_values(by="pagerank", ascending=False)
    return p_pagerank
