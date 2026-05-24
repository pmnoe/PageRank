import numpy as np


def page_rank_update(p_curr, M, alpha, v):
    p_next = alpha * M @ p_curr + (1 - alpha) * v
    return p_next


def get_PageRank_vector(M, alpha, v, max_itter):
    n = len(v)
    p_pagerank = np.ones(n) / n
    for it in range(max_itter):
        p_pagerank = page_rank_update(p_pagerank, M, alpha, v)

    return p_pagerank
