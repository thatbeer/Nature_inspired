import copy
import numpy as np
import random
from typing import List
from .dataclass import Chromosome
from .selection import create_chromosome


def mutate(candidate: List[int], p_mutate: float = 1.0) -> List[int]:
    # res = candidate.copy()
    # res = candidate
    if np.random.rand() > p_mutate:
        return candidate
    res = candidate.copy()
    n = len(res)
    idx1, idx2 = np.random.choice(n, size=2, replace=False)
    res[idx1], res[idx2] = res[idx2] , res[idx1]
    return res

def inversion(candidate: List[int], p_mutate: float=1.0) -> List[int]:
    # res = candidate.copy()
    # res = candidate
    if np.random.rand() > p_mutate:
        return candidate
    res = candidate.copy()
    n = len(res)
    idx1, idx2 = np.sort(np.random.choice(n, 2, replace=False))
    # res[idx1:idx2+1] = res[idx2+1:idx1-1:-1]
    res[idx1:idx2+1] = list(reversed(res[idx1:idx2+1]))
    return res

# # somthing went wrong, have not fixed yet
# def scramble(candidate:List[int], p_mutate: float=1.0) -> List[int]:
#     # res = candidate.copy()
#     if np.random.rand() > p_mutate:
#         return candidate
#     # res = candidate.copy()
#     res = candidate
#     n = len(res)
#     idx1, idx2 = np.sort(np.random.choice(n, 2, replace=False))
#     random.shuffle(res[idx1:idx2+1])
#     return res

def scramble(candidate: List[int], p_mutate: float = 1.0) -> List[int]:
    if np.random.rand() > p_mutate:
        return candidate

    res = candidate.copy()  # Make a copy of the candidate
    n = len(res)
    idx1, idx2 = np.sort(np.random.choice(n, 2, replace=False))

    batch = np.random.choice(res[idx1:idx2+1], idx2+1-idx1, replace=False)
    res[idx1:idx2+1] = batch

    return res


def multi_mutate(candidate:List[int], p_mutate: float=1.0, num_mutate: int = 2) -> List[int]:
    # res = candidate.copy()
    # num_mutate = 2
    # res = candidate
    if np.random.rand() > p_mutate:
        return candidate
    res = candidate.copy()
    for _ in range(num_mutate):
        res = mutate(res)
    return res