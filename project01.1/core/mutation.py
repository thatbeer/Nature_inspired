import copy
import numpy as np

from .dataclass import Chromosome


def mutate(candidate:Chromosome, p_mutate: float) -> Chromosome:
    # res = candidate.copy()
    # res = candidate
    if np.random.rand() > p_mutate:
        return candidate
    res = copy.copy(candidate)
    n = len(res.gene)
    idx1, idx2 = np.random.choice(n, size=2, replace=False)
    res.gene[idx1], res.gene[idx2] = res.gene[idx2] , res.gene[idx1]
    return res

def multi_mutate(candidate:Chromosome, num_mutate:int=2,p_mutate: float=1.0) -> Chromosome:
    # res = candidate.copy()
    # res = candidate
    res = copy.copy(candidate)
    for _ in range(num_mutate):
        res = mutate(res)
    return res

def inversion(candidate:Chromosome) -> Chromosome:
    # res = candidate.copy()
    # res = candidate
    res = copy.copy(candidate)
    n = len(res.gene)
    idx1, idx2 = np.sort(np.random.choice(n, 2, replace=False))
    # res.gene[idx1:idx2+1] = res.gene[idx2+1:idx1-1:-1]
    res.gene[idx1:idx2+1] = list(reversed(res.gene[idx1:idx2+1]))
    return res

def scramble(candidate:Chromosome) -> Chromosome:
    # res = candidate.copy()
    # res = candidate
    res = copy.copy(candidate)
    n = len(res.gene)
    idx1, idx2 = np.sort(np.random.choice(n, 2, replace=False))
    res.gene[idx1:idx2+1] = list(np.random.shuffle(res[idx1:idx2+1]))
    return res