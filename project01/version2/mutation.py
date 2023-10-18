import numpy as np
from .classes import Chromosome
    
def mutate(candidate:Chromosome) -> Chromosome:
    res = candidate.copy()
    n = len(res.gene)
    idx1, idx2 = np.random.choice(n, size=2, replace=False)
    res.gene[idx1], res.gene[idx2] = res.gene[idx2] , res.gene[idx1]
    return res

def multi_mutate(candidate:Chromosome, num_mutate:int=2) -> Chromosome:
    res = candidate.copy()
    for _ in range(num_mutate):
        res = mutate(res)
    return res

def inversion(candidate:Chromosome) -> Chromosome:
    res = candidate.copy()
    n = len(res.gene)
    idx1, idx2 = np.sort(np.random.choice(n, 2, replace=False))
    # res.gene[idx1:idx2+1] = res.gene[idx2+1:idx1-1:-1]
    res.gene[idx1:idx2+1] = list(reversed(res.gene[idx1:idx2+1]))
    return res

def scramble(candidate:Chromosome) -> Chromosome:
    res = candidate.copy()
    n = len(res.gene)
    idx1, idx2 = np.sort(np.random.choice(n, 2, replace=False))
    res.gene[idx1:idx2+1] = list(np.random.shuffle(res[idx1:idx2+1]))
    return res
