import numpy as np
from .classes import Chromosome
from .base import Mutate
from typing import List

def mutate(candidate:Chromosome) -> Chromosome:
    # res = candidate.copy()
    res = candidate
    n = len(res.gene)
    idx1, idx2 = np.random.choice(n, size=2, replace=False)
    res.gene[idx1], res.gene[idx2] = res.gene[idx2] , res.gene[idx1]
    return res

def multi_mutate(candidate:Chromosome, num_mutate:int=2) -> Chromosome:
    # res = candidate.copy()
    res = candidate
    for _ in range(num_mutate):
        res = mutate(res)
    return res

def inversion(candidate:Chromosome) -> Chromosome:
    # res = candidate.copy()
    res = candidate
    n = len(res.gene)
    idx1, idx2 = np.sort(np.random.choice(n, 2, replace=False))
    # res.gene[idx1:idx2+1] = res.gene[idx2+1:idx1-1:-1]
    res.gene[idx1:idx2+1] = list(reversed(res.gene[idx1:idx2+1]))
    return res

def scramble(candidate:Chromosome) -> Chromosome:
    # res = candidate.copy()
    res = candidate
    n = len(res.gene)
    idx1, idx2 = np.sort(np.random.choice(n, 2, replace=False))
    res.gene[idx1:idx2+1] = list(np.random.shuffle(res[idx1:idx2+1]))
    return res


class SingleSwap(Mutate):
    def __init__(self, distance_metric) -> None:
        super().__init__(distance_metric)
    
    def forward(self, x):
        return self.fitness_fn(self.mutate(self,x))
    
    def mutate(self, candidate:Chromosome) -> Chromosome:
        res = candidate
        n = len(res.gene)
        idx1, idx2 = np.random.choice(n, size=2, replace=False)
        res.gene[idx1], res.gene[idx2] = res.gene[idx2] , res.gene[idx1]
        return res

class MultiSwap(Mutate):
    def __init__(self, distance_metric) -> None:
        super().__init__(distance_metric)
    
    def forward(self, x):
        return self.fitness_fn(self.mutate(self,x))
    
    def mutate(candidate:Chromosome, num_mutate:int=2) -> Chromosome:
        # res = candidate.copy()
        res = candidate
        for _ in range(num_mutate):
            res = mutate(res)
        return res

class Inversion(Mutate):
    def __init__(self, distance_metric) -> None:
        super().__init__(distance_metric)
    
    def forward(self, x):
        return self.fitness_fn(self.mutate(self,x))
    
    def mutate(candidate:Chromosome) -> Chromosome:
        # res = candidate.copy()
        res = candidate
        n = len(res.gene)
        idx1, idx2 = np.sort(np.random.choice(n, 2, replace=False))
        # res.gene[idx1:idx2+1] = res.gene[idx2+1:idx1-1:-1]
        res.gene[idx1:idx2+1] = list(reversed(res.gene[idx1:idx2+1]))
        return res

class Scramble(Mutate):
    def __init__(self, distance_metric) -> None:
        super().__init__(distance_metric)
    
    def forward(self, x):
        return self.fitness_fn(self.mutate(self,x))
    
    def mutate(candidate:Chromosome) -> Chromosome:
        # res = candidate.copy()
        res = candidate
        n = len(res.gene)
        idx1, idx2 = np.sort(np.random.choice(n, 2, replace=False))
        res.gene[idx1:idx2+1] = list(np.random.shuffle(res[idx1:idx2+1]))
        return res