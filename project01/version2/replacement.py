import numpy as np
from .classes import Chromosome
from typing import List

def replace_firstweak(population:List[Chromosome],
                    candidate:Chromosome,
                    inplace=False) -> List[Chromosome]:
    pop_temp = population.copy()
    for i, pop in enumerate(pop_temp):
        if candidate.phenome > pop.phenome:
            pop_temp[i] = candidate
            return pop_temp
    return pop_temp

def replace_weakest(population:List[Chromosome],
                    candidate:Chromosome,
                    inplace=False) -> List[Chromosome]:
    pop_temp = population.copy()
    weakest_idx = np.argmin([x.phenome for x in pop_temp])
    if candidate.phenome > pop_temp[weakest_idx].phenome:
        pop_temp[weakest_idx] = candidate
    return pop_temp

    