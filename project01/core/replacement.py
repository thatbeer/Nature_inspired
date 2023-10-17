import numpy as np
from .utils import fitness


def replace_firstweak(population, candidate, distance_metric):
    pop_temp = population.copy()
    cand_fitness = fitness(candidate, distance_metric)
    for i, pop in enumerate(pop_temp):
        if cand_fitness > fitness(pop, distance_metric):
            pop_temp[i] = candidate
    return pop_temp

def replace_weakest(population, candidate, distance_metric):
    pop_temp = population.copy()
    cand_fitness = fitness(candidate, distance_metric)
    pop_fitness = [fitness(pop, distance_metric) for pop in pop_temp]
    idx = np.argsort(pop_fitness)
    for i in idx:
        if pop_fitness[i] <= cand_fitness:
            pop_temp[i] = candidate
            return pop_temp
        else:
            return pop_temp
    return pop_temp