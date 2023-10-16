from typing import Optional, Literal
import numpy as np
import torch.nn as nn

def cost_function(travel_list, distance_metric):
    total_distance = 0
    for i in range(len(travel_list)-1):
        total_distance += distance_metric[travel_list[i],travel_list[i+1]]
    total_distance += distance_metric[-1,0]
    return total_distance

def fitness(travel_list, distance_metric):
    return 1 / cost_function(travel_list, distance_metric)

def create_candidate(distance_metric):
    return list(np.random.permutation(len(distance_metric)))

def generate_population(distance_metric, pop_size:int=10):
    return [create_candidate(distance_metric) for _ in range(pop_size)]

def tournament_selection(population, fitness_value, tournament_size):
    fit_value = -np.inf
    res = None
    for _ in range(tournament_size):
        idx = np.random.randint(len(population))
        # new_fit = fitness(population[idx], distance_metric=distance_metric)
        new_fit = fitness_value[idx]    
        if new_fit >= fit_value:
            fit_value = new_fit
            res = population[idx]
        
    return res, fit_value

def crossover(arg1, arg2, locus:Optional[int]=None):
    n = len(arg1)
    if locus is None:
        locus = np.random.randint(n-1)
    child1 = arg1[:locus] + arg2[locus:] 
    child2 = arg2[:locus] + arg1[locus:]
    return child1, child2

def mutate(arg):
    """
    Mutate (swap operator): Swapping between 2 random indexes in a list.

    Args:
        arg (list): The list to perform the mutation on.

    Returns:
        list: The mutated list with elements swapped.
    """
    n = len(arg)
    if n < 2:
        # Nothing to swap in a list with less than 2 elements
        return arg
    idx1, idx2 = np.random.choice(n, size=2, replace=False)
    res = arg.copy()  
    res[idx1], res[idx2] = res[idx2], res[idx1]  
    return res

def replace_firstweak(population, candidate, distance_metric):
    pop_temp = population.copy()
    cand_fitness = fitness(candidate, distance_metric)
    for i, pop in enumerate(pop_temp):
        if cand_fitness >= fitness(pop, distance_metric):
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