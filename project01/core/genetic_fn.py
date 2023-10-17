from typing import Optional, Literal
import numpy as np
import torch.nn as nn

def cost_function(travel_list, distance_metric):
    total_distance = 0
    for i in range(len(travel_list)-1):
        total_distance += distance_metric[travel_list[i]][travel_list[i+1]]
    total_distance += distance_metric[travel_list[-1]][travel_list[0]]
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
        locus = np.random.randint(n)
    child1 = arg1[:locus] + arg2[locus:] 
    child2 = arg2[:locus] + arg1[locus:]
    return child1, child2

def fixed_crossover(arg1, arg2):
    assert len(arg1) == len(arg2) , "parents' size should be equal"
    n = len(arg1)
    point = np.random.randint(n)
    child1 = arg2[:point]
    child2 = arg1[:point]
    return child1 + [x for x in arg1 if x not in child1] , child2 + [x for x in arg2 if x not in child2]

def fixed_crossover_twopoint(arg1, arg2):
    assert len(arg1) == len(arg2) , "parents' size should be equal"
    n = len(arg1)
    point1 = np.random.randint(n-1)
    point2 = np.random.choice(range(point1,n+1))
    child1 = arg2[point1:point2+1]
    child2 = arg1[point1:point2+1]
    return child1 + [x for x in arg1 if x not in child1] , child2 + [x for x in arg2 if x not in child2]

def ordered_crossover(parent1, parent2, p1, p2):
    n = len(parent1)
    child1 = [-1] * n
    child2 = [-1] * n
    
    child1[p1:p2+1] = parent2[p1:p2+1]
    child2[p1:p2+1] = parent1[p1:p2+1]
    
    remaining1 = [x for x in parent1 if x not in child1]
    remaining2 = [x for x in parent2 if x not in child2]
    
    child1 = [remaining1.pop(0) if x == -1 else x for x in child1]
    child2 = [remaining2.pop(0) if x == -1 else x for x in child2]
    
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