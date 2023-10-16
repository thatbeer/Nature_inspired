import math
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
        # new_fit = fitness(population[idx], distance_metric=weight_metric)
        new_fit = fitness_value[idx]    
        if new_fit >= fit_value:
            fit_value = new_fit
            res = population[idx]
        
    return res, fit_value




