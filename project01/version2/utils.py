import numpy as np
import itertools
from typing import List, Callable
from .classes import Chromosome, Parameters

def find_combinations(max_generations,population_sizes,tour_size,crossover_functions,mutate_functions,replace_functions):
    return list(itertools.product(
    max_generations, population_sizes, tour_size,
    crossover_functions, mutate_functions, replace_functions
))

def create_parameter_list(max_generations,population_sizes,tour_size,crossover_functions,mutate_functions,replace_functions):
    res = []
    parameter_combinations = find_combinations(max_generations,population_sizes,tour_size,crossover_functions,mutate_functions,replace_functions)
    for params in parameter_combinations:
        res.append(Parameters(*params))
    return res

def create_gene(distance_metric:List[List[int]]):
    return list(np.random.permutation(len(distance_metric)))

def generate_population(population_size, distance_metric:List[List[int]]):
    res = []
    for _ in range(population_size):
        gene = create_gene(distance_metric)
        phenome = fitness_function(gene=gene,distance_metric=distance_metric)
        res.append(Chromosome(gene,phenome))
    return res

def cost_function(gene:List[int],distance_metric:List[List[int]]):
    total_distance = 0
    for i in range(len(gene)-1):
        total_distance += distance_metric[gene[i]][gene[i+1]]
    total_distance += distance_metric[gene[-1]][gene[0]]
    return total_distance

def fitness_function(gene:List[int], distance_metric:List[List[int]]) -> float:
    return 1.0 / cost_function(gene, distance_metric)

def tournament_selection(population:List[Chromosome], tournament_size:int) -> Chromosome:
    fit_value = -np.inf
    res = None
    for _ in range(tournament_size):
        idx = np.random.randint(len(population))
        new_fit = population[idx].phenome 
        if new_fit >= fit_value:
            fit_value = new_fit
            res = population[idx]
    return res

def parents_selection(population:List[Chromosome], tournament_size:int) -> [Chromosome,Chromosome]:
    return [tournament_selection(population,tournament_size) for _ in range(2)]

