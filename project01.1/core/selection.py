import numpy as np
from typing import List, Any

from .dataclass import Chromosome
from .utils import fitness_function

def create_gene(distance_metric:List[List[int]]) -> List[int]:
    return list(np.random.permutation(len(distance_metric)))

def create_chromosome(gene:List[int], distance_metric:List[List[int]]) -> Chromosome:
    phenome = fitness_function(gene=gene,distance_metric=distance_metric)
    return Chromosome(gene,phenome)

def generate_population(population_size:int, distance_metric:List[List[int]]) -> List[Chromosome]:
    res = []
    for _ in range(population_size):
        gene = create_gene(distance_metric)
        phenome = fitness_function(gene=gene,distance_metric=distance_metric)
        res.append(Chromosome(gene,phenome))
    return res

def best_gene(population:List[Chromosome]) -> Chromosome:
    phenomes = [ x.phenome for x in population]
    idx = np.argmax(np.array(phenomes))
    return population[idx]

def get_mean(population:List[Chromosome]) -> float:
    phenomes = [ x.phenome for x in population]
    return np.mean(phenomes)

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