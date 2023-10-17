import torch.nn as nn
import numpy as np

from dataclasses import dataclass
from typing import Optional, Literal, Callable, List
# from .utils import fitness_function, create_gene


@dataclass
class Parameters:
    max_gen : int
    pop_size : int
    tour_size : int
    cross_fn : Callable
    mutate_fn : Callable
    replace_fn : Callable

@dataclass
class Chromosome:
    gene : List[int]
    phenome : float

# class Genetic(nn.Module):
#     def __init__(self, distance_metric, p_mutate, p_crossover) -> None:
#         super().__init__()
#         self.distance_metric = distance_metric
#         self.p_mutate = p_mutate
#         self.p_crossover = p_crossover
#         self.population = None #should be list of Chromosome object
#         self.best_gene = None # should be best Chromosome object amoung the population
    
#     def generate_population(self, population_size):
#         res = []
#         for _ in range(population_size):
#             gene = create_gene(self.distance_metric)
#             phenome = fitness_function(gene=gene,distance_metric=self.distance_metric)
#             res.append(Chromosome(gene,phenome))
#         self.population = res

#     def select_two_parent(self, tour_size) -> [Chromosome,Chromosome]:
#         res = []
#         for _ in range(2):
#             idx = np.random.randint(len(self.population))
#             temp = self.population[idx]
#             for _ in range(tour_size-1):
#                 new_idx = np.random.randint(len(self.population))
#                 if self.population[new_idx].phenome > temp.phenome:
#                     temp = self.population[new_idx]
#             res.append(temp)
        
#         return res

        
    
#     def search(self,cfg):
#         # TODO : make search return everything we need to track upon experiencing the lab
#         pass

#     def forward(self,cfg):
#         return self.search(cfg)