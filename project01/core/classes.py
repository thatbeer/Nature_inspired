from dataclasses import dataclass
from typing import Optional, Literal, Callable, List
from .utils import fitness
import torch.nn as nn

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

## OLD CLASS
# class Chromosome:
#     def __init__(self, genes:List[int], distance_metric):
#         self.distance_metric = distance_metric
#         self.genes = genes
#         self.phenomes = self.__fitness__(genes=genes)

#     def __fitness__(self, genes:List[int]) -> int:
#         return fitness(genes, distance_metric=self.distance_metric)

#     def __str__(self):
#         return f"Genes: {self.genes}\nPhenomes: {self.phenomes:.5f}"

class Genetic(nn.Module):
    def __init__(self, distance_metric, p_mutate, p_crossover) -> None:
        super().__init__()
        self.distance_metric = distance_metric
        self.p_mutate = p_mutate
        self.p_crossover = p_crossover
    
    def search(self,cfg):
        # TODO : make search return everything we need to track upon experiencing the lab
        pass

    def forward(self,cfg):
        return self.search(cfg)