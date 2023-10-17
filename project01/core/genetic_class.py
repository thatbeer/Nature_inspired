import torch.nn as nn
import numpy as np

from core import fitness
from typing import List

class Chromosome:
    def __init__(self, genes:List[int], distance_metric=distance_metric):
        self.genes = genes
        self.distance_metric = distance_metric
        self.phenomes = self.__fitness__(genes=genes)

    def __fitness__(self, genes:List[int]) -> int:
        return fitness(genes, distance_metric=self.distance_metric)

    def __str__(self):
        return f"Genes: {self.genes}\nPhenomes: {self.phenomes:.5f}"

class TSP_GENETICALGO(nn.Module):
    def __init__(self, distance_metric, population_size:int=10, max_iter:int=100, mutate_rate:float=0.2) -> None:
        super(TSP_GENETICALGO, self).__init__
        assert len(distance_metric) == len(distance_metric[0]), "the distance metric supposed to be identical"
        self.distance_metric = distance_metric
        self.num_node = len(distance_metric)
        self.population_size = population_size
        self.max_iter = max_iter
        self.mutate_rate = mutate_rate
        # self.population = np.zeros((population_size,self.num_node))
        self.population = self.generate_population(self.distance_metric,self.population_size)
        self.pop_fitness = [self.fitness(candidate) for candidate in self.population]
    
    def create_candidate(distance_metric):
        return list(np.random.permutation(len(distance_metric)))

    def generate_population(self, distance_metric, pop_size:int=10):
        return [self.create_candidate(distance_metric) for _ in range(pop_size)]

    def cost_function(self, candidate):
        total_cost = 0
        for i in range(len(candidate)):
            total_cost += self.distance_metric[candidate[i],candidate[i+1]]
        total_cost += self.distance_metric[-1,0]
        return total_cost

    def fitness(self, candidate):
        return 1 / self.cost_function(candidate)
    
    def forward(self, x):
        pass