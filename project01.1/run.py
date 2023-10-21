from omegaconf import OmegaConf
from typing import Callable, List
from dataclasses import dataclass
import numpy as np
import fire
import torch.nn as nn
import yaml

from core.crossover import no_crossover, point_crossover, ordered_crossover, partialmap_crossover
from core.mutation import mutate, multi_mutate, inversion, scramble
from core.replacement import replace_firstweak, replace_weakest
from core.selection import generate_population, best_gene, parents_selection, create_chromosome
from core.utils import load_metrics

from core.classes import Speculator
from core.dataclass import Config, Parameters, Functions

# @dataclass
# class Parameters:
#     max_generations:int
#     population_size:int
#     tournament_size:int
#     p_select:float
#     p_crossover:float
#     p_mutate:0.2
# @dataclass
# class Functions:
#     crossover_fn: Callable
#     mutate_fn: Callable
#     replace_fn: Callable
# @dataclass
# class Config:
#     path : str
#     num_trial: 10
#     parameters: Parameters
#     functions: Functions

# def load_fn(fn_name):
#     # Use a dictionary to map function names to their corresponding functions
#     functions = {
#         "mutate": mutate,
#         "multi_mutate": multi_mutate,
#         "inversion": inversion,
#         "scramble": scramble,
#         "no_crossover": no_crossover,
#         "point_crossover": point_crossover,
#         "ordered_crossover": ordered_crossover,
#         "partialmap_crossover": partialmap_crossover,
#         "replace_firstweak": replace_firstweak,
#         "replace_weakest": replace_weakest
#     }
    
#     # Check if the provided function name is in the dictionary
#     if fn_name in functions:
#         return functions[fn_name]
#     else:
#         raise FileNotFoundError("Are you sure that you named a correct function?")

# class GeneticAlgorithm(nn.Module):
#     def __init__(self, config) -> None:
#         super(GeneticAlgorithm, self).__init__()
#         self.distance_metric = load_metrics(config.path)
#         self.num_city = self.distance_metric.shape[0]
#         self.num_trial = config.num_trial
#         self.params = config.parameters
#         self.mutate = load_fn(config.functions.mutate_fn)
#         self.crossover = load_fn(config.functions.crossover_fn)
#         self.replacement = load_fn(config.functions.replace_fn)
#         self.population = self._generate_population()
    
#     def _generate_population(self):
#         return generate_population(
#             population_size=self.params.population_size,
#             distance_metric=self.distance_metric
#         )

#     def _create_chromosome(self, gene):
#         return create_chromosome(gene=gene,distance_metric=self.distance_metric)
    
#     def _mutate(self, gene):
#         return self.mutate(gene, self.params.p_mutate)
    
#     def _crossover(self, parent1, parent2):
#         return self.crossover(parent1, parent2, self.params.p_crossover)

#     def _replacement(self, population, candidate):
#         return self.replacement(population, candidate)
    
#     def forward(self):
#         population = generate_population(
#             population_size=self.params.population_size,
#             distance_metric=self.distance_metric
#         )
#         best_one = best_gene(population=population)
#         for i in range(self.params.max_generations):
#             population, _ = self.evolve(population_origin=population)
#             new_best = best_gene(population)
#             if new_best.phenome > best_one.phenome:
#                 best_one = new_best

#         return population, best_one

#     def evolve(self, population_origin):
#         populationx = population_origin.copy()
#         parents = parents_selection(population=populationx,
#                                     tournament_size=self.params.tournament_size)
#         gene1, gene2 = self._crossover(parents[0], parents[1])
#         child1, child2 = self._create_chromosome(gene1), self._create_chromosome(gene2)
#         child1, child2 = self._mutate(child1), self._mutate(child2)
#         for child in [child1,child2]:
#             populationx = self._replacement(population=populationx,
#                                                   candidate=child)
#         return populationx
        
# class Speculator:
#     def __init__(self, config) -> None:
#         self.config = config

#     def run(self):
#         for _ in self.config.num_trial:
#             experiment = GeneticAlgorithm(self.config)
#             experiment.run()
#         return [0]

def run_genetic_algorithm(config_path):
    # config = OmegaConf.load(config_path)
    # genetic_algorithm = GeneticAlgorithm(config)
    # genetic_algorithm.run_genetic_algorithm()
    with open(config_path,"r") as config_file:
        config_data = yaml.safe_load(config_file)
    
    config = Config(
        path=config_data["path"],
        num_trial=config_data["num_trial"],
        parameters=Parameters(**config_data["parameters"]),
        functions=Functions(**config_data["functions"])
    )

    speculator = Speculator(config)
    speculator.run()

if __name__ == "__main__":
    fire.Fire(run_genetic_algorithm)

# Usage
# if __name__ == "__main__":
#     # Assuming you have a valid `config` object created somewhere
#     speculator = Speculator(config)
#     speculator.run()