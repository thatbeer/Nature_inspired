from typing import Callable, List
from dataclasses import dataclass
import numpy as np

@dataclass
class Parameters:
    max_generations:int
    population_size:int
    tournament_size:int
    p_select:float
    p_crossover:float
    p_mutate:0.2

@dataclass
class Functions:
    crossover_fn: Callable
    mutate_fn: Callable
    replace_fn: Callable
    
@dataclass
class Config:
    name : str
    path : str
    file_name : str
    num_trial: 10
    parameters: Parameters
    functions: Functions

@dataclass
class Chromosome:
    gene : List[int]
    phenome : float 
    length : int = 0

    def __post_init__(self):
        self.length = len(self.gene)
        ## TODO : create a post initial for calculate fitness value

@dataclass
class Population:
    pop : List[Chromosome]
    best_gene : Chromosome = None
    best_fitness : float = 0
    avg_fitness : float = 0
    
    def __post_init__(self):
        self.best_gene = max(self.pop, key= lambda x : x.phenome)
        self.best_fitness = self.best_gene.phenome
        self.avg_fitness = np.mean([x.phenome for x in self.pop])

@dataclass
class SimpleLog:
    best_gene: Chromosome
    best_fitness : float
    avg_fitness: float

@dataclass
class Log:
    best_gene: Chromosome
    best_fitness : List[float]
    avg_fitness: List[float]
    first_generation: List[Chromosome]
    last_generation: List[Chromosome]
    generations: List[List[Chromosome]]

@dataclass
class AdvanceLog:
    gens : int
    best_fitness : float
    avg_fitness : float
    best_gene : List[int]
@dataclass
class GenerationLog:
    gens : int # 100 , 200 , 300 , 500 , 800 , 1000


@dataclass
class TriaLog:
    trial : int
    information : Log | SimpleLog
    # avg_best_fitness : float = 0

    # def __post_init__(self):
    #     if self.trial > 1:
    #         self.avg_best_fitness = np.mean(self.information.best_fitness)
    

    