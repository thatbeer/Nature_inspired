import numpy as np
import torch.nn as nn
# from . import Chromosome, Fitness
from typing import List, Any, Callable

@dataclass
class Chromosome:
    gene : List[int]
    phenome : float 
    length : int = 0

    def __post_init__(self):
        self.length = len(self.gene)
        ## TODO : create a post initial for calculate fitness value
        self.phenome = fitness

class Mutate(nn.Module):
    def __init__(self, distance_metric) -> None:
        super().__init__()
        self.fitness_fn = Fitness(distance_metric=distance_metric)
    
    def forward(self, gene:Chromosome) -> Chromosome:
        pass

class Crossover(nn.Module):
    def __init__(self, distance_metric) -> None:
        super().__init__()
        self.fitness_fn = Fitness(distance_metric=distance_metric)
        
    def forward(self, parents: List[Chromosome]) -> List[Chromosome]:
        pass

class Replacement(nn.Module):
    def __init__(self, distance_metric) -> None:
        super().__init__()
        self.fitness_fn = Fitness(distance_metric=distance_metric)
        
    def forward(self, x : (List[Chromosome], Chromosome)) -> List[Chromosome]:
        pass

