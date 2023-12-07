from dataclasses import dataclass
from typing import Callable, List, Tuple




@dataclass
class Node:
    next_node : Node
    pheromone : int
    