from dataclasses import dataclass
from typing import Callable, List, Tuple



@dataclass
class City:
    index : int
    X : float
    Y : float

@dataclass
class Item:
    index : int
    Profit : int
    Weight : int
    Node : int
    
@dataclass
class TTP:
    Name :str = None
    DTYPE : str = None
    Dimension : int = 0
    ITEMS : int = 0
    CAPACITY : int = 0
    MIN_SPEED : float = 0
    MAX_SPEED : float = 0
    RENTING_RATIO : float = 0
    EDGE_W : str = None
    NODE : List[City] = None
    ITEM : List[Item] = None

