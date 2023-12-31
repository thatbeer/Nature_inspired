import numpy as np
from typing import List
from itertools import chain
from .dataclass import Chromosome

def no_crossover(parent1:Chromosome, parent2:Chromosome) -> List[List[int]]:
    return parent1.gene, parent2.gene

def point_crossover(parent1:Chromosome, parent2:Chromosome, p_crossover:float=1.0) -> List[List[int]]:
    assert isinstance(parent1,Chromosome) and isinstance(parent2,Chromosome) , f"parent(s) should have instance of Chromosome class"
    if np.random.rand() > p_crossover:
        return no_crossover(parent1,parent2)
    # n = len(parent1.gene)
    n = parent1.length
    point = np.random.randint(n)

    child1, child2 = [None]*n , [None]*n
    child1[point:] = parent2.gene[point:]
    child2[point:] = parent1.gene[point:]
    
    for i in range(point):
        if parent1.gene[i] not in child1:
            child1[i] = parent1.gene[i]
        if parent2.gene[i] not in child2:
            child2[i] = parent2.gene[i]
    
    remain1 = [x for x in parent1.gene if x not in child1]
    remain2 = [x for x in parent2.gene if x not in child2]

    for i in range(n):
        if child1[i] == None:
            child1[i] = remain1.pop(0)
        if child2[i] == None:
            child2[i] = remain2.pop(0)
    
    return child1 , child2

# this is partial map crossover
def partialmap_crossover(parent1:Chromosome, parent2:Chromosome, p_crossover:float=1.0) -> List[List[int]]:
    assert isinstance(parent1,Chromosome) and isinstance(parent2,Chromosome) , f"parent(s) should have instance of Chromosome class"
    if np.random.rand() > p_crossover:
        return no_crossover(parent1,parent2)
    # n = len(parent1.gene)
    n = parent1.length
    child1 = [None] * n
    child2 = [None] * n

    start, end = np.sort(np.random.choice(n, 2, replace=False))
    
    child1[start:end+1] = parent2.gene[start:end+1]
    child2[start:end+1] = parent1.gene[start:end+1]

    mapping_range = list(set(range(n)) - set(range(start,end+1)))
    for idx in mapping_range:
        if parent1.gene[idx] not in child1:
            child1[idx] = parent1.gene[idx]
        if parent2.gene[idx] not in child2:
            child2[idx] = parent2.gene[idx]

    remain1 = [x for x in parent1.gene if x not in child1]
    remain2 = [x for x in parent2.gene if x not in child2]

    child1 = [remain1.pop(0) if x == None else x for x in child1 ]
    child2 = [remain2.pop(0) if x == None else x for x in child2 ]

    return child1 , child2

# Solution 1
def ordered_crossover(parent1: Chromosome, parent2: Chromosome, p_crossover: float = 1.0) -> List[List[int]]:
    # Initialize the offspring with empty lists
    assert isinstance(parent1, Chromosome) and isinstance(parent2, Chromosome), "parent(s) should have instance of Chromosome class"
    if np.random.rand() > p_crossover:
        return no_crossover(parent1,parent2)
    n = parent1.length
    child1 = [None] * n
    child2 = [None] * n

    # Fixed start and end points for testing
    start, end = np.sort(np.random.choice(n, 2, replace=False))

    child1[start:end + 1] = parent2.gene[start:end + 1]
    child2[start:end + 1] = parent1.gene[start:end + 1]

    ptr1, ptr2 = end + 1, end + 1

    for i in chain(range(end + 1, n), range(0, n)):  # Ensuring we iterate over every gene in the parent chromosomes
        if parent1.gene[i] not in child1:
            while child1[ptr1 % n] is not None:
                ptr1 += 1
            child1[ptr1 % n] = parent1.gene[i]
            ptr1 += 1

        if parent2.gene[i] not in child2:
            while child2[ptr2 % n] is not None:
                ptr2 += 1
            child2[ptr2 % n] = parent2.gene[i]
            ptr2 += 1

    return child1 , child2

# def ordered_crossover(parent1:Chromosome, parent2:Chromosome, p_crossover:float=1.0) -> List[List[int]]:
#     # Initialize the offspring with empty lists
#     assert isinstance(parent1,Chromosome) and isinstance(parent2,Chromosome) , f"parent(s) should have instance of Chromosome class"
#     if np.random.rand() > p_crossover:
#         return no_crossover(parent1,parent2)
#     n = parent1.length
#     child1 = [None] * n
#     child2 = [None] * n

#     start, end = np.sort(np.random.choice(n, 2, replace=False))

#     child1[start:end + 1] = parent1.gene[start:end + 1]
#     child2[start:end + 1] = parent2.gene[start:end + 1]

#     remain1 = [x for x in parent1.gene[end+1] if x not in child1] + [x for x in parent1.gene[:start+1] if x not in child1] + [x for x in parent1.gene[start:end+1] if x not in child1]
#     remain2 = [x for x in parent2.gene[end+1] if x not in child2] + [x for x in parent2.gene[:start+1] if x not in child2] + [x for x in parent2.gene[start:end+1] if x not in child2]

#     if child1[-1] == None:
#         child1[-1] = remain1.pop(0)
#     if child2[-1] == None:
#         child2 = remain2.pop(0)
    
#     child1 = [remain1.pop(0) if x == None else x for x in child1 ]
#     child2 = [remain2.pop(0) if x == None else x for x in child2 ]

#     return child1, child2

# def partialmap_crossover(parent1:Chromosome, parent2:Chromosome, p_crossover:float=1.0) -> List[List[int]]:
#     assert isinstance(parent1,Chromosome) and isinstance(parent2,Chromosome) , f"parent(s) should have instance of Chromosome class"
#     if np.random.rand() > p_crossover:
#         return no_crossover(parent1,parent2)
#     n = parent1.length
#     start, end = np.sort(np.random.choice(n, 2, replace=False))

#     child1 = [-1] * n
#     child2 = [-1] * n

#     child1[start:end+1] = parent2.gene[start:end+1]
#     child2[start:end+1] = parent1.gene[start:end+1]

#     for i in range(n):
#         if i in list(range(start,end+1)):
#             continue
    
#         if parent1.gene[i] not in child1:
#             child1[i] = parent1.gene[i]
        
#         if parent2.gene[i] not in child2:
#             child2[i] = parent2.gene[i]
    
#     remain1 = [x for x in parent1.gene if x not in child1]
#     remain2 = [x for x in parent2.gene if x not in child2]
    
#     for i in range(n):
#         if child1[i] == None:
#             child1[i] = remain1.pop(0)
#         if child2[i] == None:
#             child2[i] = remain2.pop(0)
    
#     return child1 , child2