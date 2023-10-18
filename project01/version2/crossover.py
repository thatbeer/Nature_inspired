import numpy as np
from .classes import Chromosome
from typing import List

def point_crossover(parent1:Chromosome, parent2:Chromosome) -> List[int]:
    assert isinstance(parent1,Chromosome) and isinstance(parent2,Chromosome) , f"parent(s) should have instance of Chromosome class"
    n = len(parent1.gene)
    point = np.random.randint(n)
    # print("point",point)
    # point = 2

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

def ordered_crossover(parent1:Chromosome, parent2:Chromosome) -> List[int]:
    assert isinstance(parent1,Chromosome) and isinstance(parent2,Chromosome) , f"parent(s) should have instance of Chromosome class"
    n = len(parent1.gene)
    child1 = [-1] * n
    child2 = [-1] * n

    start, end = np.sort(np.random.choice(n, 2, replace=False))
    
    child1[start:end+1] = parent2.gene[start:end+1]
    child2[start:end+1] = parent1.gene[start:end+1]

    mapping_range = list(set(range(n)) - set(range(start,end+1)))
    # print(mapping_range)
    for idx in mapping_range:
        if parent1.gene[idx] not in child1:
            child1[idx] = parent1.gene[idx]
        if parent2.gene[idx] not in child2:
            child2[idx] = parent2.gene[idx]

    remain1 = [x for x in parent1.gene if x not in child1]
    remain2 = [x for x in parent2.gene if x not in child2]

    child1 = [remain1.pop(0) if x == -1 else x for x in child1 ]
    child2 = [remain2.pop(0) if x == -1 else x for x in child2 ]

    return child1 , child2

def ordered_crossover_np(parent1:Chromosome, parent2:Chromosome) -> List[int]:
    assert isinstance(parent1,Chromosome) and isinstance(parent2,Chromosome) , f"parent(s) should have instance of Chromosome class"
    n = len(parent1.gene)
    child1 = [-1] * n
    child2 = [-1] * n

    start, end = np.sort(np.random.choice(n, 2, replace=False))
    
    child1[start:end+1] = parent2.gene[start:end+1]
    child2[start:end+1] = parent1.gene[start:end+1]

    # mapping_range = list(set(range(n)) - set(range(start,end+1)))
    mapping_range = np.where(np.array(child1) == -1)[0].tolist()
    for idx in mapping_range:
        if parent1.gene[idx] not in child1:
            child1[idx] = parent1.gene[idx]
        if parent2.gene[idx] not in child2:
            child2[idx] = parent2.gene[idx]

    remain1 = [x for x in parent1.gene if x not in child1]
    remain2 = [x for x in parent2.gene if x not in child2]
    # remain1 = np.where(np.array(child1) == -1)[0].tolist()
    # remain2 = np.where(np.array(child2) == -1)[0].tolist()

    child1 = [remain1.pop(0) if x == -1 else x for x in child1 ]
    child2 = [remain2.pop(0) if x == -1 else x for x in child2 ]

    return child1 , child2

def partialmap_crossover(parent1:Chromosome, parent2:Chromosome) -> List[int]:
    assert isinstance(parent1,Chromosome) and isinstance(parent2,Chromosome) , f"parent(s) should have instance of Chromosome class"
    n = parent1.length
    start, end = np.sort(np.random.choice(n, 2, replace=False))

    child1 = [-1] * n
    child2 = [-1] * n

    child1[start:end+1] = parent2.gene[start:end+1]
    child2[start:end+1] = parent1.gene[start:end+1]

    for i in range(n):
        if i in list(range(start,end+1)):
            continue
    
        if parent1[i] not in child1:
            child1[i] = parent1.gene[i]
        
        if parent2[i] not in child2:
            child2[i] = parent2.gene[i]
    
    remain1 = [x for x in parent1.gene if x not in child1]
    remain2 = [x for x in parent2.gene if x not in child2]
    
    for i in range(n):
        if child1[i] == None:
            child1[i] = remain1.pop(0)
        if child2[i] == None:
            child2[i] = remain2.pop(0)
    
    return child1 , child2