import numpy as np
from typing import Optional, List


def crossover(parent1, parent2, locus:Optional[int]=None):
    assert len(parent1) == len(parent2) , "parents' size should be equal"
    n = len(parent1)
    if locus is None:
        locus = np.random.randint(n)
    child1 = parent1[:locus] + parent2[locus:] 
    child2 = parent2[:locus] + parent1[locus:]
    assert (len(set(child1)) == n ) and (len(set(child2)) == n ) , "found duplicated node inside child"
    return child1, child2

# def fixed_crossover(parent1, parent2):
#     assert len(parent1) == len(parent2) , "parents' size should be equal"
#     n = len(parent1)
#     point = np.random.randint(n)
#     child1 = parent2[:point]
#     child2 = parent1[:point]
#     return child1 + [x for x in parent1 if x not in child1] , child2 + [x for x in parent2 if x not in child2]

def fixed_crossover(arg1, arg2):
    assert len(arg1) == len(arg2) , "parents' size should be equal"
    n = len(arg1)
    point = np.random.randint(n)
    # point = 3
    child1 = [None] * n
    child2 = [None] * n
    child1[point:] = arg2[point:]
    child2[point:] = arg1[point:]

    for i in range(point):
        if arg1[i] not in child1:
            child1[i] = arg1[i]
        if arg2[i] not in child2:
            child2[i] = arg2[i]

    remain1 = [x for x in arg1 if x not in child1]
    remain2 = [x for x in arg2 if x not in child2]
    
    for i in range(n):
        if child1[i] == None:
            child1[i] = remain1.pop(0)
        if child2[i] == None:
            child2[i] = remain2.pop(0)
    return child1 , child2

def fixed_crossover_twopoint(parent1, parent2):
    assert len(parent1) == len(parent2) , "parents' size should be equal"
    n = len(parent1)
    point1 = np.random.randint(n-1)
    point2 = np.random.choice(range(point1,n+1))
    child1 = parent2[point1:point2+1]
    child2 = parent1[point1:point2+1]
    return child1 + [x for x in parent1 if x not in child1] , child2 + [x for x in parent2 if x not in child2]

def ordered_crossover(parent1, parent2):
    assert len(parent1) == len(parent2) , "parents' size should be equal"
    n = len(parent1)
    child1 = [-1] * n
    child2 = [-1] * n

    start, end = np.sort(np.random.choice(n, 2, replace=False))
    
    child1[start:end+1] = parent2[start:end+1]
    child2[start:end+1] = parent1[start:end+1]

    mapping_range = list(set(range(n)) - set(range(start,end+1)))
    for idx in mapping_range:
            if parent1[idx] not in child1:
                child1[idx] = parent1[idx]
            if parent2[idx] not in child2:
                child2[idx] = parent2[idx]


    remain1 = [x for x in parent1 if x not in child1]
    remain2 = [x for x in parent2 if x not in child2]

    child1 = [remain1.pop(0) if x == -1 else x for x in child1 ]
    child2 = [remain2.pop(0) if x == -1 else x for x in child2 ]

    return child1 , child2

def partialMap_crossover(parent1, parent2):
    assert len(parent1) == len(parent2) , "parents' size should be equal"
    n = len(parent1)
    point1, point2 = np.sort(np.random.choice(n, 2, replace=False)) 
    child1 = [None] * n
    child2 = [None] * n

    child1[point1:point2+1] = parent2[point1:point2+1]
    child2[point1:point2+1] = parent1[point1:point2+1]

    for i in range(n):
        if i in list(range(point1,point2+1)):
            continue
        
        if parent1[i] not in child1:
            child1[i] = parent1[i]
    
        if parent2[i] not in child2:
            child2[i] = parent2[i]

    remain1 = [x for x in parent1 if x not in child1]
    remain2 = [x for x in parent2 if x not in child2]
    
    for i in range(n):
        if child1[i] == None:
            child1[i] = remain1.pop(0)
        if child2[i] == None:
            child2[i] = remain2.pop(0)

    return child1 , child2

