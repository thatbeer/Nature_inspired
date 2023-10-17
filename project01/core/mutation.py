import numpy as np


class Mutation:

    def __init__(self,mutate_rate, operator:str='point'):
        self.mutate_rate = mutate_rate
        self.operator = operator
    
    def point_swap(self, candidate):
        pass

    def inversion(self, candidate):
        pass

    def multi_swap(self, candidate):
        pass

def mutate(arg):
    n = len(arg)
    if n < 2:
        return arg
    idx1, idx2 = np.random.choice(n, size=2, replace=False)
    res = arg.copy()  
    res[idx1], res[idx2] = res[idx2], res[idx1]  
    return res

def multiple_mutate(arg, num_mutate:int=2):
    temp = arg.copy()
    for _ in range(num_mutate):
        temp = mutate(temp)
    return temp

def inversion(candidate):
    temp = candidate.copy()
    n = len(temp)
    idx1, idx2 = np.random.choice(n, size=2, replace=False)
    cutoff = temp[idx1:idx2+1]
    temp[idx1:idx2] = cutoff[::-1]
    return temp

def scramble(candidate):
    temp = candidate.copy()
    n = len(temp)
    idx1, idx2 = np.random.choice(n, size=2, replace=False)
    scrambled = np.random.shuffle(temp[idx1:idx2+1])
    temp[idx1:idx2+1] = scrambled
    return temp

