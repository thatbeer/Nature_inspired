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

