from typing import List, Any, Callable
import numpy as np
from . import best_gene, generate_population, parents_selection, create_chromosome, pop_stats

def search(distance_metric: List[List[int]],
        max_gens:int,
        pop_size:int,
        tour_size:int, 
        co_fn:Callable,
        mut_fn:Callable,
        replace_fn:Callable) -> Any:
    population = generate_population(population_size=pop_size,
                                     distance_metric=distance_metric)
    ancestor = population
    avg_fitness = [np.mean([pop.phenome for pop in population])]
    max_fitness = [np.max([pop.phenome for pop in population])]
    best_candidate = best_gene(population=population)
    for i in range(1,max_gens+1):
        parents = parents_selection(population=population, tournament_size=tour_size)
        parent1 , parent2 = parents[0], parents[1]
        gene1, gene2 = co_fn(parent1,parent2)
        child1 , child2 = create_chromosome(gene1,distance_metric=distance_metric), create_chromosome(gene2,distance_metric=distance_metric)
        child1, child2 = mut_fn(child1), mut_fn(child2)
        population = replace_fn(population,child1)
        population = replace_fn(population,child2)
        stats = pop_stats(population=population)
        avg_fitness.append(stats[0])
        max_fitness.append(stats[1][1])
        new_best_candidate = best_gene(population=population)

        if new_best_candidate.phenome > best_candidate.phenome:
            best_candidate = new_best_candidate
        # if i % 100 == 0:
        #     print(f"Processing {i} Steps")
    print("evolved through {} generations".format(i))
    return {
        "lasted_population" : population,
        "avg_fitness" : avg_fitness,
        "max_fitness" : max_fitness,
        "last_generation" : i,
        "best_candidate" : best_candidate,
        "first_population" : ancestor
    }


# from typing import List, Callable, Any

# def search(distance_metric: List[List[int]],
#         max_gens:int,
#         pop_size:int,
#         tour_size:int, 
#         co_fn:Callable,
#         mut_fn:Callable,
#         replace_fn:Callable) -> Any:
#     population = generate_population(population_size=pop_size,
#                                      distance_metric=distance_metric)
#     avg_fitness = [np.mean([pop.phenome for pop in population])]
#     max_fitness = [np.max([pop.phenome for pop in population])]
#     best_candidate = best_gene(population=population)
#     for i in range(max_gens):
#         parents = parents_selection(population=population, tournament_size=tour_size)
#         parent1 , parent2 = parents[0], parents[1]
#         gene1, gene2 = co_fn(parent1,parent2)
#         child1 , child2 = create_chromosome(gene1,distance_metric=distance_metric), create_chromosome(gene2,distance_metric=distance_metric)
#         child1, child2 = mut_fn(child1), mut_fn(child2)
#         population = replace_fn(population,child1)
#         population = replace_fn(population,child2)
#         stats = pop_stats(population=population)
#         avg_fitness.append(stats[0])
#         max_fitness.append(stats[1][1])
#         new_best_candidate = best_gene(population=population)

#         if new_best_candidate.phenome > best_candidate.phenome:
#             best_candidate = new_best_candidate
#         if i % 100:
#             print("Processing {i} Steps")
#         if (i > 2) and (avg_fitness[i] - avg_fitness[i-1] < 1e-5):
#             return population , avg_fitness , max_fitness , i ,best_candidate

#     return population , avg_fitness , max_fitness , i ,best_candidate



class Genetic:
    def __init__(self) -> None:
        self.x = None