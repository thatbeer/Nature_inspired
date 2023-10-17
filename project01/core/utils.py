import xml.etree.ElementTree as ET
import numpy as np
import itertools

# function to load weight metrics
def load_metrics(xml_path, info=False):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # meta-data
    name = root.find('name').text
    source = root.find('source').text
    description = root.find('description').text
    doublePrecision = int(root.find('doublePrecision').text)
    ignoredDigits = int(root.find('ignoredDigits').text)
    num_node = len(root.findall(".//vertex"))
    weights_metric = np.nan_to_num(np.identity(num_node) * -np.inf, 0.0)
    # weights_metric = np.identity(num_node)

    for i, vertex in enumerate(root.findall('.//vertex')):
        for edge in vertex.findall('.//edge'):
            cost = float(edge.get("cost"))
            node = int(edge.text)
            # print(f"line:{i} node:{node}->cost:{cost}")
            # if i == node:
            #     weights_metric[i,node] = -np.Inf
            # else:
            weights_metric[i,node] = cost
    if info is True:
        return weights_metric , (name, source, description, doublePrecision, ignoredDigits)

    return weights_metric

def cost_function(travel_list, distance_metric):
    total_distance = 0
    for i in range(len(travel_list)-1):
        total_distance += distance_metric[travel_list[i]][travel_list[i+1]]
    total_distance += distance_metric[travel_list[-1]][travel_list[0]]
    return total_distance

def fitness(travel_list, distance_metric):
    return 1 / cost_function(travel_list, distance_metric)

def create_candidate(distance_metric):
    return list(np.random.permutation(len(distance_metric)))

def generate_population(distance_metric, pop_size:int=10):
    return [create_candidate(distance_metric) for _ in range(pop_size)]

def tournament_selection(population, fitness_value, tournament_size):
    fit_value = -np.inf
    res = None
    for _ in range(tournament_size):
        idx = np.random.randint(len(population))
        # new_fit = fitness(population[idx], distance_metric=distance_metric)
        new_fit = fitness_value[idx]    
        if new_fit >= fit_value:
            fit_value = new_fit
            res = population[idx]
    return res, fit_value

def binary_tournament(population,fitness_value, tournament_size):
    parent1, _ = tournament_selection(population=population, fitness_value=fitness_value, tournament_size=tournament_size)
    parent2, _ = tournament_selection(population=population, fitness_value=fitness_value, tournament_size=tournament_size)
    return parent1 , parent2

def search(max_gens, pop_size, tour_size, co_fn, mut_fn, replace_fn, distance_metric):
    population = generate_population(distance_metric=distance_metric, pop_size=pop_size)
    pop_fitness = [fitness(pop, distance_metric=distance_metric) for pop in population]
    fit_avg = [np.mean(pop_fitness)]
    fit_upper = [np.max(pop_fitness)]
    for i in range(max_gens):
        # parent1 = tournament_selection(population=population, fitness_value=pop_fitness, tournament_size=tour_size)
        parent1, parent2 = binary_tournament(population=population, fitness_value=pop_fitness, tournament_size=tour_size)
        child1 , child2 = co_fn(parent1=parent1, parent2=parent2)
        child1 , child2 = mut_fn(child1) , mut_fn(child2)
        population = replace_fn(population=population,candidate=child1,distance_metric=distance_metric)
        population = replace_fn(population=population,candidate=child2,distance_metric=distance_metric)
        pop_fitness = [fitness(pop, distance_metric=distance_metric) for pop in population]
        fit_avg.append(np.mean(pop_fitness))
        fit_upper.append(np.max(pop_fitness))
        if i > 2:
            if fit_avg[i] - fit_avg[i-1]:
                print(f"improve avg fitness from {fit_avg[i-1]} -> {fit_avg[i]}")

    return population, pop_fitness , fit_avg, fit_upper


def find_combinations(max_generations,population_sizes,tour_size,crossover_functions,mutate_functions,replace_functions):
    return list(itertools.product(
    max_generations, population_sizes, tour_size,
    crossover_functions, mutate_functions, replace_functions
))

def create_parameter_list(max_generations,population_sizes,tour_size,crossover_functions,mutate_functions,replace_functions):
    from .classes import Parameters
    res = []
    parameter_combinations = find_combinations(max_generations,population_sizes,tour_size,crossover_functions,mutate_functions,replace_functions)
    for params in parameter_combinations:
    # print(Parameters(*params))
        res.append(Parameters(*params))
    return res
