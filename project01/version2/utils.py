import xml.etree.ElementTree as ET
import numpy as np
import itertools
from typing import List
from .classes import Chromosome, Parameters

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

def cost_function(gene:List[int],distance_metric:List[List[int]]) -> float:
    total_distance = 0
    for i in range(len(gene)-1):
        total_distance += distance_metric[gene[i]][gene[i+1]]
    total_distance += distance_metric[gene[-1]][gene[0]]
    return total_distance

def fitness_function(gene:List[int], distance_metric:List[List[int]]) -> float:
    return 1.0 / cost_function(gene, distance_metric)


def create_gene(distance_metric:List[List[int]]) -> List[int]:
    return list(np.random.permutation(len(distance_metric)))

def create_chromosome(gene:List[int], distance_metric:List[List[int]]) -> Chromosome:
    phenome = fitness_function(gene=gene,distance_metric=distance_metric)
    return Chromosome(gene,phenome)

def generate_population(population_size:int, distance_metric:List[List[int]]) -> List[Chromosome]:
    res = []
    for _ in range(population_size):
        gene = create_gene(distance_metric)
        phenome = fitness_function(gene=gene,distance_metric=distance_metric)
        res.append(Chromosome(gene,phenome))
    return res

def best_gene(population:List[Chromosome]) -> Chromosome:
    phenomes = [ x.phenome for x in population]
    idx = np.argmax(np.array(phenomes))
    return population[idx]

def tournament_selection(population:List[Chromosome], tournament_size:int) -> Chromosome:
    fit_value = -np.inf
    res = None
    for _ in range(tournament_size):
        idx = np.random.randint(len(population))
        new_fit = population[idx].phenome 
        if new_fit >= fit_value:
            fit_value = new_fit
            res = population[idx]
    return res

def parents_selection(population:List[Chromosome], tournament_size:int) -> [Chromosome,Chromosome]:
    return [tournament_selection(population,tournament_size) for _ in range(2)]

def pop_stats(population:List[Chromosome]) -> (float,(int,float)):
    # TODO : return nessessary statisitc values
    phenomes_list = [x.phenome for x in population]
    avg = np.mean(phenomes_list)
    max_idx = np.argmax(phenomes_list)
    return avg , (max_idx, phenomes_list[max_idx])

def find_combinations(max_generations,population_sizes,tour_size,crossover_functions,mutate_functions,replace_functions):
    return list(itertools.product(
    max_generations, population_sizes, tour_size,
    crossover_functions, mutate_functions, replace_functions
))

def create_parameter_list(combinations) -> List[Parameters]:
    res = []
    # parameter_combinations = find_combinations(max_generations,population_sizes,tour_size,crossover_functions,mutate_functions,replace_functions)
    for params in combinations:
        res.append(Parameters(*params))
    return res