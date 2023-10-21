import xml.etree.ElementTree as ET
import itertools
import numpy as np  
from typing import List



def load_metrics(xml_path, info=False) -> List[List[int]]:
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

    for i, vertex in enumerate(root.findall('.//vertex')):
        for edge in vertex.findall('.//edge'):
            cost = float(edge.get("cost"))
            node = int(edge.text)
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

def pop_stats(population:List[List[int]]) -> (float,(int,float)):
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

# def create_parameter_list(combinations) -> List[Parameters]:
#     res = []
#     # parameter_combinations = find_combinations(max_generations,population_sizes,tour_size,crossover_functions,mutate_functions,replace_functions)
#     for params in combinations:
#         res.append(Parameters(*params))
#     return res