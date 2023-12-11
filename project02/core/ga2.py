import numpy as np
import math
import time

from dataclasses import dataclass
from typing import Any, List, Tuple, Dict


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

def read_problem(file_path:str):
    with open(file_path,'r') as file:
        lines = file.readlines()
    
    data = TTP(NODE=[],ITEM=[])
    
    for i , line in enumerate(lines):
        if line.startswith("PROBLEM NAME"):
            data.Name = line.split(':')[-1].strip()
        elif line.startswith("KNAPSACK DATA TYPE"):
            data.DTYPE = line.split(':')[-1].strip()
        elif line.startswith("DIMENSION"):
            data.Dimension = int(line.split(':')[-1].strip())
        elif line.startswith("NUMBER OF ITEMS"):
            data.ITEMS = int(line.split(':')[-1].strip())
        elif line.startswith("CAPACITY OF KNAPSACK"):
            data.CAPACITY = int(line.split(':')[-1].strip())
        elif line.startswith("MIN SPEED"):
            data.MIN_SPEED = float(line.split(':')[-1].strip())
        elif line.startswith("MAX SPEED"):
            data.MAX_SPEED = float(line.split(':')[-1].strip())
        elif line.startswith("RENTING RATIO"):
            data.RENTING_RATIO = float(line.split(':')[-1].strip())
        elif line.startswith("EDGE_WEIGHT_TYPE"):
            data.EDGE_W = line.split(':')[-1].strip()
        elif line.startswith("NODE_COORD_SECTION"):
            for j in range(1,data.Dimension+1):
                node = lines[i+j].split()
                data.NODE.append(City(index=int(node[0]),X=float(node[1]),Y=float(node[2])))
        elif line.startswith("ITEMS SECTION"):
            for j in range(1,data.ITEMS+1):
                item = lines[i+j].split()
                data.ITEM.append(
                    Item(int(item[0]),int(item[1]),int(item[2]),int(item[3]))
                )
        else:
            pass
    
    return data

def generate_ttp_solution(number_of_cities: int, items: List[Item], knapsack_capacity: int) -> Tuple[List[int], List[int]]:
    # Generate a random path (tour)
    path = np.random.permutation(number_of_cities) + 1

    # Initialize knapsack plan with no items picked
    plan = [0] * len(items)
    current_weight = 0

    # Randomly decide to pick up items considering the knapsack capacity
    for i, item in enumerate(items):
        item_weight = item.Weight
        if current_weight + item_weight <= knapsack_capacity:
            decision = np.random.choice([0, 1])
            plan[i] = decision
            current_weight += item_weight * decision

    return path.tolist(), plan

def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.ceil(np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2))

def calculate_time_and_profit(solution: List[int], plan: List[int], nodes: List[City], items: List[Item], min_speed, max_speed, max_weight):
    total_time = 0
    total_profit = 0
    current_weight = 0

    # Calculate the total travel time
    for i in range(len(solution)):
        current_city_index = solution[i]
        next_city_index = solution[0] if i == len(solution) - 1 else solution[i + 1]

        current_city = nodes[current_city_index - 1]
        next_city = nodes[next_city_index - 1]

        

        # Update current weight based on items picked at the current city
        for item, is_picked in zip(items, plan):
            if is_picked and item.Node == current_city_index:
                current_weight += item.Weight

        # Calculate speed based on current weight
        speed = max_speed - (current_weight / max_weight) * (max_speed - min_speed)
        speed = max(speed, min_speed)  # Ensure speed doesn't drop below minimum

        # Distance between current city and next city
        distance = euclidean_distance((current_city.X, current_city.Y), (next_city.X, next_city.Y))

        # Update time with time to next city
        total_time += distance / speed

        if current_weight > max_weight:
            return np.Inf , 0.0

    # Calculate total profit from picked items
    for item, is_picked in zip(items, plan):
        if is_picked:
            total_profit += item.Profit


    return total_time, total_profit


@dataclass
class Phenome:
    time : float
    profit : float
    net_profit : float

@dataclass
class Chromosome:
    path : List[int]
    plan : List[int]
    phenome : Phenome


# class GA:
#     def __init__(self,problem_data : TTP, population_size, iterations, tour_size, mut_prob=1.0, cross_prob=1.0) -> None:
#         self.problem_data = problem_data
#         self.iterations = iterations
#         self.population_size = population_size
#         self.tour_size = tour_size
#         self.mut_prob = mut_prob
#         self.cross_prob = cross_prob
#         self.population = self.init_population()
#         # self.population = [Chromosome(*generate_ttp_solution(
#         #     problem_data.Dimension,problem_data.ITEM,problem_data.CAPACITY
#         # )) for _ in range(population_size)]
#         self.best_chromosome = sorted(self.population, key=lambda c: c.phenome.net_profit)[-1]
#         self.best_history = [self.best_chromosome.phenome.net_profit]
#         self.avg_history = [sum(c.phenome.net_profit for c in self.population) / len(self.population)]
#         self.time_execute = 0

#     def init_population(self):
#         pop_temp = []
#         for _ in range(self.population_size):
#             path , plan = generate_ttp_solution(self.problem_data.Dimension,self.problem_data.ITEM,self.problem_data.CAPACITY)
#             time , profit = calculate_time_and_profit(path,plan,self.problem_data.NODE,self.problem_data.ITEM,self.problem_data.MIN_SPEED,self.problem_data.MAX_SPEED,self.problem_data.CAPACITY)
#             net_profit = profit - (time*self.problem_data.RENTING_RATIO)
#             phenome = Phenome(time,profit,net_profit)
#             chorm = Chromosome(path,plan,phenome)
#             pop_temp.append(chorm)
#         return pop_temp

#     def run(self):
#         start = time.time()
#         for _ in range(self.iterations):
#             p1 = self.tournament_selection(self.population,self.tour_size)
#             p2 = self.tournament_selection(self.population,self.tour_size)
#             if np.random.rand() < self.cross_prob:
#                 p1, p2 = self.ordered_crossover(p1,p2)
#             if np.random.rand() < self.mut_prob:
#                 p1 = self.inversion_mutation(p1)
#                 p2 = self.inversion_mutation(p2)
#             self.population = self.replace_weakest(self.population, p1)
#             self.population = self.replace_weakest(self.population, p2)

#             current_best = sorted(self.population, key=lambda c: c.phenome.net_profit)[-1]
#             if current_best.phenome.net_profit > self.best_chromosome.phenome.net_profit:
#                 self.best_chromosome = current_best
            
#             self.best_history.append(self.best_chromosome.phenome.net_profit)
#             self.avg_history.append(sum(c.phenome.net_profit for c in self.population) / len(self.population))
#         end = time.time()
#         self.time_execute = end- start
#         return self.population , current_best
    
    
#     def ordered_crossover(self,parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
#         # Select crossover points for the path
#         start, end = sorted(np.random.choice(range(len(parent1.path)), 2))

#         # Create segments from parents
#         parent1_segment = parent1.path[start:end]
#         parent2_segment = parent2.path[start:end]

#         # Create offspring paths excluding parent segments
#         offspring1_path = [city for city in parent2.path if city not in parent1_segment]
#         offspring2_path = [city for city in parent1.path if city not in parent2_segment]

#         # Insert parent segments into offspring paths
#         offspring1_path[start:start] = parent1_segment
#         offspring2_path[start:start] = parent2_segment

#         # For the plan, using a simple one-point crossover
#         crossover_point = np.random.randint(1, len(parent1.plan) - 1)
#         offspring1_plan = parent1.plan[:crossover_point] + parent2.plan[crossover_point:]
#         offspring2_plan = parent2.plan[:crossover_point] + parent1.plan[crossover_point:]

#         time1 , profit1 = calculate_time_and_profit(offspring1_path,offspring1_plan,self.problem_data.NODE,self.problem_data.ITEM,self.problem_data.MIN_SPEED,self.problem_data.MAX_SPEED,self.problem_data.CAPACITY)
#         net_profit1 = profit1 - (time1*self.problem_data.RENTING_RATIO)
#         phenome1 = Phenome(time1,profit1,net_profit1)

#         time2 , profit2 = calculate_time_and_profit(offspring2_path,offspring2_plan,self.problem_data.NODE,self.problem_data.ITEM,self.problem_data.MIN_SPEED,self.problem_data.MAX_SPEED,self.problem_data.CAPACITY)
#         net_profit2 = profit2 - (time2*self.problem_data.RENTING_RATIO)
#         phenome2 = Phenome(time2,profit2,net_profit2)

#         offspring1 = Chromosome(offspring1_path, offspring1_plan, phenome1)
#         offspring2 = Chromosome(offspring2_path, offspring2_plan, phenome2)       

#         # # Create new Chromosome instances for offspring
#         # offspring1 = Chromosome(offspring1_path, offspring1_plan)
#         # offspring2 = Chromosome(offspring2_path, offspring2_plan)

#         return offspring1, offspring2

#     def inversion_mutation(self,chromosome: Chromosome):
#         # Ensure there are at least two elements in the path
#         if len(chromosome.path) < 2:
#             return chromosome
#         path = chromosome.path.copy()
#         plan = chromosome.plan.copy()
#         # Choose two distinct random positions in the path
#         pos1, pos2 = sorted(np.random.choice(range(len(chromosome.path)), 2))

#         # Invert the order of elements between pos1 and pos2
#         path[pos1:pos2 + 1] = reversed(path[pos1:pos2 + 1])
#         plan[pos1:pos2 + 1] = reversed(plan[pos1:pos2 + 1])

#         new_time , new_profit = calculate_time_and_profit(path,plan,self.problem_data.NODE,self.problem_data.ITEM,self.problem_data.MIN_SPEED,self.problem_data.MAX_SPEED,self.problem_data.CAPACITY)
#         new_net_profit = new_profit - (new_time*self.problem_data.RENTING_RATIO)
#         phenome = Phenome(new_time,new_profit,new_net_profit)

#         return Chromosome(path,plan,phenome)

#     def replace_weakest(self,population : List[Chromosome], candidates:Chromosome):
#         keys = [x.phenome.net_profit for x in population]
#         weakest_index = np.argmin(keys)

#         if candidates.phenome.net_profit > population[weakest_index].phenome.net_profit:
#             population[weakest_index] = candidates 

#         return population
#     def tournament_selection(self,population: List[int], tournament_size: int) -> Chromosome:
#         """
#         Selects a single Chromosome from the population using tournament selection.

#         :param population: An instance of the Population class containing Chromosomes.
#         :param tournament_size: The number of Chromosomes to be selected for each tournament.
#         :return: The winning Chromosome with the highest net profit.
#         """
#         # Ensure the tournament size is not larger than the population size
#         tournament_size = min(tournament_size, len(population))
        
#         # Randomly select 'tournament_size' individuals from the population
#         tournament_contestants = np.random.choice(population, size=tournament_size, replace=False)
        
#         # Determine the winner based on the highest net profit
#         winner = max(tournament_contestants, key=lambda chromo: chromo.phenome.net_profit)
        
#         return winner


def two_opt_swap(path, i, k):
    """ Perform a 2-opt swap by reversing the path segment between i and k """
    new_path = path[:i] + path[i:k+1][::-1] + path[k+1:]
    return new_path

def apply_two_opt_local_search(path, node_data, min_speed, max_speed, capacity):
    """ Apply 2-opt local search to improve the path """
    improved = True
    while improved:
        improved = False
        for i in range(1, len(path) - 2):
            for k in range(i + 1, len(path)):
                new_path = two_opt_swap(path, i, k)
                # Calculate new time and check if it's an improvement
                new_time, _ = calculate_time_and_profit(new_path, plan, node_data, item_data, min_speed, max_speed, capacity)
                current_time, _ = calculate_time_and_profit(path, plan, node_data, item_data, min_speed, max_speed, capacity)
                if new_time < current_time:
                    path = new_path
                    improved = True
                    break  # Improvement found, exit inner loop
            if improved:
                break  # Improvement found, exit outer loop
    return path

class GA:
    def __init__(self,problem_data : TTP, population_size, iterations, tour_size, mut_prob=1.0, cross_prob=1.0) -> None:
        self.problem_data = problem_data
        self.iterations = iterations
        self.population_size = population_size
        self.tour_size = tour_size
        self.mut_prob = mut_prob
        self.cross_prob = cross_prob
        self.population = self.init_population()
        # self.population = [Chromosome(*generate_ttp_solution(
        #     problem_data.Dimension,problem_data.ITEM,problem_data.CAPACITY
        # )) for _ in range(population_size)]
        self.best_chromosome = sorted(self.population, key=lambda c: c.phenome.net_profit)[-1]
        self.best_history = [self.best_chromosome.phenome.net_profit]
        self.avg_history = [sum(c.phenome.net_profit for c in self.population) / len(self.population)]
        self.time_execute = 0

    def init_population(self):
        pop_temp = []
        for _ in range(self.population_size):
            path , plan = generate_ttp_solution(self.problem_data.Dimension,self.problem_data.ITEM,self.problem_data.CAPACITY)
            time , profit = calculate_time_and_profit(path,plan,self.problem_data.NODE,self.problem_data.ITEM,self.problem_data.MIN_SPEED,self.problem_data.MAX_SPEED,self.problem_data.CAPACITY)
            net_profit = profit - (time*self.problem_data.RENTING_RATIO)
            phenome = Phenome(time,profit,net_profit)
            chorm = Chromosome(path,plan,phenome)
            pop_temp.append(chorm)
        return pop_temp

    def run(self):
        start = time.time()
        for _ in range(self.iterations):
            p1 = self.tournament_selection(self.population,self.tour_size)
            p2 = self.tournament_selection(self.population,self.tour_size)
            if np.random.rand() < self.cross_prob:
                p1, p2 = self.ordered_crossover(p1,p2)
            if np.random.rand() < self.mut_prob:
                p1 = self.inversion_mutation(p1)
                p2 = self.inversion_mutation(p2)
            self.population = self.replace_weakest(self.population, p1)
            self.population = self.replace_weakest(self.population, p2)

            current_best = sorted(self.population, key=lambda c: c.phenome.net_profit)[-1]
            if current_best.phenome.net_profit > self.best_chromosome.phenome.net_profit:
                self.best_chromosome = current_best
            
            self.best_history.append(self.best_chromosome.phenome.net_profit)
            self.avg_history.append(sum(c.phenome.net_profit for c in self.population) / len(self.population))
        end = time.time()
        self.time_execute = end- start
        return self.population , current_best
    
    
    def ordered_crossover(self,parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        # Select crossover points for the path
        start, end = sorted(np.random.choice(range(len(parent1.path)), 2))

        # Create segments from parents
        parent1_segment = parent1.path[start:end]
        parent2_segment = parent2.path[start:end]

        # Create offspring paths excluding parent segments
        offspring1_path = [city for city in parent2.path if city not in parent1_segment]
        offspring2_path = [city for city in parent1.path if city not in parent2_segment]

        # Insert parent segments into offspring paths
        offspring1_path[start:start] = parent1_segment
        offspring2_path[start:start] = parent2_segment

        # For the plan, using a simple one-point crossover
        crossover_point = np.random.randint(1, len(parent1.plan) - 1)
        offspring1_plan = parent1.plan[:crossover_point] + parent2.plan[crossover_point:]
        offspring2_plan = parent2.plan[:crossover_point] + parent1.plan[crossover_point:]
        
        offspring1_path =   (offspring1_path, self.problem_data.NODE, self.problem_data.MIN_SPEED, self.problem_data.MAX_SPEED, self.problem_data.CAPACITY)
        offspring2_path = apply_two_opt_local_search(offspring2_path, self.problem_data.NODE, self.problem_data.MIN_SPEED, self.problem_data.MAX_SPEED, self.problem_data.CAPACITY)

        time1 , profit1 = calculate_time_and_profit(offspring1_path,offspring1_plan,self.problem_data.NODE,self.problem_data.ITEM,self.problem_data.MIN_SPEED,self.problem_data.MAX_SPEED,self.problem_data.CAPACITY)
        net_profit1 = profit1 - (time1*self.problem_data.RENTING_RATIO)
        phenome1 = Phenome(time1,profit1,net_profit1)

        time2 , profit2 = calculate_time_and_profit(offspring2_path,offspring2_plan,self.problem_data.NODE,self.problem_data.ITEM,self.problem_data.MIN_SPEED,self.problem_data.MAX_SPEED,self.problem_data.CAPACITY)
        net_profit2 = profit2 - (time2*self.problem_data.RENTING_RATIO)
        phenome2 = Phenome(time2,profit2,net_profit2)

        
        offspring1 = Chromosome(offspring1_path, offspring1_plan, phenome1)
        offspring2 = Chromosome(offspring2_path, offspring2_plan, phenome2)       


        # # Create new Chromosome instances for offspring
        # offspring1 = Chromosome(offspring1_path, offspring1_plan)
        # offspring2 = Chromosome(offspring2_path, offspring2_plan)

        return offspring1, offspring2

    def inversion_mutation(self,chromosome: Chromosome):
        # Ensure there are at least two elements in the path
        if len(chromosome.path) < 2:
            return chromosome
        path = chromosome.path.copy()
        plan = chromosome.plan.copy()
        # Choose two distinct random positions in the path
        pos1, pos2 = sorted(np.random.choice(range(len(chromosome.path)), 2))

        # Invert the order of elements between pos1 and pos2
        path[pos1:pos2 + 1] = reversed(path[pos1:pos2 + 1])
        plan[pos1:pos2 + 1] = reversed(plan[pos1:pos2 + 1])
        mutated_path = apply_two_opt_local_search(path, self.problem_data.NODE, self.problem_data.MIN_SPEED, self.problem_data.MAX_SPEED, self.problem_data.CAPACITY)

        new_time , new_profit = calculate_time_and_profit(mutated_path,plan,self.problem_data.NODE,self.problem_data.ITEM,self.problem_data.MIN_SPEED,self.problem_data.MAX_SPEED,self.problem_data.CAPACITY)
        new_net_profit = new_profit - (new_time*self.problem_data.RENTING_RATIO)
        phenome = Phenome(new_time,new_profit,new_net_profit)

        return Chromosome(mutated_path,plan,phenome)

    def replace_weakest(self,population : List[Chromosome], candidates:Chromosome):
        keys = [x.phenome.net_profit for x in population]
        weakest_index = np.argmin(keys)

        if candidates.phenome.net_profit > population[weakest_index].phenome.net_profit:
            population[weakest_index] = candidates 

        return population
    def tournament_selection(self,population: List[int], tournament_size: int) -> Chromosome:
        """
        Selects a single Chromosome from the population using tournament selection.

        :param population: An instance of the Population class containing Chromosomes.
        :param tournament_size: The number of Chromosomes to be selected for each tournament.
        :return: The winning Chromosome with the highest net profit.
        """
        # Ensure the tournament size is not larger than the population size
        tournament_size = min(tournament_size, len(population))
        
        # Randomly select 'tournament_size' individuals from the population
        tournament_contestants = np.random.choice(population, size=tournament_size, replace=False)
        
        # Determine the winner based on the highest net profit
        winner = max(tournament_contestants, key=lambda chromo: chromo.phenome.net_profit)
        
        return winner


class GA3_develop:
    def __init__(self,problem_data : TTP, population_size, iterations, tour_size, mut_prob=1.0, cross_prob=1.0) -> None:
        self.problem_data = problem_data
        self.iterations = iterations
        self.population_size = population_size
        self.tour_size = tour_size
        self.mut_prob = mut_prob
        self.cross_prob = cross_prob
        self.population = self.init_population()
        # self.population = [Chromosome(*generate_ttp_solution(
        #     problem_data.Dimension,problem_data.ITEM,problem_data.CAPACITY
        # )) for _ in range(population_size)]
        self.best_chromosome = sorted(self.population, key=lambda c: c.phenome.net_profit)[-1]
        self.best_history = [self.best_chromosome.phenome.net_profit]
        self.avg_history = [sum(c.phenome.net_profit for c in self.population) / len(self.population)]
        self.time_execute = 0
        self.prev_best_fitness = None
        self.convergence_count = 0
        self.convergence_threshold = 10

    def init_population(self):
        pop_temp = []
        for _ in range(self.population_size):
            path , plan = generate_ttp_solution(self.problem_data.Dimension,self.problem_data.ITEM,self.problem_data.CAPACITY)
            time , profit = calculate_time_and_profit(path,plan,self.problem_data.NODE,self.problem_data.ITEM,self.problem_data.MIN_SPEED,self.problem_data.MAX_SPEED,self.problem_data.CAPACITY)
            net_profit = profit - (time*self.problem_data.RENTING_RATIO)
            phenome = Phenome(time,profit,net_profit)
            chorm = Chromosome(path,plan,phenome)
            pop_temp.append(chorm)
        return pop_temp

    def run(self):
        start = time.time()
        for _ in range(self.iterations):
            p1 = self.tournament_selection(self.population,self.tour_size)
            p2 = self.tournament_selection(self.population,self.tour_size)
            if np.random.rand() < self.cross_prob:
                p1, p2 = self.ordered_crossover(p1,p2)
            if np.random.rand() < self.mut_prob:
                p1 = self.inversion_mutation(p1)
                p2 = self.inversion_mutation(p2)
            self.population = self.replace_weakest(self.population, p1)
            self.population = self.replace_weakest(self.population, p2)

            current_best = sorted(self.population, key=lambda c: c.phenome.net_profit)[-1]
            if current_best.phenome.net_profit > self.best_chromosome.phenome.net_profit:
                self.best_chromosome = current_best
            
            self.best_history.append(self.best_chromosome.phenome.net_profit)
            self.avg_history.append(sum(c.phenome.net_profit for c in self.population) / len(self.population))
        end = time.time()
        self.time_execute = end- start
        return self.population , current_best

    def check_low_diversity(self):
        fitness_values = [chromo.phenome.net_profit for chromo in self.population]
        fitness_variance = np.var(fitness_values)
        
        # Adjust the threshold based on 10% of the variance
        diversity_threshold = 0.1 * fitness_variance

        return fitness_variance < diversity_threshold
    
    
    def ordered_crossover(self,parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        # Select crossover points for the path
        start, end = sorted(np.random.choice(range(len(parent1.path)), 2))

        # Create segments from parents
        parent1_segment = parent1.path[start:end]
        parent2_segment = parent2.path[start:end]

        # Create offspring paths excluding parent segments
        offspring1_path = [city for city in parent2.path if city not in parent1_segment]
        offspring2_path = [city for city in parent1.path if city not in parent2_segment]

        # Insert parent segments into offspring paths
        offspring1_path[start:start] = parent1_segment
        offspring2_path[start:start] = parent2_segment

        # For the plan, using a simple one-point crossover
        crossover_point = np.random.randint(1, len(parent1.plan) - 1)
        offspring1_plan = parent1.plan[:crossover_point] + parent2.plan[crossover_point:]
        offspring2_plan = parent2.plan[:crossover_point] + parent1.plan[crossover_point:]
        
        offspring1_path = apply_two_opt_local_search(offspring1_path, self.problem_data.NODE, self.problem_data.MIN_SPEED, self.problem_data.MAX_SPEED, self.problem_data.CAPACITY)
        offspring2_path = apply_two_opt_local_search(offspring2_path, self.problem_data.NODE, self.problem_data.MIN_SPEED, self.problem_data.MAX_SPEED, self.problem_data.CAPACITY)

        time1 , profit1 = calculate_time_and_profit(offspring1_path,offspring1_plan,self.problem_data.NODE,self.problem_data.ITEM,self.problem_data.MIN_SPEED,self.problem_data.MAX_SPEED,self.problem_data.CAPACITY)
        net_profit1 = profit1 - (time1*self.problem_data.RENTING_RATIO)
        phenome1 = Phenome(time1,profit1,net_profit1)

        time2 , profit2 = calculate_time_and_profit(offspring2_path,offspring2_plan,self.problem_data.NODE,self.problem_data.ITEM,self.problem_data.MIN_SPEED,self.problem_data.MAX_SPEED,self.problem_data.CAPACITY)
        net_profit2 = profit2 - (time2*self.problem_data.RENTING_RATIO)
        phenome2 = Phenome(time2,profit2,net_profit2)

        
        offspring1 = Chromosome(offspring1_path, offspring1_plan, phenome1)
        offspring2 = Chromosome(offspring2_path, offspring2_plan, phenome2)       


        # # Create new Chromosome instances for offspring
        # offspring1 = Chromosome(offspring1_path, offspring1_plan)
        # offspring2 = Chromosome(offspring2_path, offspring2_plan)

        return offspring1, offspring2

    def inversion_mutation(self,chromosome: Chromosome):
        # Ensure there are at least two elements in the path
        if len(chromosome.path) < 2:
            return chromosome
        path = chromosome.path.copy()
        plan = chromosome.plan.copy()
        # Choose two distinct random positions in the path
        pos1, pos2 = sorted(np.random.choice(range(len(chromosome.path)), 2))

        # Invert the order of elements between pos1 and pos2
        path[pos1:pos2 + 1] = reversed(path[pos1:pos2 + 1])
        plan[pos1:pos2 + 1] = reversed(plan[pos1:pos2 + 1])
        mutated_path = apply_two_opt_local_search(path, self.problem_data.NODE, self.problem_data.MIN_SPEED, self.problem_data.MAX_SPEED, self.problem_data.CAPACITY)

        new_time , new_profit = calculate_time_and_profit(mutated_path,plan,self.problem_data.NODE,self.problem_data.ITEM,self.problem_data.MIN_SPEED,self.problem_data.MAX_SPEED,self.problem_data.CAPACITY)
        new_net_profit = new_profit - (new_time*self.problem_data.RENTING_RATIO)
        phenome = Phenome(new_time,new_profit,new_net_profit)

        return Chromosome(mutated_path,plan,phenome)

    def replace_weakest(self,population : List[Chromosome], candidates:Chromosome):
        keys = [x.phenome.net_profit for x in population]
        weakest_index = np.argmin(keys)

        if candidates.phenome.net_profit > population[weakest_index].phenome.net_profit:
            population[weakest_index] = candidates 

        return population
    def tournament_selection(self,population: List[int], tournament_size: int) -> Chromosome:
        """
        Selects a single Chromosome from the population using tournament selection.

        :param population: An instance of the Population class containing Chromosomes.
        :param tournament_size: The number of Chromosomes to be selected for each tournament.
        :return: The winning Chromosome with the highest net profit.
        """
        # Ensure the tournament size is not larger than the population size
        tournament_size = min(tournament_size, len(population))
        
        # Randomly select 'tournament_size' individuals from the population
        tournament_contestants = np.random.choice(population, size=tournament_size, replace=False)
        
        # Determine the winner based on the highest net profit
        winner = max(tournament_contestants, key=lambda chromo: chromo.phenome.net_profit)
        
        return winner

    def check_low_diversity(self):
        fitness_values = [chromo.phenome.net_profit for chromo in self.population]
        fitness_variance = np.var(fitness_values)
        return fitness_variance < diversity_threshold

    def check_convergence(self):
        current_best_fitness = max(self.population, key=lambda chromo: chromo.phenome.net_profit).phenome.net_profit
        if self.prev_best_fitness is not None:
            if abs(current_best_fitness - self.prev_best_fitness) < fitness_improvement_threshold:
                self.convergence_count += 1
            else:
                self.convergence_count = 0  # Reset if there's significant improvement

# @dataclass
# class City:
#     index : int
#     X : float
#     Y : float

# @dataclass
# class Item:
#     index : int
#     Profit : int
#     Weight : int
#     Node : int
    
# @dataclass
# class TTP:
#     Name :str = None
#     DTYPE : str = None
#     Dimension : int = 0
#     ITEMS : int = 0
#     CAPACITY : int = 0
#     MIN_SPEED : float = 0
#     MAX_SPEED : float = 0
#     RENTING_RATIO : float = 0
#     EDGE_W : str = None
#     NODE : List[City] = None
#     ITEM : List[Item] = None

# def read_problem(file_path:str):
#     with open(file_path,'r') as file:
#         lines = file.readlines()
    
#     data = TTP(NODE=[],ITEM=[])
    
#     for i , line in enumerate(lines):
#         if line.startswith("PROBLEM NAME"):
#             data.Name = line.split(':')[-1].strip()
#         elif line.startswith("KNAPSACK DATA TYPE"):
#             data.DTYPE = line.split(':')[-1].strip()
#         elif line.startswith("DIMENSION"):
#             data.Dimension = int(line.split(':')[-1].strip())
#         elif line.startswith("NUMBER OF ITEMS"):
#             data.ITEMS = int(line.split(':')[-1].strip())
#         elif line.startswith("CAPACITY OF KNAPSACK"):
#             data.CAPACITY = int(line.split(':')[-1].strip())
#         elif line.startswith("MIN SPEED"):
#             data.MIN_SPEED = float(line.split(':')[-1].strip())
#         elif line.startswith("MAX SPEED"):
#             data.MAX_SPEED = float(line.split(':')[-1].strip())
#         elif line.startswith("RENTING RATIO"):
#             data.RENTING_RATIO = float(line.split(':')[-1].strip())
#         elif line.startswith("EDGE_WEIGHT_TYPE"):
#             data.EDGE_W = line.split(':')[-1].strip()
#         elif line.startswith("NODE_COORD_SECTION"):
#             for j in range(1,data.Dimension+1):
#                 node = lines[i+j].split()
#                 data.NODE.append(City(index=int(node[0]),X=float(node[1]),Y=float(node[2])))
#         elif line.startswith("ITEMS SECTION"):
#             for j in range(1,data.ITEMS+1):
#                 item = lines[i+j].split()
#                 data.ITEM.append(
#                     Item(int(item[0]),int(item[1]),int(item[2]),int(item[3]))
#                 )
#         else:
#             pass
    
#     return data

# def generate_ttp_solution(number_of_cities: int, items: List[Item], knapsack_capacity: int) -> Tuple[List[int], List[int]]:
#     # Generate a random path (tour)
#     path = np.random.permutation(number_of_cities) + 1

#     # Initialize knapsack plan with no items picked
#     plan = [0] * len(items)
#     current_weight = 0

#     # Randomly decide to pick up items considering the knapsack capacity
#     for i, item in enumerate(items):
#         item_weight = item.Weight
#         if current_weight + item_weight <= knapsack_capacity:
#             decision = np.random.choice([0, 1])
#             plan[i] = decision
#             current_weight += item_weight * decision

#     return path.tolist(), plan

# def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
#     return math.ceil(np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2))

# def calculate_time_and_profit(solution: List[int], plan: List[int], nodes: List[City], items: List[Item], min_speed, max_speed, max_weight):
#     total_time = 0
#     total_profit = 0
#     current_weight = 0

#     # Calculate the total travel time
#     for i in range(len(solution)):
#         current_city_index = solution[i]
#         next_city_index = solution[0] if i == len(solution) - 1 else solution[i + 1]

#         current_city = nodes[current_city_index - 1]
#         next_city = nodes[next_city_index - 1]

        

#         # Update current weight based on items picked at the current city
#         for item, is_picked in zip(items, plan):
#             if is_picked and item.Node == current_city_index:
#                 current_weight += item.Weight

#         # Calculate speed based on current weight
#         speed = max_speed - (current_weight / max_weight) * (max_speed - min_speed)
#         speed = max(speed, min_speed)  # Ensure speed doesn't drop below minimum

#         # Distance between current city and next city
#         distance = euclidean_distance((current_city.X, current_city.Y), (next_city.X, next_city.Y))

#         # Update time with time to next city
#         total_time += distance / speed

#         if current_weight > max_weight:
#             return np.Inf , 0.0

#     # Calculate total profit from picked items
#     for item, is_picked in zip(items, plan):
#         if is_picked:
#             total_profit += item.Profit


#     return total_time, total_profit

# @dataclass
# class Phenome:
#     time : float
#     profit : float
#     net_profit : float = 0

#     def __post_init__(self):
#         self.net_profit = self.profit - (self.time*renting_ratio)

# @dataclass
# class Chromosome:
#     path : List[int]
#     plan : List[int]
#     phenome : Phenome = None

#     def __post_init__(self):
#         self.phenome = Phenome(
#             *calculate_time_and_profit(
#                 self.path,self.plan,node,item,min_speed,max_speed,max_weight))

# def tournament_selection(population: List[int], tournament_size: int) -> Chromosome:
#     """
#     Selects a single Chromosome from the population using tournament selection.

#     :param population: An instance of the Population class containing Chromosomes.
#     :param tournament_size: The number of Chromosomes to be selected for each tournament.
#     :return: The winning Chromosome with the highest net profit.
#     """
#     # Ensure the tournament size is not larger than the population size
#     tournament_size = min(tournament_size, len(population))
    
#     # Randomly select 'tournament_size' individuals from the population
#     tournament_contestants = np.random.choice(population, size=tournament_size, replace=False)
    
#     # Determine the winner based on the highest net profit
#     winner = max(tournament_contestants, key=lambda chromo: chromo.phenome.net_profit)
    
#     return winner

# def ordered_crossover(parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
#     # Select crossover points for the path
#     start, end = sorted(np.random.choice(range(len(parent1.path)), 2))

#     # Create segments from parents
#     parent1_segment = parent1.path[start:end]
#     parent2_segment = parent2.path[start:end]

#     # Create offspring paths excluding parent segments
#     offspring1_path = [city for city in parent2.path if city not in parent1_segment]
#     offspring2_path = [city for city in parent1.path if city not in parent2_segment]

#     # Insert parent segments into offspring paths
#     offspring1_path[start:start] = parent1_segment
#     offspring2_path[start:start] = parent2_segment

#     # For the plan, using a simple one-point crossover
#     crossover_point = np.random.randint(1, len(parent1.plan) - 1)
#     offspring1_plan = parent1.plan[:crossover_point] + parent2.plan[crossover_point:]
#     offspring2_plan = parent2.plan[:crossover_point] + parent1.plan[crossover_point:]

#     # Create new Chromosome instances for offspring
#     offspring1 = Chromosome(offspring1_path, offspring1_plan)
#     offspring2 = Chromosome(offspring2_path, offspring2_plan)

#     return offspring1, offspring2

# def inversion_mutation(chromosome: Chromosome):
#     # Ensure there are at least two elements in the path
#     if len(chromosome.path) < 2:
#         return chromosome
#     path = chromosome.path.copy()
#     plan = chromosome.plan.copy()
#     # Choose two distinct random positions in the path
#     pos1, pos2 = sorted(np.random.choice(range(len(chromosome.path)), 2))

#     # Invert the order of elements between pos1 and pos2
#     path[pos1:pos2 + 1] = reversed(path[pos1:pos2 + 1])
#     plan[pos1:pos2 + 1] = reversed(plan[pos1:pos2 + 1])

#     return Chromosome(path,plan)

# def replace_weakest(population : List[Chromosome], candidates:Chromosome):
#     keys = [x.phenome.net_profit for x in population]
#     weakest_index = np.argmin(keys)

#     if candidates.phenome.net_profit > population[weakest_index].phenome.net_profit:
#         population[weakest_index] = candidates 

#     return population

# # @dataclass
# class Phenome:
#     time : float
#     profit : float
#     net_profit : float

# @dataclass
# class Chromosome:
#     path : List[int]
#     plan : List[int]
#     phenome : Phenome


# class GA:
#     def __init__(self,problem_data : TTP, population_size, iterations, tour_size, mut_prob=1.0, cross_prob=1.0) -> None:
#         self.problem_data = problem_data
#         self.iterations = iterations
#         self.population_size = population_size
#         self.tour_size = tour_size
#         self.mut_prob = mut_prob
#         self.cross_prob = cross_prob
#         self.population = self.init_population()
#         # self.population = [Chromosome(*generate_ttp_solution(
#         #     problem_data.Dimension,problem_data.ITEM,problem_data.CAPACITY
#         # )) for _ in range(population_size)]
#         self.best_chromosome = sorted(self.population, key=lambda c: c.phenome.net_profit)[-1]
#         self.best_history = [self.best_chromosome.phenome.net_profit]
#         self.avg_history = [sum(c.phenome.net_profit for c in self.population) / len(self.population)]
#         self.time_execute = 0

#     def init_population(self):
#         pop_temp = []
#         for _ in range(self.pop_size):
#             path , plan = generate_ttp_solution(self.problem_data.Dimension,self.problem_data.ITEM,self.problem_data.CAPACITY)
#             time , profit = calculate_time_and_profit(path,plan,self.problem_data.NODE,self.problem_data.ITEM,self.problem_data.MIN_SPEED,self.problem_data.MAX_SPEED,self.problem_data.CAPACITY)
#             net_profit = profit - (time*self.problem_data.renting_ratio)
#             phenome = Phenome(time,profit,net_profit)
#             chorm = Chromosome(path,plan,phenome)
#             pop_temp.append(chorm)
#         return pop_temp

#     def run(self):
#         start = time.time()
#         for _ in range(self.iterations):
#             p1 = tournament_selection(self.population,self.tour_size)
#             p2 = tournament_selection(self.population,self.tour_size)
#             if np.random.rand() < self.cross_prob:
#                 p1, p2 = ordered_crossover(p1,p2)
#             if np.random.rand() < self.mut_prob:
#                 p1 = inversion_mutation(p1)
#                 p2 = inversion_mutation(p2)
#             self.population = replace_weakest(self.population, p1)
#             self.population = replace_weakest(self.population, p2)

#             current_best = sorted(self.population, key=lambda c: c.phenome.net_profit)[-1]
#             if current_best.phenome.net_profit > self.best_chromosome.phenome.net_profit:
#                 self.best_chromosome = current_best
            
#             self.best_history.append(self.best_chromosome.phenome.net_profit)
#             self.avg_history.append(sum(c.phenome.net_profit for c in self.population) / len(self.population))
#         end = time.time()
#         self.time_execute = end- start
#         return self.population , current_best
    
#     def ordered_crossover(self,parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
#         # Select crossover points for the path
#         start, end = sorted(np.random.choice(range(len(parent1.path)), 2))

#         # Create segments from parents
#         parent1_segment = parent1.path[start:end]
#         parent2_segment = parent2.path[start:end]

#         # Create offspring paths excluding parent segments
#         offspring1_path = [city for city in parent2.path if city not in parent1_segment]
#         offspring2_path = [city for city in parent1.path if city not in parent2_segment]

#         # Insert parent segments into offspring paths
#         offspring1_path[start:start] = parent1_segment
#         offspring2_path[start:start] = parent2_segment

#         # For the plan, using a simple one-point crossover
#         crossover_point = np.random.randint(1, len(parent1.plan) - 1)
#         offspring1_plan = parent1.plan[:crossover_point] + parent2.plan[crossover_point:]
#         offspring2_plan = parent2.plan[:crossover_point] + parent1.plan[crossover_point:]

#         time1 , profit1 = calculate_time_and_profit(offspring1_path,offspring1_plan,self.problem_data.NODE,self.problem_data.ITEM,self.problem_data.MIN_SPEED,self.problem_data.MAX_SPEED,self.problem_data.CAPACITY)
#         net_profit1 = profit1 - (time1*self.problem_data.renting_ratio)
#         phenome1 = Phenome(time1,profit1,net_profit1)

#         time2 , profit2 = calculate_time_and_profit(offspring2_path,offspring2_plan,self.problem_data.NODE,self.problem_data.ITEM,self.problem_data.MIN_SPEED,self.problem_data.MAX_SPEED,self.problem_data.CAPACITY)
#         net_profit2 = profit2 - (time2*self.problem_data.renting_ratio)
#         phenome2 = Phenome(time2,profit2,net_profit2)

#         offspring1 = Chromosome(offspring1_path, offspring1_plan, phenome1)
#         offspring2 = Chromosome(offspring2_path, offspring2_plan, phenome2)       

#         # # Create new Chromosome instances for offspring
#         # offspring1 = Chromosome(offspring1_path, offspring1_plan)
#         # offspring2 = Chromosome(offspring2_path, offspring2_plan)

#         return offspring1, offspring2

#     def inversion_mutation(self,chromosome: Chromosome):
#         # Ensure there are at least two elements in the path
#         if len(chromosome.path) < 2:
#             return chromosome
#         path = chromosome.path.copy()
#         plan = chromosome.plan.copy()
#         # Choose two distinct random positions in the path
#         pos1, pos2 = sorted(np.random.choice(range(len(chromosome.path)), 2))

#         # Invert the order of elements between pos1 and pos2
#         path[pos1:pos2 + 1] = reversed(path[pos1:pos2 + 1])
#         plan[pos1:pos2 + 1] = reversed(plan[pos1:pos2 + 1])

#         new_time , new_profit = calculate_time_and_profit(path,plan,self.problem_data.NODE,self.problem_data.ITEM,self.problem_data.MIN_SPEED,self.problem_data.MAX_SPEED,self.problem_data.CAPACITY)
#         new_net_profit = new_profit - (new_time*self.problem_data.renting_ratio)
#         phenome = Phenome(new_time,new_profit,new_net_profit)

#         return Chromosome(path,plan,phenome)

#     def replace_weakest(self,population : List[Chromosome], candidates:Chromosome):
#         keys = [x.phenome.net_profit for x in population]
#         weakest_index = np.argmin(keys)

#         if candidates.phenome.net_profit > population[weakest_index].phenome.net_profit:
#             population[weakest_index] = candidates 

#         return population


# # def initialize_and_run_ga(path, population_size, iterations, tour_size, mut_prob=1.0, cross_prob=1.0):
# #     # Read problem data from the given path
# #     data = read_problem(file_path=path)
# #     max_weight = data.CAPACITY
# #     max_speed = data.MAX_SPEED
# #     min_speed = data.MIN_SPEED
# #     dimension = data.Dimension
# #     nitem = data.ITEMS
# #     node = data.NODE
# #     item = data.ITEM
# #     renting_ratio = data.RENTING_RATIO


# #     # Create an instance of the GA class
# #     ga = GA(data, population_size, iterations, tour_size, mut_prob, cross_prob)

# #     # Run the GA
# #     population, best_solution = ga.run()

# #     return ga, population, best_solution