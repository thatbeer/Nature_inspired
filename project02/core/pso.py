import numpy as np
import pandas as pd
import numpy as np
import math


from typing import List, Tuple
from dataclasses import dataclass

pbest_prob = 0.7
gbest_prob = 0.9


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
class Fitness:
    time: float
    profit: float
    net_profit: float

@dataclass
class Particle:
    data: TTP
    path: List[int]
    plan: List[int]
    current_score: Fitness = None
    velocity: List = None
    personal_best_path: List[int] = None
    personal_best_plan: List[int] = None
    personal_best_score: Fitness = None

    def __post_init__(self):
        self.velocity = self.velocity if self.velocity is not None else []
        self.current_score = self.calculate_score()
        self.personal_best_path = self.path.copy()
        self.personal_best_plan = self.plan.copy()
        self.personal_best_score = self.current_score

    def calculate_score(self):
        time, profit = calculate_time_and_profit(self.path, self.plan, self.data.NODE, self.data.ITEM, self.data.MIN_SPEED, self.data.MAX_SPEED, self.data.CAPACITY)
        net_profit = profit - (self.data.RENTING_RATIO * time)
        return Fitness(time, profit, net_profit)

    def update_personal_best(self):
        self.current_score = self.calculate_score()
        if self.current_score.net_profit > self.personal_best_score.net_profit:
            self.personal_best_path = self.path.copy()
            self.personal_best_plan = self.plan.copy()
            self.personal_best_score = self.current_score

    def update_velocity(self, global_best_particle):
        # Simple velocity update logic (can be enhanced for better performance)
        for i in range(len(self.path)):
            if np.random.rand() < pbest_prob and self.path[i] != self.personal_best_path[i]:
                self.velocity.append(('path', i, self.personal_best_path.index(self.path[i])))
            if np.random.rand() < gbest_prob and self.path[i] != global_best_particle.path[i]:
                self.velocity.append(('path', i, global_best_particle.path.index(self.path[i])))
        
        for i in range(len(self.plan)):
            if np.random.rand() < pbest_prob and self.plan[i] != self.personal_best_plan[i]:
                self.velocity.append(('plan', i, self.personal_best_plan.index(self.plan[i])))
            if np.random.rand() < gbest_prob and self.plan[i] != global_best_particle.plan[i]:
                self.velocity.append(('plan', i, global_best_particle.plan.index(self.plan[i])))

    def apply_velocity(self):
        for change in self.velocity:
            if change[0] == 'path':
                self.path[change[1]], self.path[change[2]] = self.path[change[2]], self.path[change[1]]
            elif change[0] == 'plan':
                self.plan[change[1]], self.plan[change[2]] = self.plan[change[2]], self.plan[change[1]]

    def clear_velocity(self):
        self.velocity.clear()

class PSO:
    def __init__(self, data, num_particles, iterations, gbest_prob=1.0, pbest_prob=1.0):
        self.data = data
        self.num_particles = num_particles
        self.iterations = iterations
        self.gbest_prob = gbest_prob
        self.pbest_prob = pbest_prob
        self.particles = [Particle(data, *generate_ttp_solution(data.Dimension, data.ITEM, data.CAPACITY)) for _ in range(num_particles)]
        self.gbest_particle = max(self.particles, key=lambda p: p.personal_best_score.net_profit)

    def run(self):
        for _ in range(self.iterations):
            for particle in self.particles:
                particle.clear_velocity()
                particle.update_velocity(self.gbest_particle)
                particle.apply_velocity()
                particle.update_personal_best()

                if particle.personal_best_score.net_profit > self.gbest_particle.personal_best_score.net_profit:
                    self.gbest_particle = particle

        return self.gbest_particle.personal_best_path, self.gbest_particle.personal_best_plan, self.gbest_particle.personal_best_score
