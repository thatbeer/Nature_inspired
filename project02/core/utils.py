import pandas as pd
import numpy as np
import random

from typing import Any, List, Tuple, Dict


from .dataclass import City, Item, TTP

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
        elif line.startswith("DIMENSION"):
            data.Dimension = int(line.split(':')[-1].strip())
        elif line.startswith("MIX SPEED"):
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
    
    return data.NODE , data.ITEM

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
    return np.math.ceil(np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2))

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