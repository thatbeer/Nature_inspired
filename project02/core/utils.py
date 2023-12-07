import pandas as pd
import numpy as np
import random

from dataclasses import dataclass
from typing import Any, List, Tuple, Dict
from pandas import DataFrame

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
    
    # items = [(x.index,x.Profit,x.Weight,x.Node) for x in data.ITEM]
    # nodes = [(x.index,x.X,x.Y) for x in data.NODE]
    nodes = pd.DataFrame({
        'X' : [x.X for x in data.NODE],
        'Y' : [x.Y for x in data.NODE],
    })
    items = pd.DataFrame({
        'Profit' : [x.Profit for x in data.ITEM],
        'Weight' : [x.Weight for x in data.ITEM],
        'Node' : [x.Node for x in data.ITEM]
    })
    return nodes , items
    # return data.NODE , data.ITEM


def generate_ttp_solution(number_of_cities, items_df, knapsack_capacity):
    # Generate a random path (tour)
    path = np.random.permutation(number_of_cities)

    # Initialize knapsack plan with no items picked
    plan = [0] * len(items_df)
    current_weight = 0

    # Randomly decide to pick up items considering the knapsack capacity
    for i, row in items_df.iterrows():
        item_weight = row['Weight']
        if current_weight + item_weight <= knapsack_capacity:
            decision = random.choice([0, 1])
            plan[i] = decision
            current_weight += item_weight * decision

    return path, plan


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

#assume there is only one item in the cities.
def calculate_time_and_profit(solution: List[int], plan: List[int], nodes_df, items_df, min_speed, max_speed, max_weight):
    total_time = 0
    total_profit = 0
    current_weight = 0

    # Calculate total profit from picked items
    for i, is_picked in enumerate(plan):
        if is_picked:
            total_profit += items_df.loc[i, 'Profit']

    # Calculate the total travel time
    for i in range(len(solution)):
        current_city = solution[i] - 1
        next_city = solution[0] - 1 if i == len(solution) - 1 else solution[i + 1] - 1

        # Update current weight based on items picked at the current city
        for j, is_picked in enumerate(plan):
            if is_picked and items_df.loc[j, 'Node'] == solution[i]:
                current_weight += items_df.loc[j, 'Weight']

        # Calculate speed based on current weight
        speed = max_speed - (current_weight / max_weight) * (max_speed - min_speed)
        speed = max(speed, min_speed)  # Ensure speed doesn't drop below minimum

        # Distance between current city and next city
        distance = euclidean_distance(
            (nodes_df.loc[current_city, 'X'], nodes_df.loc[current_city, 'Y']),
            (nodes_df.loc[next_city, 'X'], nodes_df.loc[next_city, 'Y'])
        )

        # Update time with time to next city
        total_time += distance / speed
 
    return total_time, total_profit

# handle multiple item in cities
def calculate_time_and_profit(solution: List[int], plan: List[int], nodes_df, items_df, min_speed, max_speed, max_weight):
    total_time = 0
    total_profit = 0
    current_weight = 0

    # Calculate total profit from picked items
    for i, is_picked in enumerate(plan):
        if is_picked:
            total_profit += items_df.loc[i, 'Profit']

    # Calculate the total travel time
    for i in range(len(solution)):
        current_city_index = solution[i] - 1  # Adjust if solution uses 1-based indexing
        next_city_index = solution[0] - 1 if i == len(solution) - 1 else solution[i + 1] - 1

        # Ensure indices are within the valid range
        current_city_index = max(0, min(current_city_index, len(nodes_df) - 1))
        next_city_index = max(0, min(next_city_index, len(nodes_df) - 1))

        # Update current weight based on items picked at the current city
        for j, row in items_df.iterrows():
            if plan[j] == 1 and row['Node'] == solution[i]:
                current_weight += row['Weight']

        # Calculate speed based on current weight
        speed = max_speed - (current_weight / max_weight) * (max_speed - min_speed)
        speed = max(speed, min_speed)  # Ensure speed doesn't drop below minimum

        # Distance between current city and next city
        distance = euclidean_distance(
            (nodes_df.loc[current_city_index, 'X'], nodes_df.loc[current_city_index, 'Y']),
            (nodes_df.loc[next_city_index, 'X'], nodes_df.loc[next_city_index, 'Y'])
        )

        # Update time with time to next city
        total_time += distance / speed

    return total_time, total_profit