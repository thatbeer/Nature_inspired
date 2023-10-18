import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from typing import List


def draw_fitness_seperate(upper:List[float], avg:List[float]):
    fig, ax = plt.subplots(3, 1, figsize=(6, 8),sharex=True)
    x = list(range(len(upper)))
    fig.tight_layout()
    ax[0].plot(x, upper, color='green', linewidth=3.0)
    ax[2].plot(x, avg, color='red', linewidth=3.0, linestyle='-')
    for a in ax:
        a.grid()
    plt.show()

def draw_fitness(upper:List[float], avg:List[float]):
    x = list(range(len(upper)))
    # plt.figure(8,8)
    plt.plot(x, avg, linestyle='--', color='red')
    plt.plot(x, upper, linestyle='--',color='g',linewidth=2.0)
    plt.title("Experiment1")
    plt.xlabel('generations')
    plt.ylabel('fitness value')
    plt.grid()
    plt.show()

def draw_cost_seperate(upper:List[float], avg:List[float]):
    fig, ax = plt.subplots(3, 1, figsize=(6, 8),sharex=True)
    upper = [1/x for x in upper]
    avg = [1/x for x in avg]
    x = list(range(len(upper)))
    fig.tight_layout()
    ax[0].plot(x, upper, color='green', linewidth=3.0)
    ax[2].plot(x, avg, color='red', linewidth=3.0, linestyle='-')
    for a in ax:
        a.grid()
    plt.show()

def draw_cost(upper:List[float], avg:List[float]):
    x = list(range(len(upper)))
    upper = [1/x for x in upper]
    avg = [1/x for x in avg]
    # plt.figure(8,8)
    plt.plot(x, avg, linestyle='--', color='red')
    plt.plot(x, upper, linestyle='--',color='g',linewidth=2.0)
    plt.title("Experiment1")
    plt.xlabel('generations')
    plt.ylabel('fitness value')
    plt.grid()
    plt.show()