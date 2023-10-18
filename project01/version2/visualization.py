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

def draw_fitnesses(upper1,avg1,upper2,avg2):
    fig, ax = plt.subplots(2,1,sharex=True,figsize=(8,6))
    x = list(range(len(upper1)))
    ax[0].plot(x, upper1, linestyle='--', color='red', label='max fitness')
    ax[0].plot(x, avg1, linestyle='--',color='g',linewidth=2.0, label='mean fitness')
    ax[0].set_title("Burma")
    ax[0].grid()
    ax[0].legend()
    ax[0].text(x[0]*0.95,avg1[0]*1.01,f"{avg1[0]:.5e}",fontsize=8)
    ax[0].text(x[0]*0.95,upper1[0]*1.01,f"{upper1[0]:.5e}",fontsize=8)
    ax[0].text(x[-1]*0.95,avg1[-1]*1.01,f"{avg1[-1]:.5e}",fontsize=8)
    ax[0].text(x[-1]*0.95,upper1[-1]*1.01,f"{upper1[-1]:.5e}",fontsize=8)

    ax[1].plot(x, upper2, linestyle='--', color='red', label='max fitness')
    ax[1].plot(x, avg2, linestyle='--',color='g',linewidth=2.0, label='mean fitness')
    ax[1].grid()
    ax[1].set_title("Brazil")
    ax[1].text(x[0]*0.95,avg2[0]*1.01,f"{avg2[0]:.5e}",fontsize=8)
    ax[1].text(x[0]*0.95,upper2[0]*1.01,f"{upper2[0]:.5e}",fontsize=8)
    ax[1].text(x[-1]*0.95,avg2[-1]*1.01,f"{avg2[-1]:.5e}",fontsize=8)
    ax[1].text(x[-1]*0.95,upper2[-1]*1.01,f"{upper2[-1]:.5e}",fontsize=8)


    plt.xlabel("generations")
    plt.ylabel('fitness value')
    plt.tight_layout()
    fig.suptitle("Experiment",y=1.05)
    plt.show()