import torch.nn as nn
import json
import datetime
import time
import pandas as pd
from dataclasses import asdict


from core import mutate, multi_mutate, inversion, scramble, no_crossover, ordered_crossover, point_crossover, partialmap_crossover, replace_firstweak, replace_weakest
from core.utils import load_metrics
from core.selection import create_chromosome, parents_selection, generate_population, best_gene, get_mean
from core.dataclass import *

def load_fn(fn_name):
    # Use a dictionary to map function names to their corresponding functions
    functions = {
        "mutate": mutate,
        "multi_mutate": multi_mutate,
        "inversion": inversion,
        "scramble": scramble,
        "no_crossover": no_crossover,
        "point_crossover": point_crossover,
        "ordered_crossover": ordered_crossover,
        "partialmap_crossover": partialmap_crossover,
        "replace_firstweak": replace_firstweak,
        "replace_weakest": replace_weakest
    }
    
    # Check if the provided function name is in the dictionary
    if fn_name in functions:
        return functions[fn_name]
    else:
        raise FileNotFoundError("Are you sure that you named a correct function?")

class GeneticAlgorithm(nn.Module):
    def __init__(self, config) -> None:
        super(GeneticAlgorithm, self).__init__()
        self.distance_metric = load_metrics(config.path)
        self.num_city = self.distance_metric.shape[0]
        self.num_trial = config.num_trial
        self.params = config.parameters
        self.mutate = load_fn(config.functions.mutate_fn)
        self.crossover = load_fn(config.functions.crossover_fn)
        self.replacement = load_fn(config.functions.replace_fn)
        self.population = self._generate_population()
        self.experiment_log = config.name
    
    def _generate_population(self):
        return generate_population(
            population_size=self.params.population_size,
            distance_metric=self.distance_metric
        )

    def _create_chromosome(self, gene):
        return create_chromosome(gene=gene,distance_metric=self.distance_metric)
    
    def _mutate(self, gene):
        return self.mutate(gene, self.params.p_mutate)
    
    def _crossover(self, parent1, parent2):
        return self.crossover(parent1, parent2, self.params.p_crossover)

    def _replacement(self, population, candidate):
        return self.replacement(population, candidate)
    
    def forward(self):
        # population = generate_population(
        #     population_size=self.params.population_size,
        #     distance_metric=self.distance_metric
        # )
        population = self.population
        best_one = best_gene(population=population)
        for _ in range(self.params.max_generations):
            population = self.evolve(population_origin=population)
            new_best = best_gene(population)
            if new_best.phenome > best_one.phenome:
                best_one = new_best

        return population, best_one

    def evolve(self, population_origin):
        populationx = population_origin.copy()
        parents = parents_selection(population=populationx,
                                    tournament_size=self.params.tournament_size)
        gene1, gene2 = self._crossover(parents[0], parents[1])
        child1, child2 = self._create_chromosome(gene1), self._create_chromosome(gene2)
        child1, child2 = self._mutate(child1), self._mutate(child2)
        for child in [child1,child2]:
            populationx = self._replacement(population=populationx,
                                                  candidate=child)
        return populationx
    

class SoleExp:
    def __init__(self, config : Config) -> None:
        self.config = config
        self.experiment = {
            "gens" : [],
            "best_fitness" : [],
            "avg_fitness" : [],
            "best_gene" : [],
        }
        self.name = config.name
        self.clear = self._clear_log()

    def run(self):
        ga = GeneticAlgorithm(self.config)
        population = ga._generate_population()
        best = best_gene(population)
        for g in range(self.config.parameters.max_generations):
            # log = AdvanceLog(
            #     gens=[],
            #     best_fitness=[],
            #     avg_fitness=[],
            #     best_gene=best
            # )
            population = ga.evolve(
                population_origin=population
            )
            new_best = best_gene(population=population)
            if new_best.phenome > best.phenome:
                best = new_best
            log = AdvanceLog(
                gens=g+1,
                best_fitness=new_best.phenome,
                avg_fitness=get_mean(population),
                best_gene=best.gene
            )
            # log.gens.append(g+1)
            # log.best_fitness.append(new_best.phenome)
            # log.avg_fitness.append(get_mean(population))
            # log.best_gene = best
            self.experiment.append(log)
    
    def save_csv(self):
        di = {
            "gens" : [x.gens for x in self.experiment],
            "best_fitness" : [x.best_fitness for x in self.experiment],
            "avg_fitness" : [x.avg_fitness for x in self.experiment],
            "best_gene" : [x.best_gene for x in self.experiment]
        }
        parents , _ = self.config.file_name.split('_')
        path_file = f'dest/{parents}/{self.config.file_name}.csv'
        df = pd.DataFrame(di)
        df.to_csv(path_file, index=False)
        return df
        
    def _clear_log(self):
        self.experiment = []
    
class Speculator:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.experiment : List[TriaLog] = []
        self.experiment_log = config.name
        self.clear = self._clear_log()
        self.random_seed = self._randomseed()
        self.seeds = []
        self.time = []

    def _randomseed(self):
        return np.random.randint(1,1000)
    
    def _clear_log(self) -> None:
        self.experiment = []
    
    def run(self):
        """Run GA experiments and log data."""
        for trial in range(self.config.num_trial):
            # clear the log at the beginning of each trial
            start_time = time.time()
            # seed = self.random_seed
            # self.seeds.append(seed)
            # np.random.seed(seed)
            log = SimpleLog(
                None,
                0,
                0
            )
            ga = GeneticAlgorithm(self.config)
            last_pop, best_gene = ga.forward()
            log.best_gene = best_gene.gene
            log.best_fitness = best_gene.phenome
            log.avg_fitness = get_mean(last_pop)
            end_time = time.time()
            self.time.append(end_time-start_time)
            self.experiment.append(TriaLog(trial+1, log))
    
    def save_csv(self):
        data = {
            "trial" : [x.trial for x in self.experiment],
            "best_gene" : [x.information.best_gene for x in self.experiment],
            "best_fitness": [x.information.best_fitness for x in self.experiment],
            "avg_fitness" : [x.information.avg_fitness for x in self.experiment],
            # "avg_bestFitness" :[ x.avg_fitness for x in self.experiment],
            # "seeds" : self.seeds,
            "times" : self.time
        }
        parent, name = self.config.file_name.split("_")
        df = pd.DataFrame(data)
        df.to_csv(f'dst/{parent}/{name}.csv',index=False)
        self.clear
        return df

    def developing_save_csv(self):
        data = {
            "trial": [x.trial for x in self.experiment],
            "best_gene": [],
            "best_fitness": [],
            "avg_fitness": [],
            # "avg_best_fitness": [],
            "times": self.time
        }
        for tria_log in self.experiment:
            information = tria_log.information
            match information:
                case Log(best_gene, best_fitness, avg_fitness, _, _, _):
                    data["best_gene"].append(best_gene)
                    data["best_fitness"].append(best_fitness)
                    data["avg_fitness"].append(avg_fitness)
                    # data["avg_best_fitness"].append(tria_log.avg_best_fitness)
                case SimpleLog(best_gene, best_fitness, avg_fitness):
                    data["best_gene"].append(best_gene)
                    data["best_fitness"].append(best_fitness)
                    data["avg_fitness"].append(avg_fitness)
                    # data["avg_best_fitness"].append(tria_log.avg_best_fitness)
                case _:
                    # Handle other classes or raise an error if needed
                    pass

    def advance_run(self):
        for trial in range(self.config.num_trial):
            log = Log(
                best_gene=[],
                best_fitness=[],
                avg_fitness=[],
                first_generation=[],
                last_generation=[],
                generations=[]
            )
            ga = GeneticAlgorithm(self.config)
            population = ga._generate_population()
            best_one = best_gene(population)
            log.best_gene.append(best_one.gene)
            log.best_fitness.append(best_one.phenome)
            log.avg_fitness.append(get_mean(population))
            log.first_generation.append(population)
            log.generations.append(population)

            for i in range(self.config.parameters.max_generations):
                population = ga.evolve(
                    population_origin=population
                )
                new_best = best_gene(population=population)
                if new_best.phenome > best_one.phenome:
                    best_one = new_best
                #log new generation data
                log.best_gene.append(new_best.gene)
                log.best_fitness.append(new_best.phenome)
                log.avg_fitness.append(get_mean(population))
                log.generations.append(population)

                if i + 1 == self.config.parameters.max_generations:
                    log.last_generation.append(population)
            
            self.experiment.append(TriaLog(trial+1, log))
            
            ## Save the log of each trials
    
    # def simple_run(self):
    #     """Run GA experiments and log data."""
    #     for trial in range(self.config.num_trial):
    #         # clear the log at the beginning of each trial
    #         log = Log(
    #             best_gene=[],
    #             best_fitness=[],
    #             avg_fitness=[],
    #             first_generation=[],
    #             last_generation=[],
    #             generations=[]
    #         )
    #         ga = GeneticAlgorithm(self.config)
    #         last_pop, best_gene = ga.forward()
    #         log.best_gene.append(best_gene.gene)
    #         log.best_fitness.append(best_gene.phenome)
    #         log.first_generation.append(ga.population)
    #         log.last_generation.append(last_pop)
    #         self.experiment.append(TriaLog(trial+1, log))
    

    # def save_to_file(self, path):
    #     """Save log data to a JSON file."""
    #     filename = f"{path}/{self.experiment_log}.json"
    #     with open(filename, 'w') as f:
    #         # Convert dataclass objects to dictionaries using asdict
    #         json_data = [asdict(trial_log) for trial_log in self.experiment]
    #         json.dump(json_data, f, indent=4)
    

