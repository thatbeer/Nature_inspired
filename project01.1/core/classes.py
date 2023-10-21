import torch.nn as nn
from datetime import datetime

from core import mutate, multi_mutate, inversion, scramble, no_crossover, ordered_crossover, point_crossover, partialmap_crossover, replace_firstweak, replace_weakest
from core.utils import load_metrics
from core.selection import create_chromosome, parents_selection, generate_population, best_gene

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
    

# class Speculator:
#     # TODO : make the class to gathering the data while running the experiments in text file
#     logg = [100,200,500,1000]

#     def __init__(self, config) -> None:
#         self.config = config
#         self.log = []

#     def run(self):
#         for _ in self.config.num_trial:
#             experiment = GeneticAlgorithm(self.config)
            
class Speculator:
    logg = [100, 200, 500, 1000]

    def __init__(self, config) -> None:
        self.config = config
        self.log = []

    def log_data(self, data):
        """Append data to the log list."""
        self.log.append(data)

    def save_to_file(self, filename="experiment_log.txt"):
        """Save log data to a text file."""
        with open(filename, 'a') as f:
            for line in self.log:
                f.write(line + "\n")
        # Clear log for next run
        self.log = []

    def run(self):
        """Run GA experiments and log data."""
        for trial in range(self.config.num_trial):
            experiment = GeneticAlgorithm(self.config)
            population, best_individual = experiment.forward()

            # This is a simple example of logging data.
            # You can modify and expand it based on your requirements.
            data = f"Trial {trial + 1}: Best Fitness: {best_individual.phenome}"
            self.log_data(data)

            # Save to file if the trial is in self.logg
            if (trial + 1) in self.logg:
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"experiment_log_{timestamp}.txt"
                self.save_to_file(filename)

