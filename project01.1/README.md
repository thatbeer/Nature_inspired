
# Genetic Algorithm Documentation

This repository contains implementations related to genetic algorithms for solving optimization problems and tracking the experiment for individual report for Nature-inspired class.


## Experimental notebook

### mut_exp.ipynb
- the notebook is used to run experiment to understand and get the insight of how each mutation operator affect the model performance

### co_exp.ipynb
- the notebook is used to run experiment to understand and get the insight of how each crossover operator affect the model performance

### params_exp.ipynb
- the notebook is used to run experiment to understand and the insight of how different parameters affect the model performance.

### performance_exp.ipynb
- the notebook is also used to run experiemnt to understand how diffrent parameters affect the model performance with utilize function to run experiment and visualize the illustration.



## File Overview and Function Descriptions

### ./run.py
- **Function**
  - `run_genetic_algorithm(config)`: perform a genetic algorithm tracker(Speculator) which will performs a genetic algorithm N trials and collect the data into csv file and also return dataframe if it called by the function, also can be called via commandline with `python run.py {yaml file path}`

### ./run_sole.py
- **Function**
  - `search_sole(config)`: perform a genetic algorithm tracker(Speculator) which will performs a genetic algorithm and collect the generation's data into csv file and also return dataframe if it called by the function, also can be called via commandline with `python run_sole.py {yaml file path}


## @ `core` directory
### core/dataclass.py

- Contains various data classes to structure and manage components and configurations for the genetic algorithm.

### core/classes.py

- **Function**:
  - `load_fn(fn_name: str) -> Callable`: Loads specific functions based on their names.
- **Classes**:
  - `GeneticAlgorithm`: Represents the genetic algorithm with various attributes and methods.
  it also presents the function to operate evolve from generation to the next generation and forward function that return the result of the whole genetic algorithm.
  - `SoleExp`: Represents the Experimental logger class to track the performance of each genetic generations. and save the data into csv file.
  - `Speculator`: Represents the Experimental logger class to track the performance of the trial that performs the each genetic algorithm and save the data into csv file.

### core/crossover.py

- **Functions**:
  - `no_crossover(parent1: Chromosome, parent2: Chromosome) -> List[List[int]]`: Returns the two parents without crossover.
  - `point_crossover(parent1: Chromosome, parent2: Chromosome, p_crossover: float = 1.0) -> List[List[int]]`: Implements point crossover with fixed position
  - `partialmap_crossover(parent1: Chromosome, parent2: Chromosome, p_crossover: float = 1.0) -> List[List[int]]`: Performs segment-based crossover.
  - `ordered_crossover(parent1: Chormosome, parent2: Chromosome, p_crossover:float = 1.0) -> List[List[int]]` : Performs ordered-based crossover.

### core/mutation.py

- **Functions**:
  - `mutate(candidate: List[int], p_mutate: float = 1.0) -> List[int]`: Performs basic mutation by swapping two genes.
  - `inversion(candidate: List[int], p_mutate: float=1.0) -> List[int]`: Implements inversion mutation which reverses the range's order of the list
  - `scramble(candidate: List[int], p_mutate: float = 1.0) -> List[int]`: Applies scramble mutation with shuffle the range of candidate gene
  - `multi_mutate(candidate:List[int], p_mutate: float=1.0, num_mutate: int = 2) -> List[int]`: Applies basic mutation multiple times.

### core/replacement.py
- **Functions**:
  - `replace_firstweak(population:List[Chromosome], candidate:Chromosome, inplace=False) -> List[Chromosome]`: Replaces the first weaker candidate.
  - `replace_weakest(population:List[Chromosome], candidate:Chromosome, inplace=False) -> List[Chromosome]`: Replaces the weakest candidate.

### core/selection.py

- **Functions**:
  - `create_gene(distance_metric:List[List[int]]) -> List[int]`: Generates a random gene based on the distance_metric
  - `create_chromosome(gene:List[int], distance_metric:List[List[int]]) -> Chromosome`: Creates a chromosome based on a gene and distance metric in Chromosome class
  - `generate_population(population_size:int, distance_metric:List[List[int]]) -> List[Chromosome]`: Produces a population of chromosomes.
  - `best_gene(population:List[Chromosome]) -> Chromosome`: Returns the best chromosome.
  - `get_mean(population:List[Chromosome]) -> float`: Computes average fitness of the population.
  - `tournament_selection(population:List[Chromosome], tournament_size:int) -> Chromosome`: Selects a chromosome via tournament selection strategy
  - `parents_selection(population:List[Chromosome], tournament_size:int) -> [Chromosome,Chromosome]`: Selects two parents using tournament selection.

### core/utils.py

- **Functions**:
  - `load_metrics(xml_path, info=False) -> List[List[int]]`: Loads weight metrics from an XML file.
  - `cost_function(gene:List[int],distance_metric:List[List[int]]) -> float`: Computes total distance traveled.
  - `fitness_function(gene:List[int], distance_metric:List[List[int]]) -> float`: Calculates fitness based on inverse cost.
  - `pop_stats(population:List[List[int]]) -> (float,(int,float))`: Returns statistical values for a population.
  - `find_combinations(...) -> List[Tuple[Any, ...]]`: Returns all combinations of provided parameters.
  - `load_yaml(path) -> Config`: Loads configuration details from a YAML file.

### core/search.py
- contains the utilization function to run the genetic algorithm experiment for maximum 1000 generation, 3000 generation, and 5000 generation with the visualization function.

## @ `conf` directory
-  contains yaml configuration files for logging with SoleExp and Speculator class. visit `conf/exp1.template.yaml` to observe how the conf is structred/

## @dest directory
- contians the result from performing the SoleExp which tracks the performance of the genetic algorithm generations in 1 trial.

## @dst directory
- contians the result from performing the Speculator which tracks the performance of the genetic algorithm in multiple trials.

## @data directory
-  contians the dataset of the path's city.


---
