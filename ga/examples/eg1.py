import pygad
import numpy as np


def fitness_func(ga_instance, solution, solution_idx):
    return np.sum(solution)


num_genes = 5

ga_instance = pygad.GA(
    num_generations=100,
    num_parents_mating=2,
    fitness_func=fitness_func,
    sol_per_pop=10,
    num_genes=num_genes,
    gene_type=int,
    init_range_low=0,
    init_range_high=10,
)

ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"Best solution : {solution}")
print(f"Best solution fitness : {solution_fitness}")
