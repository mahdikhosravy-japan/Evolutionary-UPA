import pygad
import numpy as np
import torch
from ga.fitness import fitness_func
from ga.model import predict_with_perturbation
from nn.models.googlenet import create_googlenet

model = create_googlenet()
input_tensor = torch.randn(1, 3, 224, 224)  # Random example
original_label = 123


def fitness_wrapper(ga_instance, solution, solution_idx):
    return fitness_func(
        ga_instance, solution, solution_idx, model, input_tensor, original_label
    )


ga_instance = pygad.GA(
    num_generations=10,
    num_parents_mating=2,
    fitness_func=fitness_wrapper,
    sol_per_pop=10,
    num_genes=224 * 224 * 3,
    gene_type=float,
    init_range_low=-0.1,
    init_range_high=0.1,
)

ga_instance.run()

# After the run
solution, solution_fitness, _ = ga_instance.best_solution()
perturbation = torch.tensor(solution).float().reshape(input_tensor.shape)
print(f"Best solution fitness: {solution_fitness}")
