import pygad
import numpy as np
import torch
from ga.fitness import fitness_func
from ga.model import load_model, preprocess_image
# from nn.models.googlenet import create_googlenet


########
# MODEL
########

model = load_model()
# input_tensor = torch.randn(1, 3, 224, 224)  # Random example
# original_label = 123


########
# IMAGE
########

input_image_path = "nn/data/imagenet/val/n01440764/ILSVRC2012_val_00000293.JPEG"
input_tensor = preprocess_image(input_image_path)
original_label = 0



########
# GA CONFIG
########

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
