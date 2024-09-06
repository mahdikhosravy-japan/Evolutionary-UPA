import pygad
import numpy as np
import torch
import yaml
from ga.fitness import fitness_func
from ga.model import load_model, preprocess_image
# from nn.models.googlenet import create_googlenet

########
# CONFIG
########
def load_config(config_file="ga/config.yaml"):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config

config = load_config()


########
# MODEL
########

model = load_model(config["model"]["model_type"])
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
    num_generations=config["ga"]["num_generations"],
    num_parents_mating=config["ga"]["num_parents_mating"],
    sol_per_pop=config["ga"]["sol_per_pop"],
    num_genes=input_tensor.numel(),
    gene_type=float,
    init_range_low=config["ga"]["init_range_low"],
    init_range_high=config["ga"]["init_range_high"],
    mutation_percent_genes=config["ga"]["mutation_percent_genes"],
    fitness_func=fitness_wrapper
)

ga_instance.run()

# After the run
solution, solution_fitness, _ = ga_instance.best_solution()
perturbation = torch.tensor(solution).float().reshape(input_tensor.shape)
print(f"Best solution fitness: {solution_fitness}")
