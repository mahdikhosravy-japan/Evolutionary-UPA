
import pygad
import numpy as np
import torch
from ga.model import predict


def constrained_fitness_func(ga_instance, solution, solution_idx, pixel_std, model, input_batch, original_label):

    perturbation = torch.tensor(solution).float().reshape(input_batch.shape[1:]) # [channel, height, width] (3*224*224) instead of (64*3*224*224)

    # Apply pixel constraints
    perturbation = apply_pixel_constraints(perturbation, pixel_std)

    perturbed_input = input_batch + perturbation.unsqueeze(0) # Unsqueeze to add the batch dimension to apply to the entire batch
    perturbed_input = torch.clamp(perturbed_input, 0, 1) # Ensure the pixel values are between 0 and 1

    # Get predictions after applying the perturbation
    prediction = predict(model, perturbed_input)

    # Calculate fitness based on misclassification likelihood (maximise misclassification)
    misclassification_score = (prediction != original_label).float().mean().item() # Get the mean of misclassification
    # print(f"Misclassification score: {misclassification_score}")

    ########
    # OBJECTIVE
    ########
    # Combine misclassification score and perturbation size
    # Calculate perturbation size
    # perturbation_magnitude = torch.norm(perturbation).item()
    # print(f"Perturbation magnitude: {perturbation_magnitude}")

    # fitness_double_objective = misclassification_score - config["fitness"]["perturbation_weight"] * perturbation_magnitude

    # Single objective fitness
    fitness_single_objective = misclassification_score

    return fitness_single_objective



def apply_pixel_constraints(perturbation, pixel_std):
    # Make sure the perturbation is within the pixel standard deviation
    lower_bound = -pixel_std
    upper_bound = pixel_std
    perturbation = torch.clamp(perturbation, lower_bound, upper_bound)
    return perturbation


# def calculate_snr(original_image, perturbed_image):
