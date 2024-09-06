
import pygad
import numpy as np
import torch
from ga.model import predict_with_perturbation


def fitness_func(ga_instance, solution, solution_idx, model, input_tensor, original_label):

    perturbation = torch.tensor(solution).float().reshape(input_tensor.shape)

    # Get predictions after applying the perturbation
    prediction = predict_with_perturbation(model, input_tensor, perturbation)

    print(f"Prediction: {prediction}, original label: {original_label}")

    # Calculate fitness based on misclassification likelihood (maximise misclassification)
    # predicted_label = prediction.argmax(dim=1)
    misclassification_score = 1.0 if prediction != original_label else 0.0
    print(f"Misclassification score: {misclassification_score}")

    # Minimize perturbation size
    perturbation_magnitude = torch.norm(perturbation)

    # Combine misclassification score and perturbation size
    fitness_double_objective = misclassification_score - 0.01 * perturbation_magnitude

    # Single objective fitness
    fitness_single_objective = 1 - misclassification_score

    return fitness_single_objective
    

