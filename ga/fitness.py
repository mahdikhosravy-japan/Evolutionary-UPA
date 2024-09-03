
import pygad
import numpy as np


def fitness(solution, solution_idx):
    return np.sum(solution)

