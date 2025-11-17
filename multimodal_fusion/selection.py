import numpy as np
from numba import jit


@jit(nopython=True)
def lexicase_selection(fitness_values, k):
    """
    Lexicase selection using Numba for optimization.

    :param fitness_values: A 2D NumPy array where rows are individuals and columns are fitness cases.
    :param k: The number of individuals to select.
    :returns: A list of indices of selected individuals.
    """
    n_individuals, n_cases = fitness_values.shape
    selected_indices = []

    for _ in range(k):
        candidates = np.arange(n_individuals)  # Start with all individuals
        cases = np.arange(n_cases)  # All fitness cases
        np.random.shuffle(cases)  # Shuffle the fitness cases

        for case in cases:
            if len(candidates) <= 1:
                break
            # Find the best fitness value for this case (f = max)
            best_value = np.max(fitness_values[candidates, case])

            # Filter candidates based on the best value for this case
            candidates = candidates[fitness_values[candidates, case] == best_value]

        # Randomly choose one individual from the remaining candidates
        selected_indices.append(np.random.choice(candidates))

    return selected_indices


def selLexicase(individuals, k):
    """Returns an individual that does the best on the fitness cases when considered one at a time in random order."""

    # Extract the fitness values from the individuals
    n_individuals = len(individuals)
    n_cases = len(individuals[0].semantics)

    fitness_values = np.zeros((n_individuals, n_cases))

    for i, individual in enumerate(individuals):
        fitness_values[i] = individual.semantics

    # Use Numba-optimized lexicase selection
    selected_indices = lexicase_selection(fitness_values, k)

    # Return selected individuals based on their indices
    selected_individuals = [individuals[i] for i in selected_indices]

    return selected_individuals


