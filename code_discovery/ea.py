from code_discovery.analysis.log_plot_tool import LogUtil
from sklearn.preprocessing import normalize
import numpy as np
from deap import tools, gp
from sklearn.cluster import KMeans

from code_discovery.archive import SurrogateHOF, ParetoFrontHOF
from code_discovery.surrogate_model import SurrogateModel, test_pairs
from typing import Optional
from search_engine.offline_rerank import process_and_search

import random


def varAnd(population, toolbox, cxpb, mutpb):
    offspring = [toolbox.clone(ind) for ind in population]

    # Apply crossover and mutation on the offspring
    for i in range(1, len(offspring), 2):
        if random.random() < cxpb:
            # Perform crossover
            offspring[i - 1], offspring[i] = toolbox.mate(
                offspring[i - 1], offspring[i]
            )

            offspring[i - 1].parent_fitness = (
                offspring[i - 1].fitness.wvalues[0],
                offspring[i].fitness.wvalues[0],
            )
            offspring[i].parent_fitness = (
                offspring[i].fitness.wvalues[0],
                offspring[i - 1].fitness.wvalues[0],
            )

            # Remove fitness values to trigger reevaluation
            del offspring[i - 1].fitness.values
            del offspring[i].fitness.values

    for i in range(len(offspring)):
        if random.random() < mutpb:
            (offspring[i],) = toolbox.mutate(offspring[i])

            # Only apply mutation if parent fitness has not been assigned yet
            if not hasattr(offspring[i], "parent_fitness"):
                offspring[i].parent_fitness = offspring[
                    i
                ].fitness.values  # Record fitness if not already assigned

            # Remove fitness values to trigger reevaluation
            del offspring[i].fitness.values

    return offspring


# Number of discriminative instances to select
INSTANCE_SELECTION = 10


def unique_inds(ind_list, history, pset):
    """Filter individuals by uniqueness, adding new ones to history."""
    unique = []
    for ind in ind_list:
        # key = str(ind)
        func = gp.compile(expr=ind, pset=pset)
        behavior = tuple(func(a, b) for (a, b) in test_pairs)

        mean = np.mean(behavior)
        behavior = tuple((b - mean) for b in behavior)  # Centering behavior

        ind.behavior = behavior + (
            np.max(ind.parent_fitness) if hasattr(ind, "parent_fitness") else 0,
            np.mean(ind.parent_fitness) if hasattr(ind, "parent_fitness") else 0,
            np.min(ind.parent_fitness) if hasattr(ind, "parent_fitness") else 0,
            len(ind),
            ind.height,
        )
        if behavior not in history:
            history.add(behavior)
            unique.append(ind)
        else:
            pass
    return unique


def cluster_instances(
    semantics_matrix: np.ndarray, n_clusters: int, random_state: int = 0
) -> np.ndarray:
    # Transpose to shape (n_instances, n_features) and normalize each instance to unit norm
    inst_matrix = semantics_matrix.T
    inst_norm = normalize(inst_matrix, axis=1)

    # KMeans in Euclidean on normalized data approximates cosine clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(inst_norm)
    return labels


def select_discriminative_instances(
    semantics_matrix: np.ndarray, n_clusters: int
) -> np.ndarray:
    """
    Cluster instances and select discriminative ones.
    """
    # Compute variance for each instance (across features)
    vars_ = np.var(semantics_matrix, axis=0)

    # Filter out zero-variance instances (no variability across features)
    valid_idx = np.where(vars_ > 0)[0]
    filtered_sems = semantics_matrix[:, valid_idx]  # Filtered semantics matrix

    # Perform clustering on the filtered instances
    labels = cluster_instances(filtered_sems, n_clusters)

    selected = []
    for cid in range(n_clusters):
        # Get the indices of instances in the current cluster, map back to original indices
        idx = valid_idx[np.where(labels == cid)[0]]
        if idx.size == 0:
            continue

        # Compute variance per instance
        vars_ = np.var(semantics_matrix[:, idx], axis=0)
        # Select the instance with the highest variance
        sel = idx[np.argmax(vars_)]

        # In case of ties, select the one with the lowest mean
        ties = idx[vars_ == vars_.max()]
        if ties.size > 1:
            means = np.mean(semantics_matrix[:, ties], axis=0)
            sel = ties[np.argmin(means)]

        selected.append(int(sel))

    return np.array(sorted(selected))


def select_hardest_instances(semantics_matrix: np.ndarray, n_select: int) -> np.ndarray:
    # Compute variance for each instance (across features)
    vars_ = np.var(semantics_matrix, axis=0)

    # Filter out zero-variance instances (no variability across features)
    valid_idx = np.where(vars_ > 0)[0]
    filtered_sems = semantics_matrix[:, valid_idx]  # Filtered semantics matrix

    # Perform clustering on the filtered instances
    labels = cluster_instances(filtered_sems, n_select)

    selected = []
    for cid in range(n_select):
        # Get the indices of instances in the current cluster
        idx = valid_idx[np.where(labels == cid)[0]]  # Map back to the original indices
        if idx.size == 0:
            continue

        # Select the 'hardest' representative based on the lowest variance
        sel = idx[np.argmin(np.mean(semantics_matrix[:, idx], axis=0))]
        selected.append(int(sel))
    return np.array(sorted(selected))


class Config:
    def __init__(self, selection_interval=0, surrogate_ratio=0, **kwargs):
        # gens between reclustering
        self.selection_interval = selection_interval
        self.strategy = "Var"
        self.surrogate_ratio = surrogate_ratio
        print("Surrogate ratio set to", self.surrogate_ratio)


def get_test_score(engine, dataset, topk=1, fusion_similarity_fn=None):
    """Run test set through process_and_search and return recall score."""
    return process_and_search(
        engine, dataset, topk=topk, fusion_similarity_fn=fusion_similarity_fn
    )


def eaSimple(
    population,
    toolbox,
    cxpb,
    mutpb,
    ngen,
    stats=None,
    hof=None,
    verbose=__debug__,
    config: Optional[Config] = None,
    config_dict: dict = None,
):
    # Initialize logbook
    n_pop = len(population)
    logbook = tools.Logbook()
    log_util = LogUtil(config_dict)

    # Assign header fields to logbook
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])  # type: ignore

    # Initialize persistent history set
    if not hasattr(eaSimple, "history"):
        eaSimple.history = set()

    # Full set of instances
    total_instances = 50
    full_idx = np.arange(total_instances)

    cluster_every = config.selection_interval if config else None

    for ind in population:
        ind.instances = full_idx

    # Helper to evaluate and record new individuals
    def eval_and_record(ind_list):
        invalid = [ind for ind in ind_list if not ind.fitness.valid]
        fits = toolbox.map(toolbox.evaluate, invalid)
        for ind, (fit_vals, semantics) in zip(invalid, fits):
            ind.fitness.values = fit_vals
            ind.semantics = np.array(semantics)
        return len(invalid)

    population = unique_inds(population, eaSimple.history, toolbox.pset)
    nevals = eval_and_record(population)
    if config and config.surrogate_ratio > 0:
        surrogate_model = SurrogateModel()
        surrogate_model.add_individual(population)
        hof.surrogate_model = surrogate_model
    if not isinstance(hof, SurrogateHOF):
        # no surrogate fitness for the first generation
        hof.update(population) if hof is not None else None
    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=nevals, **record)
    log_util.log_metrics(population, hof)  # Log the fitness for the initial generation
    if verbose:
        print(logbook.stream)

    # Initial surrogate instance selection
    if cluster_every:
        sem_mat = np.vstack([ind.semantics for ind in population])
        np.save("sem_mat.npy", sem_mat)
        if config and config.strategy == "Hardest":  # type: ignore
            selected = select_hardest_instances(sem_mat, INSTANCE_SELECTION)
        else:
            selected = select_discriminative_instances(sem_mat, INSTANCE_SELECTION)
        for ind in population:
            ind.instances = selected
        print("Finished instance selection, selected instances:", selected)

    # Evolutionary loop
    for gen in range(1, ngen + 1):
        offspring = []
        to_generate = n_pop
        if config and config.surrogate_ratio > 1:
            to_generate = int(n_pop * config.surrogate_ratio)

        while len(offspring) < to_generate:
            parents = toolbox.select(population, 2)
            children = varAnd(parents, toolbox, cxpb, mutpb)
            new_ch = unique_inds(children, eaSimple.history, toolbox.pset)
            needed = to_generate - len(offspring)
            offspring.extend(new_ch[:needed])

        if config and config.surrogate_ratio > 1:
            offspring = surrogate_model.select(offspring, n_pop)
            assert len(offspring) == n_pop, (
                "Surrogate selection did not return the expected number of offspring."
            )

        if cluster_every:
            for ind in offspring + population:
                ind.instances = full_idx if gen % cluster_every == 0 else selected

        # Reevaluate entire population instead of using hall-of-fame
        if cluster_every and gen % cluster_every in (0, 1):
            if gen % cluster_every == 0:
                # re-evaluate all individuals
                for ind in population:
                    del ind.fitness.values
                _ = eval_and_record(population)
            else:
                for ind in population:
                    ind.fitness.values = (
                        np.mean([ind.semantics[idx] for idx in selected]),
                        ind.fitness.values[1],
                        ind.fitness.values[2],
                    )
        nevals = eval_and_record(offspring)
        if isinstance(hof, ParetoFrontHOF):
            hof.update(list(offspring) + list(population)) if hof is not None else None
        else:
            hof.update(offspring) if hof is not None else None
        if config and config.surrogate_ratio > 0:
            # update surrogate model after hall of fame update
            surrogate_model.add_individual(offspring)

        # NSGA-II replacement
        population[:] = tools.selNSGA2(population + offspring, k=n_pop)

        # Log stats
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=nevals, **record)
        log_util.log_metrics(population, hof)  # Log the fitness after every generation
        if verbose:
            print(logbook.stream)

        if cluster_every and gen % cluster_every == 0:
            sem_mat = np.vstack([ind.semantics for ind in population])
            if config and config.strategy == "Hardest":  # type: ignore
                sel_idx = select_hardest_instances(sem_mat, INSTANCE_SELECTION)
            else:
                sel_idx = select_discriminative_instances(sem_mat, INSTANCE_SELECTION)
            for ind in population:
                ind.instances = sel_idx
            print("Finished instance selection, selected instances:", sel_idx)
    return population, logbook
