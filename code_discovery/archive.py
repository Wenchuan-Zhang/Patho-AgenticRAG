import copy
from _bisect import bisect_right
from _operator import eq


from deap import tools, gp
import numpy as np
from deap.tools import HallOfFame


from code_discovery.surrogate_model import SurrogateModel
from search_engine.offline_rerank import process_and_search


class CustomHOF(HallOfFame):
    def __init__(
        self, maxsize, comparison_function, key_metric, similar=eq, preprocess=None
    ):
        self.preprocess = preprocess
        self.comparison_function = comparison_function
        self.key_metric = key_metric
        super().__init__(maxsize, similar)

    def update(self, population):
        if self.preprocess is not None:
            self.preprocess(population)
        for ind in population:
            if len(self) == 0 and self.maxsize != 0:
                # Working on an empty hall of fame is problematic for the
                # "for else"
                self.insert(population[0])
                continue
            if self.comparison_function(ind, self[-1]) or len(self) < self.maxsize:
                for hofer in self:
                    # Loop through the hall of fame to check for any
                    # similar individual
                    if self.similar(ind, hofer):
                        break
                else:
                    # The individual is unique and strictly better than
                    # the worst
                    if len(self) >= self.maxsize:
                        self.remove(-1)
                    self.insert(ind)

    def insert(self, item):
        item = copy.deepcopy(item)
        i = bisect_right(self.keys, self.key_metric(item))
        self.items.insert(len(self) - i, item)
        self.keys.insert(i, self.key_metric(item))


class ParetoFrontHOF(HallOfFame):
    def __init__(self, maxsize, engine=None, dataset=None, pset=None):
        super().__init__(maxsize)
        self.engine = engine
        self.dataset = dataset
        self.pset = pset
        self.best_ind = None

    def calculate_harmonic_rank(self, ranks):
        """
        Calculate the harmonic mean of the ranks based on the first two objectives.
        """
        return np.mean([1 / (rank + 1) for rank in ranks])

    def update(self, population):
        # Perform non-dominated sorting using DEAP's sortNondominated
        fronts = tools.sortNondominated(
            population, len(population), first_front_only=False
        )

        best_individual = None
        best_harmonic_rank = float("inf")

        # Iterate through the sorted fronts (each front contains non-dominated individuals)
        for front in fronts[0]:  # Only the first front is Pareto optimal
            # Rank individuals based on wvalues[0] and wvalues[1] (larger is better)
            ranks_0 = sorted(
                range(len(front)), key=lambda i: front[i].wvalues[0], reverse=True
            )
            ranks_1 = sorted(
                range(len(front)), key=lambda i: front[i].wvalues[1], reverse=True
            )

            # Combine the ranks of both objectives: sum of ranks for each individual
            combined_ranks = [
                ranks_0.index(i) + ranks_1.index(i) for i in range(len(front))
            ]

            # Calculate the harmonic rank based on the combined ranks
            harmonic_rank = self.calculate_harmonic_rank(combined_ranks)

            for i, ind in enumerate(front):
                # Select the individual with the best harmonic rank
                if harmonic_rank < best_harmonic_rank:
                    best_harmonic_rank = harmonic_rank
                    best_individual = ind

        # Update the Hall of Fame with the best individual based on the harmonic rank
        if best_individual:
            self.items.clear()  # Clear the Hall of Fame
            self.insert(best_individual)  # Add the best individual

            # If the best individual has changed, evaluate and store its fitness
            if self.best_ind != best_individual:
                fusion_similarity_fn = gp.compile(best_individual, self.pset)
                score = process_and_search(
                    self.engine,
                    self.dataset,
                    topk=1,
                    fusion_similarity_fn=fusion_similarity_fn,
                )
                self.best_test_score = score
                self.best_ind = best_individual

            print("╔═ Hall of Fame Top Individual ═╗")
            print(f"║  Surrogate fitness : {best_individual.surrogate_fitness:.6f}")
            print(f"║  True fitness      : {best_individual.fitness.values}")
            print(f"║  Test fitness      : {self.best_test_score}")
            print(f"║  Tree     : {self.best_ind!s}")
            print("╚══════════════════════════════╝")


class EasyHOF(tools.HallOfFame):
    def __init__(self, maxsize, engine, dataset, pset):
        super().__init__(maxsize)

        self.engine = engine
        self.dataset = dataset
        self.pset = pset

        self.best_ind = None
        self.best_test_score = None

    def update(self, population):
        # Let the base HallOfFame update its internal list
        super().update(population)

        if len(self) == 0:
            return

        top = self[0]
        # If the champion has changed, re-evaluate its test score
        if top is not self.best_ind:
            # Compile the GP tree into a callable function
            fusion_similarity_fn = gp.compile(top, self.pset)

            # Run your external evaluation/search to get the true test score
            score = process_and_search(
                self.engine,
                self.dataset,
                topk=1,
                fusion_similarity_fn=fusion_similarity_fn,
            )
            self.best_test_score = score
            self.best_ind = top

        # Print current champ’s stats
        print("╔═ Hall of Fame Top Individual ═╗")
        print(f"║  True fitness      : {top.fitness.values}")
        print(f"║  Test fitness      : {self.best_test_score}")
        print(f"║  Tree     : {self.best_ind!s}")
        print("╚══════════════════════════════╝")


class SurrogateHOF(CustomHOF):
    def __init__(self, maxsize, engine=None, dataset=None, pset=None):
        metric = lambda x: x.surrogate_fitness
        comparison_func = lambda x, y: x.surrogate_fitness > y.surrogate_fitness
        super().__init__(maxsize, comparison_func, metric)
        self.engine = engine
        self.dataset = dataset
        self.surrogate_model: SurrogateModel = None
        self.pset = pset
        self.best_ind = None

    def update(self, population):
        # update
        if len(self) > 0:
            for ind in list(self):
                self.surrogate_model.perturb_and_predict(ind)

        for ind in population:
            self.surrogate_model.perturb_and_predict(ind)

        super().update(population)

        if len(self) > 0:
            top = self[0]
            if self.best_ind != top:
                fusion_similarity_fn = gp.compile(top, self.pset)
                score = process_and_search(
                    self.engine,
                    self.dataset,
                    topk=1,
                    fusion_similarity_fn=fusion_similarity_fn,
                )
                self.best_test_score = score
                self.best_ind = top

            print("╔═ Hall of Fame Top Individual ═╗")
            print(f"║  Surrogate fitness : {top.surrogate_fitness:.6f}")
            print(f"║  True fitness      : {top.fitness.values}")
            print(f"║  Test fitness      : {self.best_test_score}")
            print(f"║  Tree     : {self.best_ind!s}")
            print("╚══════════════════════════════╝")
