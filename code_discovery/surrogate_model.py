import random

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor


class SurrogateModel:
    def __init__(self, kappa=2.576, num_perturbations=10):
        self.model = ExtraTreesRegressor(n_estimators=100, min_samples_leaf=5)
        self.history = set()  # to store unique behaviors
        self.feature_vectors = []  # will hold the feature vectors
        self.fitness_values = []  # will hold the fitness values
        self.best_fitness = -np.inf  # Track the best fitness observed so far
        self.kappa = kappa
        self.num_perturbations = num_perturbations  # Number of perturbations to perform

    def _generate_feature_vector(self, behavior_list):
        """Generate the feature vector from the pairwise similarities."""
        num_cases = SURROGATE_CASE
        feature_vector = []

        # Compute pairwise signed differences
        for i in range(num_cases):
            for j in range(i + 1, num_cases):
                diff_a = behavior_list[i] - behavior_list[j]
                feature_vector.append(diff_a)

        return np.array(feature_vector + feature_vector[num_cases:])

    def add_individual(self, pop):
        """Add an individual to the surrogate model if unique."""
        for ind in pop:
            fitness = ind.fitness.wvalues[0]
            behavior = ind.behavior
            if behavior not in self.history:
                self.history.add(behavior)
                self.feature_vectors.append(self._generate_feature_vector(behavior))
                self.fitness_values.append(fitness)
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
        self.train()  # Train the model after adding new individuals

    def train(self):
        """Train the random forest model."""
        X = np.array(self.feature_vectors)
        y = np.array(self.fitness_values)
        print("X shape:", X.shape)
        print("Number of training samples:", len(X), "Number of features:", X.shape[1])
        ratio = 0.8
        X_train, X_test = X[: int(len(X) * ratio)], X[int(len(X) * ratio) :]
        y_train, y_test = y[: int(len(y) * ratio)], y[int(len(y) * ratio) :]
        self.model.fit(X_train, y_train)

        r2_score = self.model.score(X_test, y_test)
        train_spearman = spearmanr(y_train, self.model.predict(X_train))[0]
        spearman = spearmanr(y_test, self.model.predict(X_test))[0]
        self.score = spearman
        print(
            f"Training Spearman Correlation: {train_spearman}",
            f"Val score: {r2_score}",
            "Val Spearman Correlation:",
            spearman,
        )
        self.model.fit(X, y)

    def predict(self, ind):
        """Predict the fitness of a new individual, including standard deviation."""
        behavior = ind.behavior
        feature_vector = self._generate_feature_vector(behavior)

        # Get mean prediction and std deviation from Random Forest
        mean_prediction = self.model.predict([feature_vector])[0]
        ind.surrogate_fitness = mean_prediction
        std_deviation = np.std(
            [tree.predict([feature_vector])[0] for tree in self.model.estimators_]
        )

        return mean_prediction, std_deviation

    def acquisition(self, ind):
        """Upper Confidence Bound acquisition: mu + kappa * sigma."""
        mu, sigma = self.predict(ind)
        # return mu + self.kappa * sigma
        return mu

    def select(self, pop, num):
        """Select top-`num` individuals by UCB acquisition."""
        if self.score <= 0:
            return random.sample(pop, num)
        scores = np.array([self.acquisition(ind) for ind in pop])
        idx = np.argsort(scores)[-num:]
        return [pop[i] for i in idx]

    def perturb_and_predict(self, ind, noise_scale=0.05):
        """
        Apply perturbations to the behavior vector and predict the fitness over multiple iterations.
        Noise is added to the behavior vector to simulate variations and test robustness.
        The prediction is averaged over multiple iterations.
        """
        behavior = np.array(ind.behavior)
        perturbed_predictions = []

        # Generate multiple perturbations and make predictions
        for _ in range(self.num_perturbations):
            # Add random noise (perturbation) to the behavior vector
            perturbed_behavior = behavior + np.random.normal(
                scale=noise_scale, size=behavior.shape
            )
            feature_vector = self._generate_feature_vector(perturbed_behavior)

            # Predict fitness for the perturbed behavior
            perturbed_predictions.append(self.model.predict([feature_vector])[0])

        # Average the predictions to get the perturbed fitness
        perturbed_fitness = np.mean(perturbed_predictions)
        ind.perturbed_fitness = perturbed_fitness  # Assign to the individual

        return perturbed_fitness


def generate_test_pairs(n, shape=(10, 10)):
    """Generate n pairs of random matrices of given shape."""
    rng = np.random.default_rng(0)  # Seed set to 0
    return [(rng.random(size=shape), rng.random(size=shape)) for _ in range(n)]


SURROGATE_CASE = 5
test_pairs = generate_test_pairs(SURROGATE_CASE, shape=(50, 50))
