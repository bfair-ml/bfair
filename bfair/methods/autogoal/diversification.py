import random
from typing import Any, Callable, List

import numpy as np
from bfair.metrics import build_oracle_output_matrix, double_fault_inverse
from bfair.utils.autogoal import join_input, split_input

from autogoal.kb import Supervised, VectorCategorical
from autogoal.ml import AutoML
from autogoal.search import PESearch
from autogoal.ml.metrics import supervised_fitness_fn


class AutoGoalDiversifier:
    def __init__(
        self,
        *,
        input,
        n_classifiers: int,
        score_metric: Callable[[Any, Any], float],
        diversity_metric=double_fault_inverse,
        maximize=True,
        validation_split=0.3,
        ranking_fn: Callable[[List, List[float]], List[float]] = None,
        **automl_kwargs,
    ):
        self.n_classifiers = n_classifiers
        self.score_metric = score_metric
        self.diversity_metric = diversity_metric
        self.maximize = maximize
        self.validation_split = validation_split
        self.ranking_fn = ranking_fn

        input = (
            input + (Supervised[VectorCategorical],)
            if isinstance(input, tuple)
            else (input, Supervised[VectorCategorical])
        )

        self._automl = AutoML(
            input=input,
            output=VectorCategorical,
            number_of_pipelines=n_classifiers,
            maximize=maximize,
            validation_split=validation_split,
            score_metric=supervised_fitness_fn(score_metric),
            **automl_kwargs,
        )

    @property
    def search_parameters(self):
        return self._automl.search_kwargs

    @property
    def search_iterations(self):
        return self._automl.search_iterations

    def __call__(self, X, y, *, test_on=None, **run_kwargs):
        return self._build_base_classifiers(X, y, test_on, **run_kwargs)

    def _build_base_classifiers(self, X, y, test_on=None, **run_kwargs):
        automl = self._automl

        if self.ranking_fn is None:
            if test_on is None:
                X_train, y_train, (X_test, y_test) = split_input(
                    X,
                    y,
                    self.validation_split,
                )
            else:
                X_train, y_train = X, y
                X_test, y_test = test_on
                X, y = join_input(X_train, y_train, X_test, y_test)

            ranking_fn = self.make_ranking_fn(
                X_train,
                y_train,
                X_test,
                y_test,
                diversity_metric=self.diversity_metric,
                top_cut=self.n_classifiers,
                maximize=self.maximize,
            )
        else:
            ranking_fn = self.ranking_fn

        automl.fit(X, y, ranking_fn=ranking_fn, **run_kwargs)
        return automl.top_pipelines_, automl.top_pipelines_scores_

    @staticmethod
    def make_ranking_fn(
        X,
        y,
        X_test,
        y_test,
        *,
        diversity_metric=double_fault_inverse,
        top_cut=None,
        maximize=True,
    ) -> Callable[[List, List[float]], List[float]]:
        def ranking_fn(solutions, fns):
            if top_cut is not None and len(solutions) <= top_cut:
                return fns

            n_solutions = len(solutions)
            ranking = [None] * len(solutions)

            # COMPUTE PREDICTIONS AND FIND TOP PERFORMING PIPELINE'S INDEX
            best = None
            best_fn = None
            predictions = []
            valid_solutions = []

            for i in range(n_solutions):
                pipeline = solutions[i]
                fn = fns[i]

                try:
                    pipeline.send("train")
                    pipeline.run(X, y)
                    pipeline.send("eval")
                    y_pred = pipeline.run(X_test, None)

                    predictions.append(y_pred)
                    valid_solutions.append(pipeline)

                    if (
                        best_fn is None
                        or (fn > best_fn and maximize)
                        or (fn < best_fn and not maximize)
                    ):
                        best = i
                        best_fn = fn
                except Exception:
                    ranking[i] = -1

            if (
                top_cut is not None
                and len(valid_solutions) <= top_cut
                or len(valid_solutions) < 3
            ):
                return fns

            # COMPUTE DIVERSITY BETWEEN PAIRS OF PIPELINES
            oracle_matrix = build_oracle_output_matrix(y_test, predictions)
            diversity_matrix = diversity_metric(oracle_matrix)

            # RANK PIPELINES GREEDY (root: best_fn, step: diversity)
            # - accuracy is been ignored
            n_valid_solutions = len(valid_solutions)
            ranking[best] = n_valid_solutions
            selected = [best]

            def compute_aggregated_diversity(pipeline_index):
                return diversity_matrix[selected, pipeline_index].sum()

            for rank in range(n_valid_solutions - 1, 1, -1):
                best = None
                best_diversity = float("-inf")
                for i in range(n_solutions):
                    if ranking[i] is not None:
                        continue
                    diversity = compute_aggregated_diversity(i)
                    if diversity > best_diversity:
                        best = i
                        best_diversity = diversity
                ranking[best] = rank
                selected.append(best)

            last = next(i for i in range(n_solutions) if ranking[i] is None)
            ranking[last] = 1

            return ranking

        return ranking_fn


def build_random_ranking_fn():
    def ranking_fn(solutions, fns):
        ranking = list(range(len(solutions)))
        random.shuffle(ranking)
        return ranking

    return ranking_fn


def build_random_but_single_best_ranking_fn(maximize):
    def ranking_fn(solutions, fns):
        n_solutions = len(solutions)

        ranking = list(range(1, n_solutions))  # one element is missing
        random.shuffle(ranking)

        indexes = np.argsort(fns if maximize else -np.asarray(fns))
        best_index = indexes[-1]
        ranking.insert(best_index, n_solutions)

        return ranking

    return ranking_fn


def build_best_performance_ranking_fd(maximize):
    def ranking_fn(solutions, fns):
        indexes = np.argsort(fns if maximize else -np.asarray(fns))

        ranking = [None] * len(solutions)
        for i, solution_index in enumerate(indexes):
            ranking[solution_index] = i
        return ranking

    return ranking_fn
