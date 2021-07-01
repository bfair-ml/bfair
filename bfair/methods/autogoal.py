from re import search
from typing import Callable, List

from bfair.metrics import build_oracle_output_matrix, disagreement, diversity

from autogoal.kb import Supervised, VectorCategorical
from autogoal.ml import AutoML
from autogoal.search import PESearch


class AutoGoalMitigator:
    def __init__(self, *, input, n_classifiers=None, maximize=True):
        self.n_classifiers = n_classifiers
        self.maximize = maximize

        input = (
            input + (Supervised[VectorCategorical],)
            if isinstance(input, tuple)
            else (input, Supervised[VectorCategorical])
        )

        self._automl = AutoML(
            input=input,
            output=VectorCategorical,
            search_algorithm=PESearch,
            number_of_pipelines=n_classifiers,
            maximize=maximize,
        )

    def __call__(self, X, y):
        classifiers = self._build_base_classifiers(X, y)
        ensembler = self._build_ensembler(X, y, classifiers)
        return ensembler

    def _build_base_classifiers(self, X, y):
        automl = self._automl
        ranking_fn = self._make_ranking_fn(X, y)

        automl.fit(X, y, ranking_fn=ranking_fn)
        return automl.top_pipelines_

    def _make_ranking_fn(self, X, y) -> Callable[[List, List[float]], List[float]]:
        def ranking_fn(solutions, fns):
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
                    y_pred = pipeline.run(X, None)

                    predictions.append(y_pred)
                    valid_solutions.append(pipeline)

                    if (
                        best_fn is None
                        or (fn > best_fn and self.maximize)
                        or (fn < best_fn and not self.maximize)
                    ):
                        best = i
                        best_fn = fn
                except Exception:
                    ranking[i] = -1

            # COMPUTE DIVERSITY BETWEEN PAIRS OF PIPELINES
            oracle_matrix = build_oracle_output_matrix(y, predictions)
            diversity_matrix = disagreement(oracle_matrix)

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

    def _build_ensembler(self, X, y, classifiers):
        pass
