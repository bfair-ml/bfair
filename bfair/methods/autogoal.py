from re import search
from typing import Callable, List

from bfair.metrics import build_oracle_output_matrix, disagreement, diversity

from autogoal.kb import Supervised, VectorCategorical
from autogoal.ml import AutoML
from autogoal.search import PESearch


class AutoGoalMitigator:
    def __init__(self, diversifier, ensembler):
        self.diversifier = diversifier
        self.ensembler = ensembler

    @classmethod
    def build(cls, *, input, n_classifiers=None, maximize=True, **automl_kwargs):
        diversifier = cls.build_diversifier(
            input=input,
            n_classifiers=n_classifiers,
            maximize=maximize,
            **automl_kwargs,
        )
        ensembler = cls.build_ensembler(**automl_kwargs)
        return cls(diversifier, ensembler)

    @classmethod
    def build_diversifier(
        cls, *, input, n_classifiers=None, maximize=True, **automl_kwargs
    ):
        return AutoGoalDiversifier(
            input=input,
            n_classifiers=n_classifiers,
            maximize=maximize,
            **automl_kwargs,
        )

    @classmethod
    def build_ensembler(cls, **automl_kwargs):
        raise NotImplementedError()

    def __call__(self, X, y, **search_kwargs):
        classifiers, scores = self.diversifier(X, y, **search_kwargs)
        model = self.ensembler(X, y, classifiers, scores, **search_kwargs)
        return model


class AutoGoalDiversifier:
    def __init__(self, *, input, n_classifiers=None, maximize=True, **automl_kwargs):
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
            **automl_kwargs,
        )

    def __call__(self, X, y, **search_kwargs):
        return self._build_base_classifiers(X, y, **search_kwargs)

    def _build_base_classifiers(self, X, y, **search_kwargs):
        automl = self._automl
        ranking_fn = self._make_ranking_fn(X, y)

        automl.fit(X, y, ranking_fn=ranking_fn, **search_kwargs)
        return automl.top_pipelines_, automl.top_pipelines_scores_

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
