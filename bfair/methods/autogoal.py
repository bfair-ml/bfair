from re import search

from autogoal.kb import VectorCategorical
from autogoal.ml import AutoML
from autogoal.search import PESearch


class AutoGoalMitigator:
    def __init__(self, *, n_classifiers=None):
        self.n_classifiers = n_classifiers

        self._automl = AutoML(
            output=VectorCategorical,
            search_algorithm=PESearch,
            number_of_pipelines=n_classifiers,
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

    def _make_ranking_fn(self, X, y):
        pass

    def _build_ensembler(self, X, y, classifiers):
        pass
