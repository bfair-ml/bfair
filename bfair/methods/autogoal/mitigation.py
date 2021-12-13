from typing import Callable, Union

from bfair.utils import ClassifierWrapper

from .diversification import AutoGoalDiversifier
from .ensembling import AutoGoalEnsembler


class AutoGoalMitigator:
    def __init__(
        self,
        diversifier: AutoGoalDiversifier,
        ensembler: AutoGoalEnsembler,
        detriment: Union[int, float],
    ):
        if not isinstance(detriment, (int, float)):
            raise TypeError
        if isinstance(detriment, int) and not diversifier.maximize:
            raise ValueError

        self.diversifier = diversifier
        self.ensembler = ensembler
        self.detriment = detriment

    @classmethod
    def build(
        cls,
        *,
        input,
        n_classifiers: int,
        detriment: Union[int, float],
        maximize=True,
        include_filter=".*",
        exclude_filter=None,
        registry=None,
        **automl_kwargs,
    ):
        diversifier = cls.build_diversifier(
            input=input,
            n_classifiers=n_classifiers,
            maximize=maximize,
            include_filter=include_filter,
            exclude_filter=exclude_filter,
            registry=registry,
            **automl_kwargs,
        )

        score_metric = diversifier.score_metric
        search_kwargs = diversifier.search_parameters

        ensembler = cls.build_ensembler(
            score_metric=score_metric,
            include_filter=include_filter,
            exclude_filter=exclude_filter,
            registry=registry,
            **search_kwargs,
        )

        return cls(diversifier, ensembler, detriment)

    @classmethod
    def build_diversifier(
        cls, *, input, n_classifiers: int, maximize=True, **automl_kwargs
    ):
        return AutoGoalDiversifier(
            input=input,
            n_classifiers=n_classifiers,
            maximize=maximize,
            **automl_kwargs,
        )

    @classmethod
    def build_ensembler(
        cls,
        *,
        score_metric: Callable,
        include_filter=".*",
        exclude_filter=None,
        registry=None,
        **search_kwargs,
    ):
        return AutoGoalEnsembler(
            score_metric=score_metric,
            include_filter=include_filter,
            exclude_filter=exclude_filter,
            registry=registry,
            **search_kwargs,
        )

    def __call__(self, X, y, *, test_on=None, **run_kwargs):
        pipelines, scores = self.diversifier(
            X,
            y,
            test_on=test_on,
            **run_kwargs,
        )

        classifiers = ClassifierWrapper.wrap_and_fit(pipelines, X, y)

        constraint = self._build_constraint_fn(
            scores,
            X,
            y,
            test_on=test_on,
            detriment=self.detriment,
            score_metric=self.diversifier.score_metric,
            maximize_scores=self.diversifier.maximize,
        )

        ensemble, score = self.ensembler(
            X,
            y,
            classifiers,
            scores,
            test_on=test_on,
            generations=self.diversifier.search_iterations,
            constraint=constraint,
            **run_kwargs,
        )
        return ensemble.model

    def _build_constraint_fn(
        self,
        scores,
        X,
        y,
        *,
        test_on,
        detriment,
        score_metric,
        maximize_scores,
    ):
        best_score = max(scores) if maximize_scores else min(scores)
        measure_of_detriment = (
            (lambda score: 1 - score / best_score <= detriment / 100)
            if isinstance(detriment, int)
            else (lambda score: best_score - score <= detriment)
            if maximize_scores
            else (lambda score: score - best_score <= detriment)
        )
        X_test, y_test = (X, y) if test_on is None else test_on

        def constraint(generated, disparity_fn):
            ensemble = generated.model
            y_pred = ensemble.predict(X_test)
            score = score_metric(y_test, y_pred)
            return measure_of_detriment(score)

        return constraint
