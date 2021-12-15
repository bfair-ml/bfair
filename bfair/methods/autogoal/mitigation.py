from typing import Any, Callable, Dict, List, Union

import pandas as pd
from bfair.metrics import base_metric
from bfair.utils import ClassifierWrapper
from pandas import DataFrame, Series

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
        fairness_metric: base_metric = None,
        maximize_fmetric=False,
        protected_attributes: Union[List[str], str] = None,
        target_attribute: str = None,
        positive_target=None,
        metric_kwargs: Dict[str, object] = None,
        sensor: Callable[..., DataFrame] = None,
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

        search_kwargs = diversifier.search_parameters

        if fairness_metric is not None:
            search_kwargs["maximize"] = maximize_fmetric

        score_metric = (
            cls.build_fairness_fn(
                fairness_metric,
                protected_attributes,
                target_attribute,
                positive_target,
                metric_kwargs,
                sensor,
            )
            if fairness_metric is not None
            else (lambda X, y, y_pred: diversifier.score_metric(y, y_pred))
        )

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
        score_metric: Callable[[Any, Any, Any], float],
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

    @classmethod
    def build_fairness_fn(
        cls,
        fairness_metric: base_metric,
        protected_attributes: Union[List[str], str],
        target_attribute: str,
        positive_target,
        metric_kwargs: Dict[str, object] = None,
        sensor: Callable[..., DataFrame] = None,
    ) -> Callable[[Any, Any, Any], float]:
        if protected_attributes is None:
            raise ValueError("No protected attributes were provided")
        if target_attribute is None:
            raise ValueError("No target attribute was provided")
        if positive_target is None:
            raise ValueError("Positive target was not provided")
        if metric_kwargs is None:
            metric_kwargs = {}

        def fairness_fn(X, y, y_pred):
            if sensor is not None:
                X = sensor(X)
            if not isinstance(X, DataFrame):
                X = DataFrame(X, columns=protected_attributes)

            data = pd.concat((X, Series(y, name=target_attribute)), axis=1)
            y_pred = pd.Series(y_pred, data.index)

            return fairness_metric(
                data=data,
                protected_attributes=protected_attributes,
                target_attribute=target_attribute,
                target_predictions=y_pred,
                positive_target=positive_target,
                **metric_kwargs,
            )

        return fairness_fn

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
