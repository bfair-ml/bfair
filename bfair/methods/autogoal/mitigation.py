from typing import Any, Callable, Dict, List, Union

import pandas as pd
from bfair.methods.voting import stack_predictions
from bfair.metrics import base_metric
from bfair.metrics.diversity import double_fault_inverse
from bfair.utils import ClassifierWrapper
from bfair.utils.autogoal import split_input
from pandas import DataFrame, Series

from .diversification import AutoGoalDiversifier
from .ensembling import AutoGoalEnsembler


class AutoGoalMitigator:
    def __init__(
        self,
        diversifier: AutoGoalDiversifier,
        ensembler: AutoGoalEnsembler,
        detriment: Union[int, float, None],
        *,
        validation_split: float = None,
    ):
        """
        If `detriment` is negative, then the score should be improved in at least `abs(detriment)` units.
        """
        if detriment is not None and not isinstance(detriment, (int, float)):
            raise TypeError
        if isinstance(detriment, int) and not diversifier.maximize:
            raise ValueError

        self.diversifier = diversifier
        self.ensembler = ensembler
        self.detriment = detriment
        self.validation_split = validation_split

    @property
    def score_metric(self):
        return self.diversifier.score_metric

    @property
    def fairness_metric(self):
        return self.ensembler.score_metric

    @classmethod
    def build(
        cls,
        *,
        input,
        n_classifiers: int,
        detriment: Union[int, float],
        score_metric: Callable[[Any, Any], float],
        diversity_metric=double_fault_inverse,
        fairness_metrics: Union[base_metric, List[base_metric]] = None,
        maximize_fmetric: Union[bool, List[bool]] = False,
        protected_attributes: Union[List[str], str] = None,
        target_attribute: str = None,
        positive_target=None,
        metric_kwargs: Dict[str, object] = None,
        sensor: Callable[..., DataFrame] = None,
        maximize=True,
        validation_split=0.3,
        ranking_fn: Callable[[List, List[float]], List[float]] = None,
        include_filter=".*",
        exclude_filter=None,
        registry=None,
        **automl_kwargs,
    ):
        diversifier = cls.build_diversifier(
            input=input,
            n_classifiers=n_classifiers,
            score_metric=score_metric,
            diversity_metric=diversity_metric,
            maximize=maximize,
            validation_split=validation_split,
            ranking_fn=ranking_fn,
            include_filter=include_filter,
            exclude_filter=exclude_filter,
            registry=registry,
            **automl_kwargs,
        )

        search_kwargs = diversifier.search_parameters

        if fairness_metrics is not None:
            del search_kwargs["maximize"]

            if isinstance(maximize_fmetric, bool):
                maximize_fmetric = [maximize_fmetric] * len(fairness_metrics)

        second_phase_score_metric = (
            cls.build_fairness_fn(
                fairness_metrics,
                protected_attributes,
                target_attribute,
                positive_target,
                metric_kwargs,
                sensor,
                score_metric=score_metric,
            )
            if fairness_metrics is not None
            else (lambda X, y, y_pred: score_metric(y, y_pred))
        )

        ensembler = cls.build_ensembler(
            score_metric=second_phase_score_metric,
            maximize=maximize_fmetric,
            validation_split=validation_split,
            include_filter=include_filter,
            exclude_filter=exclude_filter,
            registry=registry,
            **search_kwargs,
        )

        return cls(diversifier, ensembler, detriment, validation_split=validation_split)

    @classmethod
    def build_diversifier(
        cls,
        *,
        input,
        n_classifiers: int,
        score_metric: Callable[[Any, Any], float],
        diversity_metric=double_fault_inverse,
        maximize=True,
        validation_split=0.3,
        ranking_fn=None,
        **automl_kwargs,
    ):
        return AutoGoalDiversifier(
            input=input,
            n_classifiers=n_classifiers,
            diversity_metric=diversity_metric,
            maximize=maximize,
            validation_split=validation_split,
            score_metric=score_metric,
            ranking_fn=ranking_fn,
            **automl_kwargs,
        )

    @classmethod
    def build_ensembler(
        cls,
        *,
        score_metric: Callable[[Any, Any, Any], float],
        maximize: List[bool],
        validation_split=0.3,
        include_filter=".*",
        exclude_filter=None,
        registry=None,
        **search_kwargs,
    ):
        return AutoGoalEnsembler(
            score_metric=score_metric,
            maximize=maximize,
            validation_split=validation_split,
            include_filter=include_filter,
            exclude_filter=exclude_filter,
            registry=registry,
            **search_kwargs,
        )

    @classmethod
    def build_fairness_fn(
        cls,
        fairness_metrics: Union[base_metric, List[base_metric]],
        protected_attributes: Union[List[str], str],
        target_attribute: str,
        positive_target,
        metric_kwargs: Dict[str, object] = None,
        sensor: Callable[..., DataFrame] = None,
        score_metric: Callable[[Any, Any], float] = None,
        aggregate_fn: Callable[[List[float]], float] = None,
    ) -> Callable[[Any, Any, Any], float]:
        if protected_attributes is None:
            raise ValueError("No protected attributes were provided")
        if target_attribute is None:
            raise ValueError("No target attribute was provided")
        if positive_target is None:
            raise ValueError("Positive target was not provided")
        if metric_kwargs is None:
            metric_kwargs = {}
        if callable(fairness_metrics):
            fairness_metrics = (fairness_metrics,)
        if isinstance(protected_attributes, str):
            protected_attributes = [protected_attributes]

        def fairness_fn(X, y, y_pred):
            if not (len(X) == len(y) == len(y_pred)):
                raise ValueError(
                    f"Input shapes do not match (X[{len(X)}], y[{len(y)}], y_pred[{len(y_pred)}])"
                )
            if sensor is not None:
                X = sensor(X)
            if not isinstance(X, DataFrame):
                X = DataFrame(X, columns=protected_attributes)

            data = pd.concat((X, Series(y, name=target_attribute)), axis=1)
            y_pred = pd.Series(y_pred, data.index)

            evaluations = [
                fairness_metric(
                    data=data,
                    protected_attributes=protected_attributes,
                    target_attribute=target_attribute,
                    target_predictions=y_pred,
                    positive_target=positive_target,
                    **metric_kwargs,
                )
                for fairness_metric in fairness_metrics
            ]

            if score_metric is not None:
                evaluations.append(score_metric(y, y_pred))

            return (
                aggregate_fn(evaluations) if aggregate_fn is not None else evaluations
            )

        return fairness_fn

    def __call__(
        self,
        X,
        y,
        *,
        test_on=None,
        diversifier_run_kwargs,
        ensembler_run_kwargs,
        **run_kwargs,
    ):
        if test_on is None and self.validation_split is not None:
            X, y, test_on = split_input(X, y, self.validation_split)

        pipelines, scores = self.diversify(
            X,
            y,
            test_on=test_on,
            **run_kwargs,
            **diversifier_run_kwargs,
        )
        model, _ = self.ensemble(
            pipelines,
            scores,
            X,
            y,
            test_on=test_on,
            **run_kwargs,
            **ensembler_run_kwargs,
        )
        return model

    def diversify(self, X, y, *, test_on=None, **run_kwargs):
        # should pipelines be trained in the whole training set (with cross validation)? R.\ YES

        pipelines, scores = self.diversifier(
            X,
            y,
            test_on=test_on,
            **run_kwargs,
        )
        return pipelines, scores

    def ensemble(self, pipelines, scores, X, y, *, test_on=None, **run_kwargs):
        # should fmetric be scored on fully trained pipelines? R.\ NO

        if test_on is None:
            X, y, test_on = split_input(X, y, self.ensembler.validation_split)

        classifiers = ClassifierWrapper.wrap_and_fit(pipelines, X, y)

        detriment_constraint = (
            self._build_constraint_fn(
                X,
                y,
                classifiers,
                scores,
                test_on=test_on,
                detriment=self.detriment,
                score_metric=self.diversifier.score_metric,
                maximize_scores=self.diversifier.maximize,
            )
            if self.detriment is not None
            else None
        )

        try:
            user_constraint = run_kwargs["constraint"]
            constraint = lambda solution, fn: (
                user_constraint(solution, fn) and detriment_constraint(solution, fn)
            )
        except KeyError:
            constraint = detriment_constraint

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
        return ensemble.model, score

    def _build_constraint_fn(
        self,
        X,
        y,
        classifiers,
        scores,
        *,
        test_on,
        detriment,
        score_metric,
        maximize_scores,
        pre_caching=True,
    ):
        best_score = max(scores) if maximize_scores else min(scores)

        if isinstance(detriment, int) and detriment > 0:
            detriment = (
                abs(best_score * detriment / 100)
                if best_score != 0
                else detriment / 100
            )

        measure_of_detriment = (
            (lambda score: best_score - score <= detriment)
            if maximize_scores
            else (lambda score: score - best_score <= detriment)
        )
        X_test, y_test = (X, y) if test_on is None else test_on

        e_input = stack_predictions(X, classifiers) if pre_caching else X
        e_input_test = stack_predictions(X_test, classifiers) if pre_caching else X_test

        def constraint(generated, disparity_fn):
            ensemble = generated.model
            indexes = generated.info["indexes"]
            if not ensemble.fitted:
                ensemble.fit(e_input, y, on_predictions=pre_caching, selection=indexes)
            y_pred = ensemble.predict(
                e_input_test,
                on_predictions=pre_caching,
                selection=indexes,
            )
            score = score_metric(y_test, y_pred)
            return measure_of_detriment(score)

        return constraint
