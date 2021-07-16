from bfair.utils import ClassifierWrapper

from .diversification import AutoGoalDiversifier
from .ensembling import AutoGoalEnsembler


class AutoGoalMitigator:
    def __init__(self, diversifier, ensembler):
        self.diversifier = diversifier
        self.ensembler = ensembler

    @classmethod
    def build(
        cls,
        *,
        input,
        n_classifiers: int,
        maximize=True,
        include_filter=".*",
        exclude_filter=None,
        registry=None,
        **automl_kwargs
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

        return cls(diversifier, ensembler)

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
        score_metric,
        include_filter=".*",
        exclude_filter=None,
        registry=None,
        **search_kwargs
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

        classifiers = [ClassifierWrapper(p) for p in pipelines]

        ensemble, score = self.ensembler(
            X,
            y,
            classifiers,
            scores,
            test_on=test_on,
            **run_kwargs,
        )
        return ensemble.model
