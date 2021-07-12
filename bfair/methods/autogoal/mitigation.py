from bfair.utils import ClassifierWrapper

from .diversification import AutoGoalDiversifier
from .ensembling import AutoGoalEnsembler


class AutoGoalMitigator:
    def __init__(self, diversifier, ensembler):
        self.diversifier = diversifier
        self.ensembler = ensembler

    @classmethod
    def build(cls, *, input, n_classifiers: int, maximize=True, **automl_kwargs):
        diversifier = cls.build_diversifier(
            input=input,
            n_classifiers=n_classifiers,
            maximize=maximize,
            **automl_kwargs,
        )
        search_kwargs = diversifier.search_parameters
        ensembler = cls.build_ensembler(**search_kwargs)
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
    def build_ensembler(cls, **search_kwargs):
        raise NotImplementedError()

    def __call__(self, X, y, **run_kwargs):
        pipelines, scores = self.diversifier(X, y, **run_kwargs)
        classifiers = [ClassifierWrapper(p) for p in pipelines]
        model = self.ensembler(X, y, classifiers, scores, **run_kwargs)
        return model
